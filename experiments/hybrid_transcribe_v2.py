"""
Hybrid live transcription v2: Vosk (fast preview) + Whisper (refined, final).

Features:
- Vosk:
    - Live preview only (no history, single updating line).
    - Reset between Whisper segments so it doesn't glue sentences together.
    - Configurable preview length; optional ANSI color.

- Whisper:
    - Refined final transcript only (one line per segment, kept as history).
    - Time-based segmentation with overlap to avoid cutting words.
    - Skips low-energy (mostly silent) segments.
    - Configurable model, segment length, backend device.

- CLI options:
    --segment-seconds   Seconds per Whisper segment (default ~2.0)
    --whisper-model     Whisper model name (tiny.en, base.en, small.en, medium.en, etc.)
    --input-device      PyAudio input device index (default: system default)
    --rate              Sample rate (default: 16000)
    --channels          Number of input channels (default: 1)
    --out               Path to append Whisper lines to a text file
    --no-color          Disable ANSI color in terminal

Dependencies:
    pip install vosk pyaudio numpy torch
    pip install openai-whisper    # NOT 'whisper'

If you don't have a Vosk model yet:
    python -m vosk --download en-us

Run (example):
    python hybrid_transcribe_v2.py --segment-seconds 2.0 --whisper-model small.en
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

import numpy as np
import pyaudio
import torch
import vosk
import whisper

# ------------------------------
# Global config defaults
# ------------------------------

# Audio defaults (can be overridden by CLI)
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK = 1024

# Whisper segmentation defaults
DEFAULT_SEGMENT_SECONDS = 2.0  # shorter = snappier final lines
DEFAULT_OVERLAP_SECONDS = 0.5  # overlap to avoid cutting words

# Whisper model default
DEFAULT_WHISPER_MODEL = "small.en"  # good balance of speed/accuracy on CPU

# Vosk behavior
VOSK_PREVIEW_CHARS = 80           # max characters to display for preview
VOSK_RESET_DEBOUNCE = 0.3         # seconds between resets

# Silence threshold for Whisper (mean absolute amplitude)
SILENCE_THRESHOLD = 1e-3

# Torch CPU thread limit (helps on laptops)
try:
    max_threads = min(4, os.cpu_count() or 4)
    torch.set_num_threads(max_threads)
except Exception:
    pass

# ------------------------------
# Runtime globals (set in main)
# ------------------------------

FORMAT = pyaudio.paInt16
RATE = DEFAULT_RATE
CHANNELS = DEFAULT_CHANNELS
CHUNK = DEFAULT_CHUNK

SEGMENT_SECONDS = DEFAULT_SEGMENT_SECONDS
SEGMENT_CHUNKS = 0
OVERLAP_SECONDS = DEFAULT_OVERLAP_SECONDS
OVERLAP_SAMPLES = 0

vosk_q: "queue.Queue[bytes]"
whisper_q: "queue.Queue[bytes]"

reset_vosk_event = threading.Event()
vosk_model: vosk.Model | None = None
last_vosk_reset_time = 0.0

# Enable/disable components (set from CLI)
USE_VOSK = True
USE_WHISPER = True

# Color tags (can be disabled with --no-color)
VOSK_TAG = "[VOSK]"
WHISPER_TAG = "[WHISPER]"


# ------------------------------
# Audio callback
# ------------------------------
def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback: push data into the queues for enabled components."""
    if USE_VOSK:
        vosk_q.put(in_data)
    if USE_WHISPER:
        whisper_q.put(in_data)
    return (None, pyaudio.paContinue)


# ------------------------------
# Vosk worker (fast live feed)
# ------------------------------
def vosk_worker(recognizer: vosk.KaldiRecognizer) -> None:
    """Consume audio chunks and produce fast partial results with Vosk.

    IMPORTANT:
    - Only partial results are shown.
    - No final lines are printed, so Vosk does not create history.
    - When reset_vosk_event is set (by Whisper), the recognizer is
      re-created so Vosk only reflects *new* audio after a Whisper
      segment boundary.
    """
    global vosk_model, last_vosk_reset_time

    try:
        while True:
            data = vosk_q.get()
            if data is None:  # shutdown signal
                break

            # If Whisper signaled a reset, re-create the recognizer so
            # we don't keep extending old sentences. Debounce to avoid
            # thrashing if segments arrive very quickly.
            if reset_vosk_event.is_set():
                now = time.time()
                if now - last_vosk_reset_time >= VOSK_RESET_DEBOUNCE:
                    if vosk_model is not None:
                        recognizer = vosk.KaldiRecognizer(vosk_model, RATE)
                        # Clear any lingering state defensively
                        try:
                            _ = recognizer.Result()
                        except Exception:
                            pass
                    last_vosk_reset_time = now
                    # Also clear any stale preview on the terminal.
                    sys.stdout.write("\r" + " " * 120 + "\r")
                    sys.stdout.flush()
                reset_vosk_event.clear()

            if len(data) == 0:
                continue

            if recognizer.AcceptWaveform(data):
                # Intentionally ignore Vosk final results.
                _ = recognizer.Result()  # consume but do nothing
            else:
                # Partial Vosk result (inline, no newline)
                partial = json.loads(recognizer.PartialResult())
                ptext = partial.get("partial", "").strip()
                if ptext:
                    # Truncate preview to avoid overly long lines
                    if len(ptext) > VOSK_PREVIEW_CHARS:
                        ptext_disp = ptext[:VOSK_PREVIEW_CHARS] + "…"
                    else:
                        ptext_disp = ptext
                    # Temporary preview on a single line
                    line = f"\r{VOSK_TAG} {ptext_disp:<{VOSK_PREVIEW_CHARS}}"
                    sys.stdout.write(line)
                    sys.stdout.flush()
    except Exception as e:  # noqa: BLE001
        print(f"\n[VOSK worker error]: {e}", file=sys.stderr)


# ------------------------------
# Whisper worker (refined transcript)
# ------------------------------
def whisper_worker(model, out_path: str | None = None) -> None:
    """Consume audio chunks, batch into segments, and transcribe with Whisper.

    Whisper outputs are printed as final lines (history). Each time Whisper
    produces a line, it clears the Vosk preview line first. Optionally
    appends final lines to an output text file.
    """
    buffer: list[bytes] = []
    chunks_in_buffer = 0
    overlap_tail_bytes = b""  # keep a small tail to prepend to next segment

    out_file = None
    if out_path:
        try:
            out_file = open(out_path, "a", encoding="utf-8")
        except Exception as e:
            print(f"Could not open output file '{out_path}': {e}", file=sys.stderr)
            out_file = None

    try:
        while True:
            data = whisper_q.get()
            if data is None:  # shutdown signal
                break

            buffer.append(data)
            chunks_in_buffer += 1

            # Once we have ~SEGMENT_SECONDS of audio, transcribe
            if chunks_in_buffer >= SEGMENT_CHUNKS:
                segment_bytes = b"".join(buffer)
                buffer.clear()
                chunks_in_buffer = 0

                # Prepend overlap tail from previous segment, if any
                if overlap_tail_bytes:
                    segment_bytes = overlap_tail_bytes + segment_bytes

                # Convert int16 bytes -> float32 [-1, 1]
                audio_np = (
                    np.frombuffer(segment_bytes, dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )

                # Skip mostly-silent segments
                if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                    overlap_tail_bytes = b""
                    continue

                # Prepare overlap_tail_bytes for the next segment
                try:
                    if len(audio_np) > OVERLAP_SAMPLES:
                        tail = audio_np[-OVERLAP_SAMPLES:]
                    else:
                        tail = audio_np
                    tail_int16 = np.clip(tail * 32768.0, -32768, 32767).astype(
                        np.int16
                    )
                    overlap_tail_bytes = tail_int16.tobytes()
                except Exception:
                    overlap_tail_bytes = b""

                # Whisper transcription tuned for speed / determinism
                try:
                    result = model.transcribe(
                        audio_np,
                        language="en",
                        task="transcribe",
                        # Use a temperature schedule so Whisper can fall back
                        # to sampling if greedy decoding looks degenerate
                        # (e.g., lots of repeated words).
                        temperature=(0.0, 0.2, 0.4),
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False,
                        fp16=torch.cuda.is_available(),
                    )
                    text = result.get("text", "").strip()
                    if text:
                        # Signal Vosk to reset so future previews only reflect
                        # new audio after this Whisper segment.
                        reset_vosk_event.set()
                        # Clear Vosk preview line so it doesn't stay in history
                        sys.stdout.write("\r" + " " * 120 + "\r")
                        sys.stdout.flush()
                        # Print final Whisper line as history
                        line = f"{WHISPER_TAG} {text}"
                        print(line)
                        # Optionally write to file
                        if out_file is not None:
                            out_file.write(text + "\n")
                            out_file.flush()
                except Exception as e:  # noqa: BLE001
                    print(f"\n[Whisper worker error]: {e}", file=sys.stderr)

    except Exception as e:  # noqa: BLE001
        print(f"\n[Whisper worker outer error]: {e}", file=sys.stderr)
    finally:
        if out_file is not None:
            try:
                out_file.close()
            except Exception:
                pass


# ------------------------------
# Main
# ------------------------------
def main() -> None:
    global RATE, CHANNELS, CHUNK
    global SEGMENT_SECONDS, SEGMENT_CHUNKS, OVERLAP_SECONDS, OVERLAP_SAMPLES
    global vosk_q, whisper_q, vosk_model, VOSK_TAG, WHISPER_TAG

    parser = argparse.ArgumentParser(description="Hybrid Vosk + Whisper transcriber (v2)")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Seconds of audio per Whisper segment (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help="Whisper model name (e.g., tiny.en, base.en, small.en, medium.en)",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="PyAudio input device index (default: system default)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_RATE,
        help="Sample rate (default: %(default)s)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help="Number of input channels (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to append final Whisper lines to a text file",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color in terminal output",
    )
    parser.add_argument(
        "--no-vosk",
        action="store_true",
        help="Disable Vosk live preview",
    )
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable Whisper final transcription",
    )

    args = parser.parse_args()

    global USE_VOSK, USE_WHISPER
    USE_VOSK = not args.no_vosk
    USE_WHISPER = not args.no_whisper

    if not USE_VOSK and not USE_WHISPER:
        print("Nothing to do: both Vosk and Whisper are disabled (use at least one).")
        return

    # Apply audio settings
    RATE = args.rate
    CHANNELS = args.channels
    CHUNK = DEFAULT_CHUNK  # keep buffer size constant unless you want it configurable

    # Segment config
    SEGMENT_SECONDS = max(0.5, float(args.segment_seconds))
    SEGMENT_CHUNKS = int(SEGMENT_SECONDS * RATE / CHUNK) or 1
    OVERLAP_SECONDS = DEFAULT_OVERLAP_SECONDS
    OVERLAP_SAMPLES = int(OVERLAP_SECONDS * RATE)

    # Color tags
    if args.no_color:
        VOSK_TAG = "[VOSK]"
        WHISPER_TAG = "[WHISPER]"
    else:
        VOSK_TAG = "\x1b[36m[VOSK]\x1b[0m"
        WHISPER_TAG = "\x1b[32m[WHISPER]\x1b[0m"

    vosk_q = queue.Queue()
    whisper_q = queue.Queue()

    recognizer = None
    if USE_VOSK:
        # Load Vosk model
        print("Loading Vosk model (en-us)...")
        try:
            vosk_model = vosk.Model(lang="en-us")
        except Exception as e:
            print(f"Failed to load Vosk model: {e}")
            return
        recognizer = vosk.KaldiRecognizer(vosk_model, RATE)

    whisper_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if USE_WHISPER:
        # Load Whisper model
        print(f"Loading Whisper model '{args.whisper_model}' on {device}...")
        try:
            whisper_model = whisper.load_model(args.whisper_model, device=device)
        except Exception as e:
            print(f"Failed to load Whisper model '{args.whisper_model}': {e}")
            return

    # Initialize PyAudio
    try:
        pa = pyaudio.PyAudio()
        stream_kwargs = dict(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback,
        )
        if args.input_device is not None:
            stream_kwargs["input_device_index"] = args.input_device
        stream = pa.open(**stream_kwargs)
    except Exception as e:
        print(f"Failed to open audio input: {e}")
        return

    stream.start_stream()

    # Start worker threads (only for enabled components)
    t_vosk = None
    t_whisper = None

    if USE_VOSK and recognizer is not None:
        t_vosk = threading.Thread(target=vosk_worker, args=(recognizer,), daemon=True)
        t_vosk.start()

    if USE_WHISPER and whisper_model is not None:
        t_whisper = threading.Thread(
            target=whisper_worker,
            args=(whisper_model, args.out),
            daemon=True,
        )
        t_whisper.start()

    print("Listening...")
    if USE_VOSK:
        print(f"- Vosk: live preview only (no history), device: {args.input_device}")
    if USE_WHISPER:
        print("- Whisper: final refined lines (kept as history)")
    print(
        f"Segment length: {SEGMENT_SECONDS:.2f}s, "
        f"overlap: {OVERLAP_SECONDS:.2f}s, model: {args.whisper_model}. "
        "Press Ctrl+C to stop.\n"
    )

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping hybrid transcription...")
    finally:
        # Signal workers to stop
        vosk_q.put(None)
        whisper_q.put(None)

        # Stop/close audio
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass

        # Join threads (they are daemon, but we clean up anyway)
        if t_vosk is not None:
            t_vosk.join(timeout=1.0)
        if t_whisper is not None:
            t_whisper.join(timeout=1.0)


if __name__ == "__main__":
    main()