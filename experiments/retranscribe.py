"""
Hybrid live transcription: Vosk (fast) + Whisper (refined in background).

Dependencies:
    pip install vosk pyaudio numpy torch
    pip install openai-whisper   # NOT "whisper"

If you don't have a Vosk model yet:
    python -m vosk --download en-us

Run:
    python hybrid_transcribe.py
"""

import json
import queue
import threading
import time
import sys
import argparse
import os

import numpy as np
import pyaudio
import torch
import vosk
import whisper

# Limit PyTorch CPU threads to a reasonable number (helps on some laptops)
try:
    max_threads = min(4, os.cpu_count() or 4)
    torch.set_num_threads(max_threads)
except Exception:
    pass

# ------------------------------
# Audio parameters
# ------------------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # frames per buffer

# How many seconds of audio per Whisper segment (tunable)
SEGMENT_SECONDS = 3.0  # default, can be overridden via CLI
SEGMENT_CHUNKS = int(SEGMENT_SECONDS * RATE / CHUNK)

# Overlap between Whisper segments (seconds) to avoid cutting words
OVERLAP_SECONDS = 0.5
OVERLAP_SAMPLES = int(OVERLAP_SECONDS * RATE)

# Queues for fan-out from the audio callback
vosk_q: "queue.Queue[bytes]"      # type: ignore
whisper_q: "queue.Queue[bytes]"   # type: ignore

# Event to let Whisper tell Vosk to reset between segments
reset_vosk_event = threading.Event()
# Global handle to the Vosk model so we can recreate recognizers
vosk_model = None  # will be set in main()

# Debounce time (seconds) between Vosk resets to avoid thrashing
VOSK_RESET_DEBOUNCE = 0.3
last_vosk_reset_time = 0.0


# ------------------------------
# Audio callback
# ------------------------------
def audio_callback(in_data, frame_count, time_info, status):
    """
    PyAudio callback: push data into both queues so that
    Vosk and Whisper workers can consume the same audio.
    """
    vosk_q.put(in_data)
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
                    max_len = 80
                    if len(ptext) > max_len:
                        ptext_disp = ptext[:max_len] + "…"
                    else:
                        ptext_disp = ptext
                    # Temporary preview on a single line
                    sys.stdout.write(f"\r[VOSK] {ptext_disp:<80}")
                    sys.stdout.flush()
    except Exception as e:  # noqa: BLE001
        print(f"\n[VOSK worker error]: {e}", file=sys.stderr)


# ------------------------------
# Whisper worker (refined transcript)
# ------------------------------
def whisper_worker(model) -> None:
    """Consume audio chunks, batch into segments, and transcribe with Whisper.

    Whisper outputs are printed as final lines (history). Each time Whisper
    produces a line, it clears the Vosk preview line first.
    """
    buffer = []
    chunks_in_buffer = 0
    overlap_tail_bytes = b""  # keep a small tail to prepend to next segment

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
                        temperature=0.0,
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
                        print(f"[WHISPER] {text}")
                except Exception as e:  # noqa: BLE001
                    print(f"\n[Whisper worker error]: {e}", file=sys.stderr)

    except Exception as e:  # noqa: BLE001
        print(f"\n[Whisper worker outer error]: {e}", file=sys.stderr)


# ------------------------------
# Main
# ------------------------------
def main() -> None:
    global vosk_q, whisper_q, vosk_model, SEGMENT_SECONDS, SEGMENT_CHUNKS

    parser = argparse.ArgumentParser(description="Hybrid Vosk + Whisper transcriber")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=SEGMENT_SECONDS,
        help="Seconds of audio per Whisper segment (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="medium.en",
        help="Whisper model name (e.g., tiny.en, base.en, small.en)",
    )
    args = parser.parse_args()

    # Update global segment config from CLI
    SEGMENT_SECONDS = max(0.5, float(args.segment_seconds))
    SEGMENT_CHUNKS = int(SEGMENT_SECONDS * RATE / CHUNK) or 1

    vosk_q = queue.Queue()
    whisper_q = queue.Queue()

    # Load Vosk model
    print("Loading Vosk model (en-us)...")
    try:
        vosk_model = vosk.Model(lang="en-us")
    except Exception as e:
        print(f"Failed to load Vosk model: {e}")
        return
    recognizer = vosk.KaldiRecognizer(vosk_model, RATE)

    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model '{args.whisper_model}' on {device}...")
    try:
        whisper_model = whisper.load_model(args.whisper_model, device=device)
    except Exception as e:
        print(f"Failed to load Whisper model '{args.whisper_model}': {e}")
        return

    # Initialize PyAudio
    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback,
        )
    except Exception as e:
        print(f"Failed to open audio input: {e}")
        return

    stream.start_stream()

    # Start worker threads
    t_vosk = threading.Thread(target=vosk_worker, args=(recognizer,), daemon=True)
    t_whisper = threading.Thread(
        target=whisper_worker,
        args=(whisper_model,),
        daemon=True,
    )
    t_vosk.start()
    t_whisper.start()

    print("Listening...")
    print("- Vosk: live preview only (no history)")
    print("- Whisper: final refined lines (kept as history)")
    print(
        f"Segment length: {SEGMENT_SECONDS:.2f}s, model: {args.whisper_model}. "
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
        t_vosk.join(timeout=1.0)
        t_whisper.join(timeout=1.0)


if __name__ == "__main__":
    main()