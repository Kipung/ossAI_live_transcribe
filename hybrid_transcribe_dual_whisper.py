"""
Dual-Whisper live transcription:
- Fast Whisper model: quick, low-latency segments (FAST).
- Slow Whisper model: higher-accuracy refinement (REF) over same segments.

Architecture:
- PyAudio -> raw audio queue
- Segmenter thread:
    - collects audio into fixed-length *fast* segments (e.g., 2.0s)
    - assigns segment_id = 1, 2, 3, ...
    - pushes (segment_id, segment_bytes) into both fast and slow queues
- Slow worker:
    - groups multiple fast segments until it reaches a larger window
      (e.g., 6.0s) and refines the combined audio as one chunk

Notes:
- The terminal view is managed with ANSI clear + redraw:
  refined segments appear as history lines, while all current fast
  segments are shown as a single concatenated live line.

Dependencies:
    pip install pyaudio numpy torch
    pip install openai-whisper  # NOT 'whisper'

Run (example):
    python hybrid_transcribe_dual_whisper.py \
        --fast-model tiny.en \
        --slow-model small.en \
        --fast-segment-seconds 1.5 \
        --slow-segment-seconds 6.0 

On Windows, use caret (^) for line continuation:
    python hybrid_transcribe_dual_whisper.py ^
        --fast-model tiny.en ^
        --slow-model small.en ^
        --fast-segment-seconds 1.5 ^
        --slow-segment-seconds 6.0 ^
        --device-index 1

"""

import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import pyaudio
import torch
import whisper
from typing import Dict, Any

# ------------------------------
# Defaults
# ------------------------------

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK = 1024

DEFAULT_FAST_SEGMENT_SECONDS = 2.0
DEFAULT_SLOW_SEGMENT_SECONDS = 6.0

DEFAULT_FAST_MODEL = "tiny.en"
DEFAULT_SLOW_MODEL = "small.en"

SILENCE_THRESHOLD = 1e-3  # skip near-silent segments

# Limit Torch threads for laptop friendliness
try:
    max_threads = min(4, os.cpu_count() or 4)
    torch.set_num_threads(max_threads)
except Exception:
    pass

# ------------------------------
# Runtime globals
# ------------------------------

FORMAT = pyaudio.paInt16
RATE = DEFAULT_RATE
CHANNELS = DEFAULT_CHANNELS
CHUNK = DEFAULT_CHUNK

FAST_SEGMENT_SECONDS = DEFAULT_FAST_SEGMENT_SECONDS
FAST_SEGMENT_FRAMES = 0  # set in main
SLOW_SEGMENT_SECONDS = DEFAULT_SLOW_SEGMENT_SECONDS

# Queues
audio_q: "queue.Queue[bytes]"
fast_q: "queue.Queue[tuple[int, bytes]]"
slow_q: "queue.Queue[tuple[int, bytes]]"

# Segment index & transcript bookkeeping
segment_counter_lock = threading.Lock()
next_segment_id = 1

# Transcript storage: segment_id -> {'text': str, 'refined': bool}
transcripts_lock = threading.Lock()
transcripts: Dict[int, Dict[str, Any]] = {}

# For nice output
FAST_TAG = "[FAST]"
SLOW_TAG = "[REF] "


# ------------------------------
# Rendering
# ------------------------------
def render_transcript() -> None:
    """Clear the screen and redraw the current transcript.

    - Refined segments (from the slow model) are treated as history lines.
    - All unrefined segments (from the fast model) are combined into a
      single live line by concatenating their texts in segment-id order.
    """
    with transcripts_lock:
        items = sorted(transcripts.items(), key=lambda kv: kv[0])

        refined_lines = []
        live_segments = []  # list[(seg_id, text)]

        for seg_id, entry in items:
            refined = bool(entry.get("refined", False))
            text = entry.get("text", "")
            if refined:
                refined_lines.append((seg_id, text))
            else:
                live_segments.append((seg_id, text))

    # ANSI clear screen and move cursor to home
    sys.stdout.write("\x1b[2J\x1b[H")

    # Print history (refined) lines
    for seg_id, text in refined_lines:
        line = f"[#{seg_id} {SLOW_TAG}] {text}"
        sys.stdout.write(line + "\n")

    # Combine all live (fast) segments into a single live line
    if live_segments:
        live_segments_sorted = sorted(live_segments, key=lambda x: x[0])
        # Use the highest seg_id among live segments for the live line tag
        live_seg_id = live_segments_sorted[-1][0]
        parts = [t for _, t in live_segments_sorted if t]
        live_text = " ".join(parts).strip()
        if live_text:
            live = f"[#{live_seg_id} {FAST_TAG}] {live_text}"
            sys.stdout.write(live + "\n")

    sys.stdout.flush()


# ------------------------------
# Audio callback
# ------------------------------
def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback: push mic audio into a raw audio queue."""
    audio_q.put(in_data)
    return (None, pyaudio.paContinue)


# ------------------------------
# Segmenter worker
# ------------------------------
def segmenter_worker() -> None:
    """
    Read raw audio frames from audio_q, accumulate into segments of
    FAST_SEGMENT_SECONDS, and send each segment as (segment_id, bytes) to
    both fast_q and slow_q.
    """
    global next_segment_id

    buffer: list[bytes] = []
    frames_in_buffer = 0

    frames_per_segment = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1

    try:
        while True:
            data = audio_q.get()
            if data is None:  # shutdown signal
                break

            buffer.append(data)
            frames_in_buffer += 1

            if frames_in_buffer >= frames_per_segment:
                segment_bytes = b"".join(buffer)
                buffer.clear()
                frames_in_buffer = 0

                # Compute segment id
                with segment_counter_lock:
                    seg_id = next_segment_id
                    next_segment_id += 1

                # Send to both fast and slow workers
                fast_q.put((seg_id, segment_bytes))
                slow_q.put((seg_id, segment_bytes))

    except Exception as e:  # noqa: BLE001
        print(f"\n[Segmenter worker error]: {e}", file=sys.stderr)


# ------------------------------
# Fast Whisper worker
# ------------------------------
def fast_worker(model, device: str) -> None:
    """
    Fast Whisper worker:
    - Small model.
    - Lower latency.
    - Appends transcript lines as soon as possible.
    """
    try:
        while True:
            item = fast_q.get()
            if item is None:
                break
            seg_id, segment_bytes = item

            # bytes -> float32 [-1, 1]
            audio_np = (
                np.frombuffer(segment_bytes, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            # Skip mostly-silent segments
            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                continue

            try:
                result = model.transcribe(
                    audio_np,
                    language="en",
                    task="transcribe",
                    temperature=0.0,
                    condition_on_previous_text=False,
                    fp16=(device == "cuda"),
                )
                text = result.get("text", "").strip()
                if text:
                    # Remove mid-sentence punctuation to avoid premature periods
                    # Keep only apostrophes and basic characters
                    cleaned = text.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
                    cleaned = cleaned.strip()

                    if cleaned:
                        # Only update if this segment has not yet been refined.
                        with transcripts_lock:
                            entry = transcripts.get(seg_id)
                            if entry is None or not entry.get("refined", False):
                                transcripts[seg_id] = {"text": cleaned, "refined": False}
                        render_transcript()
            except Exception as e:  # noqa: BLE001
                print(f"\n[FAST worker error]: {e}", file=sys.stderr)

    except Exception as e:  # noqa: BLE001
        print(f"\n[FAST worker outer error]: {e}", file=sys.stderr)


# ------------------------------
# Slow Whisper worker
# ------------------------------
def slow_worker(model, device: str) -> None:
    """
    Slow Whisper worker:
    - Larger model.
    - Higher accuracy.
    - Processes the same segments (same ids) and prints refined lines.
    """
    per_fast_duration = FAST_SEGMENT_SECONDS
    group_bytes: list[bytes] = []
    group_ids: list[int] = []
    group_duration = 0.0

    try:
        while True:
            item = slow_q.get()
            if item is None:
                break
            seg_id, segment_bytes = item

            # Add this fast segment to the current slow-group window
            group_bytes.append(segment_bytes)
            group_ids.append(seg_id)
            group_duration += per_fast_duration

            # If we have not yet reached the slow window size, keep accumulating
            if group_duration < SLOW_SEGMENT_SECONDS:
                continue

            # We reached or exceeded the slow window; combine and transcribe
            combined_bytes = b"".join(group_bytes)
            # Keep a copy of the IDs in this group before resetting
            ids_for_group = group_ids[:]

            # Reset for the next group
            group_bytes = []
            group_ids = []
            group_duration = 0.0

            audio_np = (
                np.frombuffer(combined_bytes, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            # Skip mostly-silent groups
            if np.abs(audio_np).mean() < SILENCE_THRESHOLD or not ids_for_group:
                continue

            try:
                result = model.transcribe(
                    audio_np,
                    language="en",
                    task="transcribe",
                    # Use a temperature schedule with repetition safeguards
                    temperature=(0.0, 0.2, 0.4),
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    fp16=(device == "cuda"),
                )
                text = result.get("text", "").strip()
                if text:
                    # Use the last segment id in this group as the canonical id
                    final_seg_id = ids_for_group[-1]
                    # Mark this group as refined and overwrite text.
                    with transcripts_lock:
                        # Remove intermediate fast-only ids from transcripts
                        for gid in ids_for_group:
                            if gid != final_seg_id:
                                transcripts.pop(gid, None)
                        transcripts[final_seg_id] = {"text": text, "refined": True}
                    render_transcript()
            except Exception as e:  # noqa: BLE001
                print(f"\n[SLOW worker error]: {e}", file=sys.stderr)

    except Exception as e:  # noqa: BLE001
        print(f"\n[SLOW worker outer error]: {e}", file=sys.stderr)


# ------------------------------
# Main
# ------------------------------
def main() -> None:
    global RATE, CHANNELS, CHUNK, FAST_SEGMENT_SECONDS, FAST_SEGMENT_FRAMES, SLOW_SEGMENT_SECONDS
    global audio_q, fast_q, slow_q

    parser = argparse.ArgumentParser(
        description="Dual-Whisper live transcriber (fast + refined)"
    )
    parser.add_argument(
        "--fast-segment-seconds",
        type=float,
        default=DEFAULT_FAST_SEGMENT_SECONDS,
        help="Seconds of audio per fast segment (default: %(default)s)",
    )
    parser.add_argument(
        "--slow-segment-seconds",
        type=float,
        default=DEFAULT_SLOW_SEGMENT_SECONDS,
        help="Seconds of audio per slow refinement window (default: %(default)s)",
    )
    parser.add_argument(
        "--fast-model",
        type=str,
        default=DEFAULT_FAST_MODEL,
        help="Fast (low-latency) Whisper model (default: %(default)s)",
    )
    parser.add_argument(
        "--slow-model",
        type=str,
        default=DEFAULT_SLOW_MODEL,
        help="Slow (high-accuracy) Whisper model (default: %(default)s)",
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
        "--input-device",
        type=int,
        default=None,
        help="PyAudio input device index (default: system default)",
    )
    parser.add_argument(
        "--list-devices", 
        action="store_true", 
        help="Print available PyAudio devices and exit."
    )
    parser.add_argument(
        "--device-index", 
        type=int, 
        help="PyAudio device index to capture from (use --list-devices to inspect)."
    )

    args = parser.parse_args()

    # Apply audio config
    RATE = args.rate
    CHANNELS = args.channels
    CHUNK = DEFAULT_CHUNK

    # Segment sizes
    fast_sec = max(0.2, float(args.fast_segment_seconds))
    slow_sec = max(fast_sec, float(args.slow_segment_seconds))

    FAST_SEGMENT_SECONDS = fast_sec
    FAST_SEGMENT_FRAMES = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1
    SLOW_SEGMENT_SECONDS = slow_sec

    # Queues
    audio_q = queue.Queue()
    fast_q = queue.Queue()
    slow_q = queue.Queue()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print(f"Loading FAST Whisper model '{args.fast_model}' on {device}...")
    try:
        fast_model = whisper.load_model(args.fast_model, device=device)
    except Exception as e:
        print(f"Failed to load fast Whisper model '{args.fast_model}': {e}")
        return

    print(f"Loading SLOW Whisper model '{args.slow_model}' on {device}...")
    try:
        slow_model = whisper.load_model(args.slow_model, device=device)
    except Exception as e:
        print(f"Failed to load slow Whisper model '{args.slow_model}': {e}")
        return

    # Initialize PyAudio
    try:
        pa = pyaudio.PyAudio()
        
        pyaudio_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": RATE,
            "input": True,
            "frames_per_buffer": CHUNK,
            "stream_callback": audio_callback,
        }
        if args.device_index is not None:
            pyaudio_kwargs["input_device_index"] = args.device_index
        stream_kwargs = dict(**pyaudio_kwargs)
        if args.input_device is not None:
            stream_kwargs["input_device_index"] = args.input_device
        stream = pa.open(**stream_kwargs)
    except Exception as e:
        print(f"Failed to open audio input: {e}")
        return

    stream.start_stream()

    # Threads
    t_segmenter = threading.Thread(target=segmenter_worker, daemon=True)
    t_fast = threading.Thread(target=fast_worker, args=(fast_model, device), daemon=True)
    t_slow = threading.Thread(target=slow_worker, args=(slow_model, device), daemon=True)

    t_segmenter.start()
    t_fast.start()
    t_slow.start()

    print("Listening with dual Whisper models...")
    print(
        f"- FAST model: {args.fast_model}, "
        f"segment: {FAST_SEGMENT_SECONDS:.2f}s"
    )
    print(
        f"- SLOW model: {args.slow_model}, "
        f"window: {SLOW_SEGMENT_SECONDS:.2f}s (groups multiple fast segments)"
    )
    print("Press Ctrl+C to stop.\n")

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping dual-Whisper transcription...")
    finally:
        # Stop workers
        audio_q.put(None)
        fast_q.put(None)
        slow_q.put(None)

        # Stop audio
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass

        # Join threads
        t_segmenter.join(timeout=1.0)
        t_fast.join(timeout=1.0)
        t_slow.join(timeout=1.0)


if __name__ == "__main__":
    main()