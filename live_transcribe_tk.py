"""
Simple Tkinter GUI live transcription with Whisper.

Features:
- Mic audio streamed via PyAudio.
- Fixed-length segments (fast) transcribed with Whisper.
- A second "refinement" pass over a larger window for better accuracy.
- Tkinter GUI with:
    - History panel (refined text)
    - Single live line (fast output)
    - Start / Stop / Quit buttons

Dependencies:
    pip install pyaudio numpy torch
    pip install openai-whisper  # NOT 'whisper'

Run:
    python live_transcribe_tk.py --model small.en --segment-seconds 1.5 --refine-window 6.0
"""

import argparse
import os
import queue
import sys
import threading
import time
from typing import List, Tuple

import numpy as np
import pyaudio
import torch
import whisper
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ------------------------------
# Defaults
# ------------------------------

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK = 1024

DEFAULT_SEGMENT_SECONDS = 1.5
DEFAULT_REFINE_WINDOW = 6.0

DEFAULT_MODEL = "small.en"
SILENCE_THRESHOLD = 1e-3

try:
    max_threads = min(4, os.cpu_count() or 4)
    torch.set_num_threads(max_threads)
except Exception:
    pass

# ------------------------------
# Audio / transcription state
# ------------------------------

FORMAT = pyaudio.paInt16
RATE = DEFAULT_RATE
CHANNELS = DEFAULT_CHANNELS
CHUNK = DEFAULT_CHUNK

SEGMENT_SECONDS = DEFAULT_SEGMENT_SECONDS
REFINE_WINDOW = DEFAULT_REFINE_WINDOW

# threading & queues
audio_q: "queue.Queue[bytes]" = queue.Queue()
segment_q: "queue.Queue[Tuple[int, bytes]]" = queue.Queue()
ui_q: "queue.Queue[str]" = queue.Queue()
stop_event = threading.Event()

segment_counter_lock = threading.Lock()
next_segment_id = 1

# transcript state
transcripts_lock = threading.Lock()
fast_segments: List[Tuple[int, str]] = []   # [(seg_id, text)]
history_lines: List[str] = []               # refined history lines

# highest segment id that has been included in a refined block
last_refined_id = 0

# PyAudio and Whisper model
pa = None
stream = None
whisper_model = None
device_type = "cpu"


# ------------------------------
# Helpers
# ------------------------------

def schedule_ui_render() -> None:
    """Ask the Tkinter main loop to redraw."""
    try:
        ui_q.put_nowait("render")
    except queue.Full:
        pass


def compute_snapshot() -> Tuple[List[str], str]:
    """
    Build a snapshot of the current transcript state for the UI.

    Returns:
    - history: list of refined lines (strings)
    - live_line: the single current fast line (string)
    """
    global last_refined_id

    with transcripts_lock:
        hist_copy = list(history_lines)

        # Only include fast segments that have not been refined yet,
        # i.e., those with seg_id strictly greater than last_refined_id.
        visible_fast = [(sid, t) for (sid, t) in fast_segments if sid > last_refined_id and t]
        # concatenate texts in seg_id order
        visible_fast.sort(key=lambda x: x[0])
        parts = [t for _, t in visible_fast]
        live_line = " ".join(parts).strip()

    return hist_copy, live_line


# ------------------------------
# Workers
# ------------------------------

def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback -> push raw audio chunks into audio_q."""
    if not stop_event.is_set():
        audio_q.put(in_data)
    return (None, pyaudio.paContinue)


def segmenter_worker() -> None:
    """
    Consume raw audio, group into SEGMENT_SECONDS chunks,
    place them on segment_q with increasing segment IDs.
    """
    global next_segment_id

    buffer: List[bytes] = []
    frames_in_buffer = 0
    frames_per_segment = int(SEGMENT_SECONDS * RATE / CHUNK) or 1

    try:
        while not stop_event.is_set():
            try:
                data = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buffer.append(data)
            frames_in_buffer += 1

            if frames_in_buffer >= frames_per_segment:
                segment_bytes = b"".join(buffer)
                buffer.clear()
                frames_in_buffer = 0

                with segment_counter_lock:
                    seg_id = next_segment_id
                    next_segment_id += 1

                segment_q.put((seg_id, segment_bytes))
    except Exception as e:
        print(f"[segmenter_worker error]: {e}", file=sys.stderr)


def transcribe_worker() -> None:
    """
    Transcribe each segment with Whisper (fast output) and
    periodically refine a larger window for history.
    """
    global fast_segments, history_lines, last_refined_id

    # amount of time covered by each segment
    per_seg_seconds = SEGMENT_SECONDS

    # track last time we saw non-empty speech on the fast path
    last_speech_time = time.monotonic()

    # for refinement window
    refine_seconds = REFINE_WINDOW
    refine_buffer: List[Tuple[int, str]] = []
    refine_accum = 0.0
    refine_audio_bytes: List[bytes] = []

    try:
        while not stop_event.is_set():
            try:
                seg_id, seg_bytes = segment_q.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_np = (
                np.frombuffer(seg_bytes, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                continue

            # ---------- FAST: low-latency per-segment transcription ----------
            try:
                result = whisper_model.transcribe(
                    audio_np,
                    language="en",
                    task="transcribe",
                    temperature=0.0,
                    condition_on_previous_text=False,
                    fp16=(device_type == "cuda"),
                )
                text = result.get("text", "").strip()
            except Exception as e:
                print(f"[transcribe_worker fast error]: {e}", file=sys.stderr)
                text = ""

            # Clean mid-sentence punctuation from fast path
            if text:
                cleaned = (
                    text.replace(".", "")
                    .replace(",", "")
                    .replace("?", "")
                    .replace("!", "")
                    .strip()
                )
            else:
                cleaned = ""

            if cleaned:
                with transcripts_lock:
                    fast_segments.append((seg_id, cleaned))
                last_speech_time = time.monotonic()
                schedule_ui_render()

            # If we've been silent for significantly longer than the refine window,
            # clear any remaining fast segments so old words don't stay "stuck".
            now = time.monotonic()
            if now - last_speech_time > (refine_seconds * 1.5):
                with transcripts_lock:
                    fast_segments = [(sid, t) for (sid, t) in fast_segments if sid > last_refined_id]
                schedule_ui_render()

            # ---------- REFINE: accumulate into larger window ----------
            refine_buffer.append((seg_id, cleaned))
            refine_audio_bytes.append(seg_bytes)
            refine_accum += per_seg_seconds

            if refine_accum < refine_seconds:
                continue

            # we reached/exceeded refine window
            combined_bytes = b"".join(refine_audio_bytes)
            refine_ids = [sid for sid, _ in refine_buffer if sid is not None]

            # reset window accumulators
            refine_buffer = []
            refine_audio_bytes = []
            refine_accum = 0.0

            # skip if empty
            if not refine_ids:
                continue

            combined_np = (
                np.frombuffer(combined_bytes, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            if np.abs(combined_np).mean() < SILENCE_THRESHOLD:
                continue

            # refine using Whisper with better decoding settings
            try:
                result_ref = whisper_model.transcribe(
                    combined_np,
                    language="en",
                    task="transcribe",
                    temperature=(0.0, 0.2, 0.4),
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    fp16=(device_type == "cuda"),
                )
                refined_text = result_ref.get("text", "").strip()
            except Exception as e:
                print(f"[transcribe_worker refine error]: {e}", file=sys.stderr)
                refined_text = ""

            if not refined_text:
                continue

            # Move refined text into history, update last_refined_id, and
            # clear corresponding fast segments (and any older ones).
            with transcripts_lock:
                if refine_ids:
                    last_in_window = max(refine_ids)
                    last_refined_id = max(last_refined_id, last_in_window)
                # Keep only fast segments that come after the refined window
                fast_segments = [(sid, t) for (sid, t) in fast_segments if sid > last_refined_id]
                history_lines.append(refined_text)

            schedule_ui_render()

    except Exception as e:
        print(f"[transcribe_worker outer error]: {e}", file=sys.stderr)


# ------------------------------
# Tkinter GUI
# ------------------------------

class TranscribeGUI:
    def __init__(self, root: tk.Tk, model_name: str):
        self.root = root
        self.root.title(f"Live Transcribe – {model_name}")

        # history
        self.history_box = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, height=15, width=80, state="disabled"
        )
        self.history_box.grid(row=0, column=0, columnspan=3, padx=8, pady=(8, 4), sticky="nsew")

        # live line
        self.live_label = tk.Label(
            root,
            text="",
            anchor="w",
            justify="left",
            font=("Menlo", 12, "italic"),
        )
        self.live_label.grid(row=1, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="w")

        # buttons
        self.btn_start = tk.Button(root, text="Start", command=self.on_start)
        self.btn_stop = tk.Button(root, text="Stop", command=self.on_stop, state="disabled")
        self.btn_quit = tk.Button(root, text="Quit", command=self.on_quit)

        self.btn_start.grid(row=2, column=0, padx=8, pady=8, sticky="w")
        self.btn_stop.grid(row=2, column=1, padx=8, pady=8, sticky="w")
        self.btn_quit.grid(row=2, column=2, padx=8, pady=8, sticky="e")

        # status
        self.status_label = tk.Label(
            root, text="Ready", anchor="w", justify="left"
        )
        self.status_label.grid(row=3, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="w")

        # grid weight
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        self.running = False
        self.segment_thread = None
        self.transcribe_thread = None

        # Poll UI queue
        self.root.after(50, self._poll_ui_queue)

    def set_status(self, msg: str) -> None:
        self.status_label.config(text=msg)

    def _poll_ui_queue(self) -> None:
        """Process UI render requests from worker threads."""
        redraw = False
        try:
            while True:
                _ = ui_q.get_nowait()
                redraw = True
        except queue.Empty:
            pass

        if redraw:
            history, live = compute_snapshot()
            # update history
            self.history_box.config(state="normal")
            self.history_box.delete("1.0", tk.END)
            if history:
                self.history_box.insert(tk.END, "\n".join(history) + "\n")
            self.history_box.config(state="disabled")
            # update live line
            self.live_label.config(text=live)

        self.root.after(50, self._poll_ui_queue)

    def on_start(self) -> None:
        if self.running:
            return
        self.running = True
        stop_event.clear()
        self.set_status("Listening...")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        # start workers
        self.segment_thread = threading.Thread(target=segmenter_worker, daemon=True)
        self.transcribe_thread = threading.Thread(target=transcribe_worker, daemon=True)
        self.segment_thread.start()
        self.transcribe_thread.start()

    def on_stop(self) -> None:
        if not self.running:
            return
        self.running = False
        stop_event.set()
        self.set_status("Stopped.")
        self.btn_start.config(state="disabled")  # no restart in this simple version
        self.btn_stop.config(state="disabled")

    def on_quit(self) -> None:
        self.on_stop()
        self.root.destroy()


# ------------------------------
# Init / main
# ------------------------------

def main() -> None:
    global pa, stream, whisper_model, device_type, RATE, CHANNELS, CHUNK
    global SEGMENT_SECONDS, REFINE_WINDOW

    parser = argparse.ArgumentParser(description="Tkinter GUI Whisper live transcriber")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Whisper model name (default: %(default)s)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Seconds per fast segment (default: %(default)s)",
    )
    parser.add_argument(
        "--refine-window",
        type=float,
        default=DEFAULT_REFINE_WINDOW,
        help="Seconds for refinement window (default: %(default)s)",
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
        "--device-index",
        type=int,
        default=None,
        help="PyAudio input device index (optional)",
    )

    args = parser.parse_args()

    RATE = args.rate
    CHANNELS = args.channels
    CHUNK = DEFAULT_CHUNK
    SEGMENT_SECONDS = max(0.2, float(args.segment_seconds))
    REFINE_WINDOW = max(SEGMENT_SECONDS, float(args.refine_window))

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Whisper model
    print(f"Loading Whisper model '{args.model}' on {device_type}...")
    try:
        whisper_model = whisper.load_model(args.model, device=device_type)
    except Exception as e:
        print(f"Failed to load Whisper model '{args.model}': {e}")
        sys.exit(1)

    # Init PyAudio
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
        if args.device_index is not None:
            stream_kwargs["input_device_index"] = args.device_index
        stream = pa.open(**stream_kwargs)
        stream.start_stream()
    except Exception as e:
        print(f"Failed to open audio input: {e}")
        sys.exit(1)

    # Start Tkinter GUI
    root = tk.Tk()
    gui = TranscribeGUI(root, args.model)
    gui.set_status("Ready. Click Start to begin.")
    root.protocol("WM_DELETE_WINDOW", gui.on_quit)

    try:
        root.mainloop()
    finally:
        # cleanup
        stop_event.set()
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()