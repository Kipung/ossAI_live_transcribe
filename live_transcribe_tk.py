"""
Dual-Whisper Tkinter GUI live transcription:

- Fast Whisper model: quick, low-latency segments (FAST).
- Slow Whisper model: higher-accuracy refinement (REF) over same segments.

Architecture:
- PyAudio -> raw audio queue
- Segmenter thread:
    - collects audio into fixed-length *fast* segments (e.g., 1.5–2.0s)
    - assigns segment_id = 1, 2, 3, ...
    - pushes (segment_id, segment_bytes) into both fast and slow queues
- Fast worker:
    - transcribes each short segment with a small model (tiny.en, etc.)
    - live line in the GUI is a concatenation of all unrefined segments.
- Slow worker:
    - groups multiple fast segments until it reaches a larger window
      (e.g., 6.0s) and refines the combined audio with a larger model.
    - replaces the corresponding fast segments with one refined line
      in the history area of the GUI.

GUI:
- Tkinter window with:
    - History panel (refined text)
    - Single live line (fast text)
    - Start / Stop / Quit buttons
    - Status label

Dependencies:
    pip install pyaudio numpy torch
    pip install openai-whisper  # NOT 'whisper'

<<<<<<< HEAD
Run (example):
    python live_transcribe_tk.py \
        --fast-model tiny.en \
        --slow-model small.en \
        --fast-segment-seconds 1.5 \
        --slow-segment-seconds 6.0
=======
Run:
    python live_transcribe_tk.py --model small.en --segment-seconds 1.5 --refine-window 6.0
    python live_transcribe_tk.py --model small.en --device-index 1 --segment-seconds 1.5 --refine-window 6.0
>>>>>>> cf5539c0eb485c8d4b7ee2517ed5fea1c4d9af73
"""

import argparse
import os
import queue
import sys
import threading
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pyaudio
import torch
import whisper
import tkinter as tk
from tkinter import scrolledtext

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
MAX_HISTORY_LINES = 20

# Limit Torch threads for laptop friendliness
try:
    max_threads = min(4, os.cpu_count() or 4)
    torch.set_num_threads(max_threads)
except Exception:
    pass

# ------------------------------
# Global audio / model state
# ------------------------------

FORMAT = pyaudio.paInt16
RATE = DEFAULT_RATE
CHANNELS = DEFAULT_CHANNELS
CHUNK = DEFAULT_CHUNK

FAST_SEGMENT_SECONDS = DEFAULT_FAST_SEGMENT_SECONDS
FAST_SEGMENT_FRAMES = 0  # set in main
SLOW_SEGMENT_SECONDS = DEFAULT_SLOW_SEGMENT_SECONDS

# Queues for audio + transcription workers
audio_q: "queue.Queue[bytes]" = queue.Queue()
fast_q: "queue.Queue[Tuple[int, bytes]]" = queue.Queue()
slow_q: "queue.Queue[Tuple[int, bytes]]" = queue.Queue()

# UI queue (for render requests from worker threads)
ui_q: "queue.Queue[str]" = queue.Queue()

# Segment index & transcript bookkeeping
segment_counter_lock = threading.Lock()
next_segment_id = 1

# Transcript storage: segment_id -> {'text': str, 'refined': bool}
transcripts_lock = threading.Lock()
transcripts: Dict[int, Dict[str, Any]] = {}

# Stop event to shut down worker loops
stop_event = threading.Event()

# PyAudio and Whisper models
pa = None           # type: pyaudio.PyAudio | None
stream = None       # type: pyaudio.Stream | None
fast_model = None
slow_model = None
device_type = "cpu"

FAST_TAG = "[FAST]"
SLOW_TAG = "[REF ]"


# ------------------------------
# UI helper
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
    with transcripts_lock:
        items = sorted(transcripts.items(), key=lambda kv: kv[0])

        refined_lines: List[str] = []
        live_segments: List[str] = []

        for seg_id, entry in items:
            text = entry.get("text", "")
            if not text:
                continue
            if entry.get("refined", False):
                refined_lines.append(text)
            else:
                live_segments.append(text)

        # Limit history length
        if len(refined_lines) > MAX_HISTORY_LINES:
            refined_lines = refined_lines[-MAX_HISTORY_LINES:]

        live_line = " ".join(live_segments).strip()

    return refined_lines, live_line


# ------------------------------
# Workers (audio / segment / fast / slow)
# ------------------------------

def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback -> push raw audio chunks into audio_q."""
    if not stop_event.is_set():
        audio_q.put(in_data)
    return (None, pyaudio.paContinue)


def segmenter_worker() -> None:
    """
    Consume raw audio, group into FAST_SEGMENT_SECONDS chunks,
    place them on both fast_q and slow_q with increasing segment IDs.
    """
    global next_segment_id

    buffer: List[bytes] = []
    frames_in_buffer = 0
    frames_per_segment = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1

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

                fast_q.put((seg_id, segment_bytes))
                slow_q.put((seg_id, segment_bytes))
    except Exception as e:
        print(f"[segmenter_worker error]: {e}", file=sys.stderr)


def fast_worker() -> None:
    """
    Fast Whisper worker:

    - Uses the fast model (e.g., tiny.en).
    - Processes each FAST segment individually.
    - Updates the live line (unrefined segments) as concatenated text.
    """
    global fast_model, device_type

    try:
        while not stop_event.is_set():
            try:
                seg_id, segment_bytes = fast_q.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_np = (
                np.frombuffer(segment_bytes, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            # Skip mostly-silent segments
            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                continue

            try:
                result = fast_model.transcribe(
                    audio_np,
                    language="en",
                    task="transcribe",
                    temperature=0.0,
                    condition_on_previous_text=False,
                    fp16=(device_type == "cuda"),
                )
                text = result.get("text", "").strip()
            except Exception as e:
                print(f"[fast_worker error]: {e}", file=sys.stderr)
                text = ""

            if not text:
                continue

            # Remove mid-sentence punctuation (avoid random periods in live line)
            cleaned = (
                text.replace(".", "")
                .replace(",", "")
                .replace("?", "")
                .replace("!", "")
                .strip()
            )
            if not cleaned:
                continue

            # Only update if this segment has not (yet) been refined
            with transcripts_lock:
                entry = transcripts.get(seg_id)
                if entry is None or not entry.get("refined", False):
                    transcripts[seg_id] = {"text": cleaned, "refined": False}

            schedule_ui_render()
    except Exception as e:
        print(f"[fast_worker outer error]: {e}", file=sys.stderr)


def slow_worker() -> None:
    """
    Slow Whisper worker:

    - Uses the slower but more accurate model (e.g., small.en).
    - Groups multiple fast segments into a larger SLOW window and
      refines them as one chunk, replacing the individual fast entries.
    """
    global slow_model, device_type

    per_fast_duration = FAST_SEGMENT_SECONDS
    group_bytes: List[bytes] = []
    group_ids: List[int] = []
    group_duration = 0.0

    try:
        while not stop_event.is_set():
            try:
                seg_id, segment_bytes = slow_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Add this fast segment to the current slow-group window
            group_bytes.append(segment_bytes)
            group_ids.append(seg_id)
            group_duration += per_fast_duration

            if group_duration < SLOW_SEGMENT_SECONDS:
                continue

            # We reached or exceeded the slow window; combine and transcribe
            combined_bytes = b"".join(group_bytes)
            ids_for_group = group_ids[:]

            # Reset for the next group
            group_bytes = []
            group_ids = []
            group_duration = 0.0

            if not ids_for_group:
                continue

            audio_np = (
                np.frombuffer(combined_bytes, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            # Skip mostly-silent groups
            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                continue

            try:
                result = slow_model.transcribe(
                    audio_np,
                    language="en",
                    task="transcribe",
                    # Repetition-safe settings
                    temperature=(0.0, 0.2, 0.4),
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    fp16=(device_type == "cuda"),
                )
                text = result.get("text", "").strip()
            except Exception as e:
                print(f"[slow_worker error]: {e}", file=sys.stderr)
                text = ""

            if not text:
                continue

            final_seg_id = ids_for_group[-1]
            with transcripts_lock:
                # Drop all fast-only ids for this window
                for gid in ids_for_group:
                    if gid != final_seg_id:
                        transcripts.pop(gid, None)
                transcripts[final_seg_id] = {"text": text, "refined": True}

            schedule_ui_render()
    except Exception as e:
        print(f"[slow_worker outer error]: {e}", file=sys.stderr)



# ------------------------------
# Audio/model init helper for GUI
# ------------------------------

def init_audio_and_models(fast_name: str, slow_name: str, device_index: int | None) -> None:
    """Load Whisper models and open the PyAudio input stream if not already open."""
    global pa, stream, fast_model, slow_model, device_type, RATE, CHANNELS, CHUNK

    # Load fast model if needed or if name changed
    if fast_model is None or getattr(fast_model, "_model_name", None) != fast_name:
        print(f"Loading FAST Whisper model '{fast_name}' on {device_type}...")
        fast_model = whisper.load_model(fast_name, device=device_type)
        setattr(fast_model, "_model_name", fast_name)

    # Load slow model if needed or if name changed
    if slow_model is None or getattr(slow_model, "_model_name", None) != slow_name:
        print(f"Loading SLOW Whisper model '{slow_name}' on {device_type}...")
        slow_model = whisper.load_model(slow_name, device=device_type)
        setattr(slow_model, "_model_name", slow_name)

    # Open audio stream if not already open
    if pa is None or stream is None:
        pa = pyaudio.PyAudio()
        stream_kwargs = dict(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback,
        )
        if device_index is not None:
            stream_kwargs["input_device_index"] = device_index
        s = pa.open(**stream_kwargs)
        s.start_stream()
        stream = s

# ------------------------------
# Tkinter GUI
# ------------------------------

class TranscribeGUI:
    def __init__(self, root: tk.Tk, fast_name: str, slow_name: str, fast_sec: float, slow_sec: float, device_index: int | None):
        self.root = root
        self.fast_model_var = tk.StringVar(value=fast_name)
        self.slow_model_var = tk.StringVar(value=slow_name)
        self.fast_sec_var = tk.DoubleVar(value=fast_sec)
        self.slow_sec_var = tk.DoubleVar(value=slow_sec)
        self.device_index = device_index

        self.root.title("Dual-Whisper Live Transcribe")

        # Configuration controls
        tk.Label(root, text="Fast model:", anchor="w").grid(row=0, column=0, padx=8, pady=(8, 2), sticky="w")
        self.fast_model_entry = tk.Entry(root, textvariable=self.fast_model_var, width=20)
        self.fast_model_entry.grid(row=0, column=1, padx=4, pady=(8, 2), sticky="w")

        tk.Label(root, text="Slow model:", anchor="w").grid(row=0, column=2, padx=8, pady=(8, 2), sticky="w")
        self.slow_model_entry = tk.Entry(root, textvariable=self.slow_model_var, width=20)
        self.slow_model_entry.grid(row=0, column=3, padx=4, pady=(8, 2), sticky="w")

        tk.Label(root, text="Fast seg (s):", anchor="w").grid(row=1, column=0, padx=8, pady=(0, 4), sticky="w")
        self.fast_sec_entry = tk.Entry(root, textvariable=self.fast_sec_var, width=8)
        self.fast_sec_entry.grid(row=1, column=1, padx=4, pady=(0, 4), sticky="w")

        tk.Label(root, text="Slow window (s):", anchor="w").grid(row=1, column=2, padx=8, pady=(0, 4), sticky="w")
        self.slow_sec_entry = tk.Entry(root, textvariable=self.slow_sec_var, width=8)
        self.slow_sec_entry.grid(row=1, column=3, padx=4, pady=(0, 4), sticky="w")

        # Header showing current config (updated on start)
        header = (
            f"FAST model: {fast_name}  (segment {fast_sec:.1f}s)\n"
            f"SLOW model: {slow_name}  (window  {slow_sec:.1f}s)\n"
        )
        self.header_label = tk.Label(
            root, text=header, justify="left", anchor="w", font=("Menlo", 10, "bold")
        )
        self.header_label.grid(row=2, column=0, columnspan=4, padx=8, pady=(4, 4), sticky="w")

        # history
        self.history_box = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, height=15, width=80, state="disabled"
        )
        self.history_box.grid(row=3, column=0, columnspan=4, padx=8, pady=(0, 4), sticky="nsew")

        # live line
        self.live_label = tk.Label(
            root,
            text="",
            anchor="w",
            justify="left",
            font=("Menlo", 12, "italic"),
        )
        self.live_label.grid(row=4, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="w")

        # buttons
        self.btn_start = tk.Button(root, text="Start", command=self.on_start)
        self.btn_stop = tk.Button(root, text="Stop", command=self.on_stop, state="disabled")
        self.btn_quit = tk.Button(root, text="Quit", command=self.on_quit)

        self.btn_start.grid(row=5, column=0, padx=8, pady=8, sticky="w")
        self.btn_stop.grid(row=5, column=1, padx=8, pady=8, sticky="w")
        self.btn_quit.grid(row=5, column=3, padx=8, pady=8, sticky="e")

        # status
        self.status_label = tk.Label(
            root, text="Ready", anchor="w", justify="left"
        )
        self.status_label.grid(row=6, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="w")

        # grid weight
        root.rowconfigure(3, weight=1)
        root.columnconfigure(0, weight=1)

        self.running = False
        self.segment_thread = None
        self.fast_thread = None
        self.slow_thread = None

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
        global FAST_SEGMENT_SECONDS, FAST_SEGMENT_FRAMES, SLOW_SEGMENT_SECONDS

        if self.running:
            return

        # Read current config from widgets
        fast_name = self.fast_model_var.get().strip() or DEFAULT_FAST_MODEL
        slow_name = self.slow_model_var.get().strip() or DEFAULT_SLOW_MODEL
        try:
            fast_sec = float(self.fast_sec_var.get())
        except Exception:
            fast_sec = DEFAULT_FAST_SEGMENT_SECONDS
        try:
            slow_sec = float(self.slow_sec_var.get())
        except Exception:
            slow_sec = DEFAULT_SLOW_SEGMENT_SECONDS

        fast_sec = max(0.2, fast_sec)
        slow_sec = max(fast_sec, slow_sec)

        FAST_SEGMENT_SECONDS = fast_sec
        FAST_SEGMENT_FRAMES = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1
        SLOW_SEGMENT_SECONDS = slow_sec

        # Update header to reflect the chosen config
        header = (
            f"FAST model: {fast_name}  (segment {FAST_SEGMENT_SECONDS:.1f}s)\n"
            f"SLOW model: {slow_name}  (window  {SLOW_SEGMENT_SECONDS:.1f}s)\n"
        )
        self.header_label.config(text=header)

        # Initialize models and audio
        init_audio_and_models(fast_name, slow_name, self.device_index)

        self.running = True
        stop_event.clear()
        self.set_status("Listening...")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        # start workers
        self.segment_thread = threading.Thread(target=segmenter_worker, daemon=True)
        self.fast_thread = threading.Thread(target=fast_worker, daemon=True)
        self.slow_thread = threading.Thread(target=slow_worker, daemon=True)

        self.segment_thread.start()
        self.fast_thread.start()
        self.slow_thread.start()

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
    global pa, stream, fast_model, slow_model, device_type
    global RATE, CHANNELS, CHUNK
    global FAST_SEGMENT_SECONDS, FAST_SEGMENT_FRAMES, SLOW_SEGMENT_SECONDS

    parser = argparse.ArgumentParser(
        description="Tkinter GUI – Dual-Whisper live transcriber (fast + refined)"
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
        "--device-index",
        type=int,
        default=None,
        help="PyAudio input device index (optional)",
    )

    args = parser.parse_args()

    RATE = args.rate
    CHANNELS = args.channels
    CHUNK = DEFAULT_CHUNK

    # Segment sizes
    fast_sec = max(0.2, float(args.fast_segment_seconds))
    slow_sec = max(fast_sec, float(args.slow_segment_seconds))

    FAST_SEGMENT_SECONDS = fast_sec
    FAST_SEGMENT_FRAMES = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1
    SLOW_SEGMENT_SECONDS = slow_sec

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Start Tkinter GUI with defaults from CLI
    root = tk.Tk()
    gui = TranscribeGUI(root, args.fast_model, args.slow_model, fast_sec, slow_sec, args.device_index)
    gui.set_status("Ready. Adjust models/intervals and click Start.")
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