"""Dual-Whisper Tkinter GUI live transcription.

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

Run (example):
    python live_transcribe_tk.py \
        --fast-model tiny.en \
        --slow-model small.en \
        --fast-segment-seconds 1.5 \
        --slow-segment-seconds 6.0
"""

import argparse
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Tuple

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

# List of available Whisper model names for dropdowns
WHISPER_MODEL_CHOICES = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v2",
    "large-v3",
]

SILENCE_THRESHOLD = 5e-3  # skip near-silent segments
MAX_HISTORY_LINES = 20
LIVE_MAX_AGE_SECONDS = 5.0  # drop fast-only segments from live line after this many seconds

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

# UI queue (for render + status requests from worker threads)
ui_q: "queue.Queue[Any]" = queue.Queue()

# Segment index & transcript bookkeeping
segment_counter_lock = threading.Lock()
next_segment_id = 1

# Transcript storage: segment_id -> {'text': str, 'refined': bool, 'ts': float}
transcripts_lock = threading.Lock()
transcripts: Dict[int, Dict[str, Any]] = {}

# Stop event to shut down worker loops
stop_event = threading.Event()

# PyAudio and Whisper models
pa: pyaudio.PyAudio | None = None
stream: pyaudio.Stream | None = None
fast_model = None
slow_model = None
device_type = "cpu"
# Track the currently open input device index for the PyAudio stream
current_device_index: int | None = None

# ------------------------------
# Speaker diarization / "who spoke" tracking (slow/refined only)
# ------------------------------
ENABLE_DIARIZATION = True  # set False to disable speaker labeling

try:
    from resemblyzer import VoiceEncoder  # pip install resemblyzer
    HAVE_RESEMBLYZER = True
except ImportError:
    VoiceEncoder = None  # type: ignore
    HAVE_RESEMBLYZER = False

# Simple incremental clustering over speaker embeddings
# Higher threshold => harder to merge speakers, more likely to create S2, S3, ...
# You can tune this between ~0.75 and 0.95 depending on your environment.
SPEAKER_SIM_THRESHOLD = 0.85
MAX_SPEAKERS = 8

speaker_encoder: "VoiceEncoder | None" = None
speaker_centroids: List[np.ndarray] = []
speaker_counts: List[int] = []

# Track most recent speaker id for the live (FAST) line
last_speaker_id: int | None = None
last_speaker_id_lock = threading.Lock()


# ------------------------------
# UI helper
# ------------------------------

def schedule_ui_render() -> None:
    """Ask the Tkinter main loop to redraw."""
    try:
        ui_q.put_nowait("render")
    except queue.Full:
        pass


def post_status(msg: str) -> None:
    """Request a status label update from worker threads."""
    try:
        ui_q.put_nowait(("status", msg))
    except queue.Full:
        pass


# ------------------------------
# Diarization helpers
# ------------------------------

def init_speaker_encoder() -> None:
    """Lazily initialize the speaker embedding encoder."""
    global speaker_encoder
    if not ENABLE_DIARIZATION or not HAVE_RESEMBLYZER:
        return
    if speaker_encoder is None:
        speaker_encoder = VoiceEncoder()


def assign_speaker_id(audio_np: np.ndarray) -> int | None:
    """Assign a speaker id for the given audio chunk using simple centroid clustering.

    Returns an integer speaker id (0, 1, 2, ...) or None if diarization is disabled.
    """
    global speaker_centroids, speaker_counts

    if not ENABLE_DIARIZATION or not HAVE_RESEMBLYZER:
        return None

    init_speaker_encoder()
    if speaker_encoder is None:
        return None

    # Ensure 1-D float32 array
    wav = audio_np.astype(np.float32).flatten()
    if wav.size == 0:
        return None

    emb = speaker_encoder.embed_utterance(wav)

    # First speaker
    if not speaker_centroids:
        speaker_centroids.append(emb)
        speaker_counts.append(1)
        return 0

    # Compute cosine similarity to existing centroids
    sims = []
    for c in speaker_centroids:
        denom = (np.linalg.norm(c) * np.linalg.norm(emb)) + 1e-8
        sims.append(float(np.dot(c, emb) / denom))
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    if best_sim >= SPEAKER_SIM_THRESHOLD:
        # Update that centroid
        count = speaker_counts[best_idx]
        new_centroid = (speaker_centroids[best_idx] * count + emb) / (count + 1)
        speaker_centroids[best_idx] = new_centroid
        speaker_counts[best_idx] = count + 1
        return best_idx

    # Otherwise, create a new speaker if we have capacity
    if len(speaker_centroids) < MAX_SPEAKERS:
        speaker_centroids.append(emb)
        speaker_counts.append(1)
        return len(speaker_centroids) - 1

    # If at capacity, just assign to the most similar existing speaker
    count = speaker_counts[best_idx]
    new_centroid = (speaker_centroids[best_idx] * count + emb) / (count + 1)
    speaker_centroids[best_idx] = new_centroid
    speaker_counts[best_idx] = count + 1
    return best_idx


def compute_snapshot() -> Tuple[List[str], str]:
    """Build a snapshot of the current transcript state for the UI.

    Returns:
    - history: list of refined lines (strings)
    - live_line: the single current fast line (string)
    """
    with transcripts_lock:
        items = sorted(transcripts.items(), key=lambda kv: kv[0])

        refined_lines: List[str] = []
        live_segments: List[str] = []

        now = time.time()

        for seg_id, entry in items:
            text = entry.get("text", "")
            if not text:
                continue
            if entry.get("refined", False):
                speaker_id = entry.get("speaker")
                if isinstance(speaker_id, int):
                    label = f"S{speaker_id + 1}: "
                else:
                    label = ""
                refined_lines.append(label + text)
            else:
                ts = entry.get("ts")
                if ts is not None:
                    age = now - ts
                    if age > LIVE_MAX_AGE_SECONDS:
                        continue
                live_segments.append(text)

        if len(refined_lines) > MAX_HISTORY_LINES:
            refined_lines = refined_lines[-MAX_HISTORY_LINES:]

        live_line = " ".join(live_segments).strip()

        # Optionally prefix live line with most recent speaker label
        if live_line:
            with last_speaker_id_lock:
                sid = last_speaker_id
            if isinstance(sid, int):
                live_line = f"S{sid + 1} • {live_line}"

    return refined_lines, live_line


# ------------------------------
# Workers (audio / segment / fast / slow)
# ------------------------------

def is_explosive(text: str, segment_seconds: float, max_wps: float = 4.0) -> bool:
    """Return True if text is suspiciously long for the given audio duration."""
    words = text.strip().split()
    if not words:
        return False
    max_words = int(segment_seconds * max_wps)
    return len(words) > max_words

def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback -> push raw audio chunks into audio_q."""
    if not stop_event.is_set():
        audio_q.put(in_data)
    return (None, pyaudio.paContinue)


def segmenter_worker() -> None:
    """Consume raw audio, group into FAST_SEGMENT_SECONDS chunks.

    For each chunk, assign a segment_id and place the bytes on fast_q and slow_q.
    """
    global next_segment_id

    buffer: List[bytes] = []
    frames_in_buffer = 0

    while not stop_event.is_set():
        try:
            data = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        buffer.append(data)
        frames_in_buffer += 1

        frames_per_segment = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1
        if frames_in_buffer < frames_per_segment:
            continue

        segment_bytes = b"".join(buffer)
        buffer.clear()
        frames_in_buffer = 0

        with segment_counter_lock:
            seg_id = next_segment_id
            next_segment_id += 1

        fast_q.put((seg_id, segment_bytes))
        slow_q.put((seg_id, segment_bytes))


def fast_worker() -> None:
    """Fast Whisper worker.

    - Uses the fast model (e.g., tiny.en).
    - Processes each FAST segment individually.
    - Updates the live line (unrefined segments) as concatenated text.
    """
    global fast_model, device_type

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

        if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
            continue

        try:
            result = fast_model.transcribe(
                audio_np,
                language="en",
                task="transcribe",
                temperature=0.0,                    # greedy decoding (no sampling)
                compression_ratio_threshold=2.0,    # lower => more strict
                logprob_threshold=-0.5,             # discard very low-confidence outputs
                no_speech_threshold=0.6,            # treat low-energy as no speech
                condition_on_previous_text=False,
                fp16=(device_type == "cuda"),
            )
            text = result.get("text", "").strip()
            if not text:
                continue

            # Explosion guard for fast segments
            if is_explosive(text, FAST_SEGMENT_SECONDS):
                # Optionally print for debugging:
                # print(f"[fast_worker] dropped explosive segment: {len(text.split())} words", file=sys.stderr)
                continue
        except Exception as e:
            print(f"[fast_worker error]: {e}", file=sys.stderr)
            text = ""

        if not text:
            continue

        cleaned = (
            text.replace(".", "")
            .replace(",", "")
            .replace("?", "")
            .replace("!", "")
            .strip()
        )
        if not cleaned:
            continue

        now_ts = time.time()
        with transcripts_lock:
            entry = transcripts.get(seg_id)
            if entry is None or not entry.get("refined", False):
                transcripts[seg_id] = {
                    "text": cleaned,
                    "refined": False,
                    "ts": now_ts,
                }

        schedule_ui_render()


def slow_worker() -> None:
    """Slow Whisper worker.

    - Uses the slower but more accurate model (e.g., small.en).
    - Groups multiple fast segments into a larger SLOW window and
      refines them as one chunk, replacing the individual fast entries.
    """
    global slow_model, device_type

    per_fast_duration = FAST_SEGMENT_SECONDS
    group_bytes: List[bytes] = []
    group_ids: List[int] = []
    group_duration = 0.0

    while not stop_event.is_set():
        try:
            seg_id, segment_bytes = slow_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # Add this fast segment to the current slow-group window
        group_bytes.append(segment_bytes)
        group_ids.append(seg_id)
        group_duration += per_fast_duration

        # Inform the UI how close we are to a slow refinement window
        remaining = max(0.0, SLOW_SEGMENT_SECONDS - group_duration)
        post_status(f"Listening... refining in ~{remaining:.1f}s")

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

        # Assign a speaker id for this combined slow window (if diarization enabled)
        speaker_id = assign_speaker_id(audio_np)

        if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
            continue

        try:
            result = slow_model.transcribe(
                audio_np,
                language="en",
                task="transcribe",
                # still allow some temperature sweep, but with stricter filters
                temperature=(0.0, 0.2),
                compression_ratio_threshold=2.0,
                logprob_threshold=-0.5,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                fp16=(device_type == "cuda"),
            )
            text = result.get("text", "").strip()

            if not text:
                continue
            #explosion guard for slow window
            if is_explosive(text, SLOW_SEGMENT_SECONDS):
                continue

        except Exception as e:
            print(f"[slow_worker error]: {e}", file=sys.stderr)
            text = ""

        if not text:
            continue

        final_seg_id = ids_for_group[-1]
        now_ts = time.time()
        with transcripts_lock:
            for gid in ids_for_group:
                if gid != final_seg_id:
                    transcripts.pop(gid, None)
            prev = transcripts.get(final_seg_id, {})
            ts_val = prev.get("ts", now_ts)
            transcripts[final_seg_id] = {
                "text": text,
                "refined": True,
                "ts": ts_val,
                "speaker": speaker_id,
            }

        # Update global "most recent speaker" for live-line labeling
        if speaker_id is not None:
            with last_speaker_id_lock:
                last_speaker_id = speaker_id

        schedule_ui_render()
        post_status("Listening... slow refinement updated.")


# ------------------------------
# Audio/model init helper for GUI
# ------------------------------

def init_audio_and_models(fast_name: str, slow_name: str, device_index: int | None) -> None:
    """Load Whisper models and open the PyAudio input stream if not already open, or if device changed."""
    global pa, stream, fast_model, slow_model, device_type, RATE, CHANNELS, CHUNK, current_device_index

    # (Re)load models if needed
    if fast_model is None or getattr(fast_model, "_model_name", None) != fast_name:
        print(f"Loading FAST Whisper model '{fast_name}' on {device_type}...")
        fast_model_local = whisper.load_model(fast_name, device=device_type)
        setattr(fast_model_local, "_model_name", fast_name)
        fast_model = fast_model_local

    if slow_model is None or getattr(slow_model, "_model_name", None) != slow_name:
        print(f"Loading SLOW Whisper model '{slow_name}' on {device_type}...")
        slow_model_local = whisper.load_model(slow_name, device=device_type)
        setattr(slow_model_local, "_model_name", slow_name)
        slow_model = slow_model_local

    # Ensure we have a PyAudio instance
    if pa is None:
        pa = pyaudio.PyAudio()

    # If the stream is missing or the device index has changed, (re)open it
    if stream is None or current_device_index != device_index:
        # Close any existing stream
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            stream = None

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
        current_device_index = device_index


# ------------------------------
# Tkinter GUI
# ------------------------------

class TranscribeGUI:
    def _on_device_change(self, *args) -> None:
        """Handle changes to the selected input device.

        If transcription is currently running, switch the PyAudio input
        stream to the newly selected device without requiring a restart.
        """
        # Update self.device_index from the current StringVar selection
        self._update_device_index_from_selection()

        # Only reinitialize audio if we are already running
        if self.running:
            fast_name = self.fast_model_var.get().strip() or DEFAULT_FAST_MODEL
            slow_name = self.slow_model_var.get().strip() or DEFAULT_SLOW_MODEL
            init_audio_and_models(fast_name, slow_name, self.device_index)
            self.set_status("Input device switched.")
    def _build_device_list(self) -> List[str]:
        """Return a list of input-capable PyAudio devices as 'index: name' labels."""
        try:
            p = pyaudio.PyAudio()
        except Exception:
            return ["Default input"]

        labels: List[str] = []
        try:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    dev_name = info.get("name", f"Device {i}")
                    labels.append(f"{i}: {dev_name}")
        except Exception:
            pass
        finally:
            try:
                p.terminate()
            except Exception:
                pass

        if not labels:
            labels = ["Default input"]
        return labels

    def _update_device_index_from_selection(self) -> None:
        """Update self.device_index based on the current selection in the device menu."""
        try:
            selected = self.device_var.get()
        except Exception:
            self.device_index = None
            return

        if selected.startswith("Default"):
            self.device_index = None
            return

        try:
            idx_str = selected.split(":", 1)[0]
            self.device_index = int(idx_str)
        except Exception:
            self.device_index = None

    def __init__(
        self,
        root: tk.Tk,
        fast_name: str,
        slow_name: str,
        fast_sec: float,
        slow_sec: float,
        device_index: int | None,
    ):
        self.root = root

        if fast_name not in WHISPER_MODEL_CHOICES:
            fast_name = DEFAULT_FAST_MODEL
        if slow_name not in WHISPER_MODEL_CHOICES:
            slow_name = DEFAULT_SLOW_MODEL

        self.fast_model_var = tk.StringVar(value=fast_name)
        self.slow_model_var = tk.StringVar(value=slow_name)
        self.fast_sec_var = tk.DoubleVar(value=fast_sec)
        self.slow_sec_var = tk.DoubleVar(value=slow_sec)
        self.device_index = device_index

        self.root.title("Dual-Whisper Live Transcribe")

        # Configuration controls
        tk.Label(root, text="Fast model:", anchor="w").grid(
            row=0, column=0, padx=8, pady=(8, 2), sticky="w"
        )
        self.fast_model_menu = tk.OptionMenu(root, self.fast_model_var, *WHISPER_MODEL_CHOICES)
        self.fast_model_menu.config(width=18)
        self.fast_model_menu.grid(row=0, column=1, padx=4, pady=(8, 2), sticky="w")

        tk.Label(root, text="Slow model:", anchor="w").grid(
            row=0, column=2, padx=8, pady=(8, 2), sticky="w"
        )
        self.slow_model_menu = tk.OptionMenu(root, self.slow_model_var, *WHISPER_MODEL_CHOICES)
        self.slow_model_menu.config(width=18)
        self.slow_model_menu.grid(row=0, column=3, padx=4, pady=(8, 2), sticky="w")

        tk.Label(root, text="Fast seg (s):", anchor="w").grid(
            row=1, column=0, padx=8, pady=(0, 4), sticky="w"
        )
        self.fast_sec_entry = tk.Entry(root, textvariable=self.fast_sec_var, width=8)
        self.fast_sec_entry.grid(row=1, column=1, padx=4, pady=(0, 4), sticky="w")

        tk.Label(root, text="Slow window (s):", anchor="w").grid(
            row=1, column=2, padx=8, pady=(0, 4), sticky="w"
        )
        self.slow_sec_entry = tk.Entry(root, textvariable=self.slow_sec_var, width=8)
        self.slow_sec_entry.grid(row=1, column=3, padx=4, pady=(0, 4), sticky="w")

        # Input device selection
        tk.Label(root, text="Input device:", anchor="w").grid(
            row=2, column=0, padx=8, pady=(0, 4), sticky="w"
        )
        self.device_choices = self._build_device_list()
        default_label = "Default input"
        if self.device_index is not None:
            for label in self.device_choices:
                if label.startswith(f"{self.device_index}:"):
                    default_label = label
                    break
        self.device_var = tk.StringVar(value=default_label)
        # When the user changes the device selection, update the active input stream.
        self.device_var.trace_add("write", self._on_device_change)
        self.device_menu = tk.OptionMenu(root, self.device_var, *self.device_choices)
        self.device_menu.config(width=30)
        self.device_menu.grid(row=2, column=1, columnspan=3, padx=4, pady=(0, 4), sticky="w")

        # Header showing current config (updated on start / reload)
        header = (
            f"FAST model: {fast_name}  (segment {fast_sec:.1f}s)\n"
            f"SLOW model: {slow_name}  (window  {slow_sec:.1f}s)\n"
        )
        self.header_label = tk.Label(
            root, text=header, justify="left", anchor="w", font=("Menlo", 10, "bold")
        )
        self.header_label.grid(row=3, column=0, columnspan=4, padx=8, pady=(4, 4), sticky="w")

        # History
        self.history_box = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, height=15, width=80, state="disabled"
        )
        self.history_box.grid(row=4, column=0, columnspan=4, padx=8, pady=(0, 4), sticky="nsew")

        # Live line
        self.live_label = tk.Label(
            root,
            text="",
            anchor="w",
            justify="left",
            font=("Menlo", 12, "italic"),
        )
        self.live_label.grid(row=5, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="w")

        # Buttons
        self.btn_start = tk.Button(root, text="Start", command=self.on_start)
        self.btn_stop = tk.Button(root, text="Stop", command=self.on_stop, state="disabled")
        self.btn_reload = tk.Button(root, text="Reload models", command=self.on_reload_models)
        self.btn_quit = tk.Button(root, text="Quit", command=self.on_quit)

        self.btn_start.grid(row=6, column=0, padx=8, pady=8, sticky="w")
        self.btn_stop.grid(row=6, column=1, padx=8, pady=8, sticky="w")
        self.btn_reload.grid(row=6, column=2, padx=8, pady=8, sticky="w")
        self.btn_quit.grid(row=6, column=3, padx=8, pady=8, sticky="e")

        # Status
        self.status_label = tk.Label(root, text="Ready", anchor="w", justify="left")
        self.status_label.grid(row=7, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="w")

        # Grid weight
        root.rowconfigure(4, weight=1)
        root.columnconfigure(0, weight=1)

        self.running = False
        self.segment_thread: threading.Thread | None = None
        self.fast_thread: threading.Thread | None = None
        self.slow_thread: threading.Thread | None = None

        # Poll UI queue
        self.root.after(50, self._poll_ui_queue)

    def set_status(self, msg: str) -> None:
        self.status_label.config(text=msg)

    def _poll_ui_queue(self) -> None:
        """Process UI render and status update requests from worker threads."""
        redraw = False
        status_msg = None
        try:
            while True:
                item = ui_q.get_nowait()
                if item == "render":
                    redraw = True
                elif isinstance(item, tuple) and len(item) == 2 and item[0] == "status":
                    status_msg = item[1]
        except queue.Empty:
            pass

        if redraw:
            history, live = compute_snapshot()
            self.history_box.config(state="normal")
            self.history_box.delete("1.0", tk.END)
            if history:
                self.history_box.insert(tk.END, "\n".join(history) + "\n")
                # Auto-scroll to the end of the history whenever new text is rendered
                self.history_box.see(tk.END)
            self.history_box.config(state="disabled")
            self.live_label.config(text=live)

        if status_msg is not None:
            self.set_status(status_msg)

        self.root.after(50, self._poll_ui_queue)

    def on_start(self) -> None:
        global FAST_SEGMENT_SECONDS, FAST_SEGMENT_FRAMES, SLOW_SEGMENT_SECONDS

        if self.running:
            return

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

        header = (
            f"FAST model: {fast_name}  (segment {FAST_SEGMENT_SECONDS:.1f}s)\n"
            f"SLOW model: {slow_name}  (window  {SLOW_SEGMENT_SECONDS:.1f}s)\n"
        )
        self.header_label.config(text=header)

        self._update_device_index_from_selection()
        init_audio_and_models(fast_name, slow_name, self.device_index)

        self.running = True
        stop_event.clear()
        self.set_status("Listening...")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        self.segment_thread = threading.Thread(target=segmenter_worker, daemon=True)
        self.fast_thread = threading.Thread(target=fast_worker, daemon=True)
        self.slow_thread = threading.Thread(target=slow_worker, daemon=True)

        self.segment_thread.start()
        self.fast_thread.start()
        self.slow_thread.start()

    def on_reload_models(self) -> None:
        global FAST_SEGMENT_SECONDS, FAST_SEGMENT_FRAMES, SLOW_SEGMENT_SECONDS

        fast_name = self.fast_model_var.get().strip() or DEFAULT_FAST_MODEL
        slow_name = self.slow_model_var.get().strip() or DEFAULT_SLOW_MODEL

        try:
            fast_sec = float(self.fast_sec_var.get())
        except Exception:
            fast_sec = FAST_SEGMENT_SECONDS or DEFAULT_FAST_SEGMENT_SECONDS
        try:
            slow_sec = float(self.slow_sec_var.get())
        except Exception:
            slow_sec = SLOW_SEGMENT_SECONDS or DEFAULT_SLOW_SEGMENT_SECONDS

        fast_sec = max(0.2, fast_sec)
        slow_sec = max(fast_sec, slow_sec)

        FAST_SEGMENT_SECONDS = fast_sec
        FAST_SEGMENT_FRAMES = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1
        SLOW_SEGMENT_SECONDS = slow_sec

        header = (
            f"FAST model: {fast_name}  (segment {FAST_SEGMENT_SECONDS:.1f}s)\n"
            f"SLOW model: {slow_name}  (window  {SLOW_SEGMENT_SECONDS:.1f}s)\n"
        )
        self.header_label.config(text=header)

        self._update_device_index_from_selection()
        init_audio_and_models(fast_name, slow_name, self.device_index)
        self.set_status("Models reloaded.")

    def on_stop(self) -> None:
        global next_segment_id
        if not self.running:
            return

        self.running = False
        stop_event.set()
        self.set_status("Stopped.")
        self.btn_stop.config(state="disabled")

        for thr in (self.segment_thread, self.fast_thread, self.slow_thread):
            if thr is not None and thr.is_alive():
                try:
                    thr.join(timeout=1.0)
                except Exception:
                    pass

        while not audio_q.empty():
            audio_q.get_nowait()
        while not fast_q.empty():
            fast_q.get_nowait()
        while not slow_q.empty():
            slow_q.get_nowait()

        with transcripts_lock:
            transcripts.clear()

        with segment_counter_lock:
            next_segment_id = 1

        # Reset speaker clustering state for a fresh session
        global speaker_centroids, speaker_counts, last_speaker_id
        speaker_centroids = []
        speaker_counts = []
        with last_speaker_id_lock:
            last_speaker_id = None

        self.btn_start.config(state="normal")

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

    fast_sec = max(0.2, float(args.fast_segment_seconds))
    slow_sec = max(fast_sec, float(args.slow_segment_seconds))

    FAST_SEGMENT_SECONDS = fast_sec
    FAST_SEGMENT_FRAMES = int(FAST_SEGMENT_SECONDS * RATE / CHUNK) or 1
    SLOW_SEGMENT_SECONDS = slow_sec

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    root = tk.Tk()
    gui = TranscribeGUI(root, args.fast_model, args.slow_model, fast_sec, slow_sec, args.device_index)
    gui.set_status("Ready. Adjust models/intervals and click Start.")
    root.protocol("WM_DELETE_WINDOW", gui.on_quit)

    try:
        root.mainloop()
    finally:
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