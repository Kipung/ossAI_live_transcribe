
# live_transcribe_whisper.py

import queue
import threading
import sys
import time
import argparse

import numpy as np
import pyaudio
import torch
import whisper

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # frames per buffer

# How many seconds of audio per transcription segment
SEGMENT_SECONDS = 3  # seconds of audio per transcription segment (lower = more live, higher = more accurate)
SEGMENT_FRAMES = int(RATE / CHUNK * SEGMENT_SECONDS)


audio_q: "queue.Queue[bytes]"  # type: ignore


def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback that pushes raw audio bytes into a queue."""
    audio_q.put(in_data)
    return (None, pyaudio.paContinue)


def transcribe_worker(model) -> None:
    """Continuously read from the audio queue, chunk into segments, and transcribe with Whisper."""
    buffer = []  # list[bytes]
    frames_in_buffer = 0

    print("Listening with Whisper... Press Ctrl+C to stop.\n")

    while True:
        try:
            data = audio_q.get()
            if data is None:
                break  # shutdown signal

            buffer.append(data)
            frames_in_buffer += 1

            # Once we have SEGMENT_SECONDS of audio, run a transcription
            if frames_in_buffer >= SEGMENT_FRAMES:
                segment_bytes = b"".join(buffer)
                buffer.clear()
                frames_in_buffer = 0

                # Convert raw bytes (int16) to float32 numpy array in [-1, 1]
                audio_np = (
                    np.frombuffer(segment_bytes, dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )

                # Transcribe with Whisper (tuned for speed / low latency)
                result = model.transcribe(
                    audio_np,
                    language="en",
                    task="transcribe",
                    temperature=0.0,  # greedy decoding is faster
                    condition_on_previous_text=False,  # treat each chunk independently
                    fp16=torch.cuda.is_available(),
                )
                text = result.get("text", "").strip()
                if text:
                    print(f"Recognized: {text}")
        except Exception as e:  # noqa: BLE001
            print(f"\n[Error in transcriber]: {e}", file=sys.stderr)


def main() -> None:
    #parse command-line arguments
    parser = argparse.ArgumentParser(description="Live transcription of system audio using Whisper.")
    parser.add_argument("--list-devices", action="store_true", help="Print available PyAudio devices and exit.")
    parser.add_argument("--device-index", type=int, help="PyAudio device index to capture from (use --list-devices to inspect).")
    args = parser.parse_args()

    if args.list_devices:
        p = pyaudio.PyAudio()
        print("Available PyAudio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"  Index {i}: {info['name']} (Input Channels: {info['maxInputChannels']})")
        p.terminate()
        return
    

    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model 'small' on {device} (optimized for speed)...")
    model = whisper.load_model("small", device=device)

    # Initialize PyAudio and start input stream with callback
    p = pyaudio.PyAudio()
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
    stream = p.open(**pyaudio_kwargs)
    stream.start_stream()

    # Background worker thread for transcription
    worker = threading.Thread(target=transcribe_worker, args=(model,), daemon=True)
    worker.start()

    try:
        # Keep main thread alive while audio stream is active
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        # Signal worker to exit and clean up audio
        audio_q.put(None)
        stream.stop_stream()
        stream.close()
        p.terminate()
        worker.join()


if __name__ == "__main__":
    audio_q = queue.Queue()
    main()