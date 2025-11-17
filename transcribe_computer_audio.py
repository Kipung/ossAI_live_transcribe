"""Live transcription of system audio using Whisper and PyAudio's WASAPI loopback."""

import argparse
import queue
import sys
import threading
import time

import numpy as np
import pyaudio
import torch
import whisper

FORMAT = pyaudio.paInt16
CHUNK = 2048
SEGMENT_SECONDS = 3  # seconds of audio per transcription chunk
TARGET_RATE = 16000  # Whisper models expect 16 kHz float32 audio


audio_q: "queue.Queue[bytes]"  # type: ignore


def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback that pushes raw audio bytes into a queue."""
    audio_q.put(in_data)
    return (None, pyaudio.paContinue)


def _format_device(pa: pyaudio.PyAudio, device: dict) -> str:
    """Nicely format a PyAudio device for logs and errors."""
    try:
        host_name = pa.get_host_api_info_by_index(device.get("hostApi", -1))["name"]
    except Exception:  # noqa: BLE001
        host_name = str(device.get("hostApi"))
    inputs = int(device.get("maxInputChannels", 0))
    outputs = int(device.get("maxOutputChannels", 0))
    sample_rate = device.get("defaultSampleRate", "?")
    return (
        f"[#{device['index']}] {device['name']} "
        f"(host={host_name}, inputs={inputs}, outputs={outputs}, default_rate={sample_rate})"
    )


def list_devices(pa: pyaudio.PyAudio) -> None:
    """Print every PyAudio device to aid manual selection."""
    print("Available PyAudio devices:")
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        print("  " + _format_device(pa, info))


def get_loopback_device(
    pa: pyaudio.PyAudio,
    preferred_index: int | None = None,
    preferred_name: str | None = None,
) -> tuple[dict, bool]:
    """Locate the best device for capturing system audio.

    Returns (device_info, use_wasapi_loopback).
    """

    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise RuntimeError(
            "PyAudio is not built with WASAPI support; loopback capture requires Windows."
        ) from exc

    wasapi_index = wasapi_info["index"]

    def resolve_by_index(idx: int) -> dict:
        if idx < 0 or idx >= pa.get_device_count():
            raise RuntimeError(f"Invalid device index {idx}.")
        return pa.get_device_info_by_index(idx)

    def resolve_by_name(name: str) -> dict | None:
        needle = name.lower()
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if needle in info["name"].lower():
                return info
        return None

    if preferred_index is not None:
        info = resolve_by_index(preferred_index)
    elif preferred_name:
        info = resolve_by_name(preferred_name)
        if info is None:
            raise RuntimeError(
                f"Could not find any device containing '{preferred_name}'. "
                "Re-run with --list-devices to inspect available options."
            )
    else:
        info = None

    def should_use_loopback(device: dict) -> bool:
        return (
            device.get("hostApi") == wasapi_index
            and int(device.get("maxInputChannels", 0)) == 0
        )

    if info is not None:
        return info, should_use_loopback(info)

    default_output = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    if default_output.get("isLoopbackDevice"):
        return default_output, False

    loopback_candidate = None
    for index in range(pa.get_device_count()):
        device = pa.get_device_info_by_index(index)
        if device.get("hostApi") != wasapi_index:
            continue
        if device.get("isLoopbackDevice"):
            if default_output["name"] in device["name"]:
                return device, False
            if loopback_candidate is None:
                loopback_candidate = device

    if loopback_candidate is not None:
        return loopback_candidate, False

    # Fall back to capturing the default output via WASAPI loopback.
    return default_output, True


def _resample_audio(audio_np: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array to dst_rate using linear interpolation."""
    if src_rate == dst_rate or audio_np.size == 0:
        return audio_np
    duration = audio_np.shape[0] / src_rate
    target_samples = max(1, int(duration * dst_rate))
    src_positions = np.linspace(0, audio_np.shape[0] - 1, num=target_samples)
    dst_audio = np.interp(src_positions, np.arange(audio_np.shape[0]), audio_np)
    return dst_audio.astype(np.float32, copy=False)


def transcribe_worker(model, segment_frames: int, channels: int, sample_rate: int) -> None:
    """Continuously pull from audio_q, chunk into segments, and transcribe with Whisper."""
    buffer: list[bytes] = []
    frames_in_buffer = 0

    print("Listening to system audio with Whisper... Press Ctrl+C to stop.\n")

    while True:
        try:
            data = audio_q.get()
            if data is None:
                break

            buffer.append(data)
            frames_in_buffer += 1

            if frames_in_buffer >= segment_frames:
                segment_bytes = b"".join(buffer)
                buffer.clear()
                frames_in_buffer = 0

                audio_np = np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32)
                if channels > 1:
                    usable = audio_np.size // channels * channels
                    audio_np = audio_np[:usable].reshape(-1, channels).mean(axis=1)
                audio_np /= 32768.0
                if sample_rate != TARGET_RATE:
                    audio_np = _resample_audio(audio_np, sample_rate, TARGET_RATE)

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
                    print(f"Recognized: {text}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[Error in transcriber]: {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-index",
        type=int,
        help="PyAudio device index to capture from (use --list-devices to inspect).",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        help="Case-insensitive substring of the device name to capture from.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=SEGMENT_SECONDS,
        help=f"Seconds of audio per transcription chunk (default: {SEGMENT_SECONDS}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Whisper model size to load (default: small).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available PyAudio devices and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pa = pyaudio.PyAudio()
    if args.list_devices:
        list_devices(pa)
        pa.terminate()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model '{args.model}' on {device} (optimized for speed)...")
    model = whisper.load_model(args.model, device=device)

    try:
        loopback_device, use_loopback = get_loopback_device(
            pa, preferred_index=args.device_index, preferred_name=args.device_name
        )
    except RuntimeError:
        pa.terminate()
        raise

    print(f"using loopback: {use_loopback}")
    channels = max(
        1,
        int(
            loopback_device.get(
                "maxOutputChannels" if use_loopback else "maxInputChannels", 2
            )
        ),
    )
    rate = int(loopback_device.get("defaultSampleRate", 44100))
    segment_frames = max(1, int(rate / CHUNK * args.segment_seconds))

    device_desc = _format_device(pa, loopback_device)
    suffix = " [WASAPI loopback]" if use_loopback else ""
    print(f"Using device {device_desc}{suffix}")

    stream_kwargs: dict = dict(
        format=FORMAT,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=loopback_device["index"],
        stream_callback=audio_callback,
    )
    if use_loopback:
        stream_kwargs["as_loopback"] = True

    stream = pa.open(**stream_kwargs)
    stream.start_stream()

    worker = threading.Thread(
        target=transcribe_worker, args=(model, segment_frames, channels, rate), daemon=True
    )
    worker.start()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping system audio transcription...")
    finally:
        audio_q.put(None)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        worker.join()


if __name__ == "__main__":
    audio_q = queue.Queue()
    main()
