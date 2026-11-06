

"""Simple Vosk live transcription script.

This uses the default English Vosk model (lang="en-us") and a microphone
input via PyAudio. It prints partial results inline and final results on
separate lines.

Usage:
    pip install vosk pyaudio
    python vosk_model.py

If you don't have a model yet, install one via:
    python -m vosk --download en-us
"""

import json

import pyaudio
import vosk

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048  # frames per buffer


def main() -> None:
    # Load Vosk model (English)
    print("Loading Vosk model (en-us)...")
    model = vosk.Model(lang="en-us")
    recognizer = vosk.KaldiRecognizer(model, RATE)

    # Initialize PyAudio microphone stream
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    stream.start_stream()

    print("Listening with Vosk... Press Ctrl+C to stop.\n")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)

            if len(data) == 0:
                continue

            # When Vosk has a full utterance, AcceptWaveform returns True
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"Recognized: {text}")
            else:
                # Partial result (inline so it updates live)
                partial = json.loads(recognizer.PartialResult())
                ptext = partial.get("partial", "").strip()
                if ptext:
                    print(f"\rPartial: {ptext}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping Vosk transcription...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()