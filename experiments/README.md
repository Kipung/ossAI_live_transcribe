# Experiments

Earlier and side scripts, kept for reference. The maintained entry point is `../hybrid_transcribe_dual_whisper.py`. These need the larger `requirements.txt` in this folder (Vosk, diarization and the rest) and were not re-tested.

- `main.py`: the first version, one Whisper model transcribing fixed 3 s segments from the microphone.
- `vosk_model.py`: minimal Vosk microphone transcriber with partial and final results.
- `retranscribe.py`: Vosk for the fast preview, Whisper re-transcribing in the background.
- `hybrid_transcribe_v2.py`: Vosk live preview line plus Whisper-refined history lines, with overlapping segments.
- `live_transcribe_tk.py`: Tkinter window around the dual-Whisper pipeline.
- `live_transcribe_tk_v2.py`: a later revision of the Tkinter window.
- `transcribe_computer_audio.py`: live transcription of system audio through Windows WASAPI loopback.
