"""Records 8 s from the default mic and shows what the speech detector in live.py sees.

    python mic_check.py
"""
import os
import time
import wave

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import sounddevice as sd
import soxr

import live

SECONDS = 8
device = sd.query_devices(kind="input")
rate = int(device["default_samplerate"])
print(f"Mic: {device['name']} at {rate} Hz")
for n in (3, 2, 1):
    print(f"Recording in {n}...", flush=True)
    time.sleep(1)
print(f">>> TALK NOW for {SECONDS} seconds (a normal sentence or two) <<<", flush=True)
recording = sd.rec(SECONDS * rate, samplerate=rate, channels=1, dtype="float32")
sd.wait()
audio = soxr.resample(recording[:, 0], rate, live.SAMPLE_RATE)
print("Done.\n")

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "mic_check.wav")
with wave.open(path, "wb") as out:
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(live.SAMPLE_RATE)
    out.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())

# Same speech detector as live.py.
detector = live.SpeechDetector()
probabilities = [detector(audio[i:i + live.VAD_WINDOW]) for i in range(0, len(audio) - live.VAD_WINDOW + 1, live.VAD_WINDOW)]
speech = sum(p >= live.SPEECH_PROB for p in probabilities) * live.VAD_WINDOW / live.SAMPLE_RATE
peak = float(np.abs(audio).max())
print(f"peak level        {peak:.3f}   (speech usually 0.1-0.8; quieter still works)")
print(f"speech detected   {speech:.2f} s   (needs {live.MIN_SPEECH} s to count)")
print(f"max speech score  {max(probabilities):.2f}   (speech is >= {live.SPEECH_PROB})")
whistle = live.Whistle()
print(f"Whistle heard     {whistle.transcribe(audio)['text']!r}")
print(f"\nSaved to {path}")
