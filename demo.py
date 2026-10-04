"""Cactus Whistle demo: on-device speech-to-text (en de fr es it nl pl).

    python demo.py              transcribe the sample clips (generated on first run)
    python demo.py clip.wav     transcribe your own WAV (16 kHz mono needs no extras)
    python demo.py --mic        record from the microphone (pip install "cactus-needle[mic]")
"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode

from needle.agent.whistle import Whistle, record

SAMPLES = Path(__file__).parent / "samples"

# (file, text, SAPI culture) -- spoken by Windows' built-in TTS voices
CLIPS = [
    ("english.wav", "Hello, this is a quick test of the Whistle speech recognition model running on my laptop.", "en-US"),
    ("german.wav", "Guten Morgen, wie geht es dir heute? Das Wetter ist wirklich schön.", "de-DE"),
    ("names.wav", "Please send the report to Siobhan and Krzysztof before Friday.", "en-US"),
]

TTS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq '{culture}' } | Select-Object -First 1
if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{path}', $fmt)
$s.Speak('{text}')
$s.Dispose()
if (-not $v) { Write-Output 'no-voice' }
"""


def make_samples():
    SAMPLES.mkdir(exist_ok=True)
    for name, text, culture in CLIPS:
        path = SAMPLES / name
        if path.exists():
            continue
        script = TTS.replace("{culture}", culture).replace("{path}", str(path)).replace("{text}", text.replace("'", "''"))
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, check=True)
        note = f" (no {culture} voice installed, used default voice)" if "no-voice" in out.stdout else ""
        print(f"  generated {path.name}{note}")


def show(whistle, audio, label, **options):
    started = time.perf_counter()
    result = whistle.transcribe(audio, **options)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"\n[{label}]  options={options or '{}'}")
    print(f"  text:     {result['text'] or '(no speech)'}")
    print(f"  language: {result['language']}   ttft {result['ttft_ms']:.0f} ms   "
          f"decode {result['decode_tps']:.0f} tok/s   total {elapsed:.0f} ms")
    for word in result.get("words", []):
        print(f"    {word['start']:5.2f}-{word['end']:5.2f}s  {word['word']:<14} p={word['probability']:.2f}")


def main():
    print("Loading Whistle (downloads the weights on first run)...")
    whistle = Whistle()

    if "--mic" in sys.argv:
        input("Press Enter to start recording...")
        show(whistle, record(), "microphone")
    elif len(sys.argv) > 1:
        show(whistle, sys.argv[1], sys.argv[1])
    else:
        make_samples()
        show(whistle, SAMPLES / "english.wav", "english, auto-detect language")
        show(whistle, SAMPLES / "german.wav", "german, auto-detect language")
        show(whistle, SAMPLES / "names.wav", "names, no biasing")
        show(whistle, SAMPLES / "names.wav", "names, keyword biasing", keywords=["Siobhan", "Krzysztof"])
        show(whistle, SAMPLES / "english.wav", "word timestamps", language="en", word_timestamps=True)


if __name__ == "__main__":
    main()
