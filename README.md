# Voice Chat Assistant

A fully local voice assistant for Windows. Everything runs on your own PC, with no cloud APIs and no API keys. (Optional web search sends only search queries to DuckDuckGo; it's off by default.)

```
microphone ─► Silero VAD ─► Whistle + Parakeet (speech to text) ─► llama.cpp (LLM) ─► Kokoro (text to speech) ─► speakers
```

- **Whistle** (`cactus-needle`): on-device speech recognition for en, de, fr, es, it, nl and pl. Fast enough for the live preview and interruptions.
- **Parakeet v3**: writes the final text of each phrase; better with names and unclear speech. Runs on the GPU with MLX on Apple Silicon and on the CPU with onnxruntime (`onnx-asr`) elsewhere. ~2.5 GB, downloaded on first start. When it returns nothing or a non-Latin script, Whistle's text is kept.
- **Silero VAD**: tells speech apart from noise, so taps and breaths don't turn into "Thank you."
- **llama.cpp**: `llama-server` runs any GGUF model. The app starts its own server on port 8091.
- **Kokoro**: natural-sounding TTS in English, Spanish, French, Italian, Portuguese and Hindi.

Replies stream into the window and are spoken sentence by sentence. Talk over the assistant (or press Esc) to interrupt it.

## Scripts

| File | What it does |
|---|---|
| `start.bat` / `start.sh` | **Start here.** Sets everything up on the first run, then starts the assistant. Add `--debug` to save every heard phrase to `logs/phrases/`. |
| `install.bat` / `install.sh` | The same setup without starting the app (optional). |
| `assistant.py` | The voice assistant window (main app). |
| `live.py` | Live transcription window (speech to text only, no LLM). |
| `demo.py` | Transcribes sample clips, your own WAV (`python demo.py clip.wav`) or the mic (`--mic`). |
| `mic_check.py` | Records 8 s and shows what the speech detector hears. Use it to debug your mic. |
| `download_models.py` | Fetches the Kokoro and Silero model files into `models/`. |
| `bootstrap.py` | The setup the start scripts run (venv, packages, models, llama.cpp). |

## Setup

Requirements: **Windows 10/11** or an **Apple Silicon Mac** (Linux should work too, untested), a microphone and speakers (headphones work best for interrupting). For larger LLMs you need an NVIDIA GPU on Windows; the 7B default runs fine on 8 GB of VRAM.

```powershell
git clone https://github.com/EricBe15/voice-chat-assistant.git
cd voice-chat-assistant
start.bat            # Windows (or double-click it)
./start.sh           # macOS / Linux
```

The first start sets everything up, asking before it installs anything outside the project folder:

1. **Python 3.12+ with tkinter.** If there is none, it offers to install Python 3.13 (winget on Windows, Homebrew on a Mac).
2. **`.venv`** with the packages from `requirements.txt` (a few minutes). It installs them again whenever `requirements.txt` changes, and creates the venv again if it stops working (e.g. Python was updated). On Apple Silicon it also adds `mlx-audio` for Kokoro on the GPU.
3. **Kokoro and Silero models** (~355 MB, checked against SHA-256 hashes) into `models/`.
4. **llama.cpp.** If `llama-server` is missing, it offers `winget install ggml.llamacpp` or `brew install llama.cpp`.

Later starts skip all of that and open the app in a second or two. The Whistle and Parakeet weights download on the first start of the app itself; the language model asks first (see below).

On Linux, install the system packages first: `sudo apt install python3-venv python3-tk libportaudio2`, and llama.cpp from [its releases](https://github.com/ggml-org/llama.cpp/releases) or Homebrew.

To check the pieces one at a time, use the venv's Python (`.venv\Scripts\python` on Windows, `.venv/bin/python` elsewhere):

```powershell
.venv\Scripts\python demo.py        # speech to text works (downloads the Whistle weights on first run)
.venv\Scripts\python mic_check.py   # your microphone is picked up
```

### The LLM

By default the assistant uses **Qwen2.5 7B Instruct (Q4_K_M)**. If it isn't on your computer yet, the app asks whether to download it from Hugging Face (`bartowski/Qwen2.5-7B-Instruct-GGUF`, ~4.7 GB, a few minutes). Choose **Yes** and `llama-server` downloads it once; progress goes to `logs/llama-server.log`. Choose **No** and nothing is downloaded: pick another model in the settings, or restart to be asked again.

To use another model, open the settings (gear icon), choose **Model → Browse for a .gguf file…** and save. Any chat GGUF works. Models with "thinking" (Qwen3) have it switched off, and any stray `<think>` blocks are hidden.

The server is started with `-ngl 99 -c 8192` (all layers on the GPU, 8k context). Change this in `LlamaServer.start` in `assistant.py` if your GPU has less memory.

### Settings

The gear icon opens the settings: model, speech recognition (Whistle + Parakeet by default, or Whistle alone), voice (or "auto" to match the language you speak), speed, whether you can interrupt by talking, web search, and the personality prompt. They are saved to `settings.json` next to the scripts. That file isn't committed because it holds local paths. Without it, the app uses the defaults in `assistant.py`.

### Web search (optional, off by default)

With **Let it search the web** switched on, the model can call a `web_search` tool for news and other current facts. The app sends the search query to DuckDuckGo (`html.duckduckgo.com`) and gives the model the top five result titles and snippets; it doesn't open the pages. For weather it uses a separate `weather` tool instead, because search snippets rarely contain the numbers: it sends just the place name to [Open-Meteo](https://open-meteo.com) (free, no key) and gets the current weather and a 3-day forecast. Questions with words like "latest", "news", "today" or "weather" always search first, because a small model tends to answer those from its dated knowledge. Only the queries leave your PC. DuckDuckGo can rate-limit heavy use, and then the assistant says the search failed.

## Files that stay local

`.gitignore` keeps these out of the repository:

- `models/`: large binaries; the start scripts (or `download_models.py`) fetch them.
- `.venv/`: the Python packages, set up by the start scripts.
- `logs/`: llama-server output and, with `--debug`, **recordings of your voice and their transcripts**.
- `samples/`: generated or recorded test clips (`demo.py` recreates its own).
- `settings.json`: your prompt and local model paths.

## Troubleshooting

- **"Could not load Kokoro"**: run the start script again, it fetches missing models.
- **Packages broken after a Python update**: delete `.venv` and run the start script; it sets it up again.
- **"llama-server stopped"**: read `logs/llama-server.log`. Usually the model doesn't fit in VRAM; pick a smaller model or lower `-ngl`.
- **`llama-server` not found**: open a new terminal after `winget install`, or add its folder to PATH.
- **It hears nothing / transcribes "Thank you."**: run `python mic_check.py` and check the default input device in Windows sound settings.
- **It interrupts itself**: use headphones, or turn off "Let me interrupt by talking" (Esc still works).

## Notes

- `llama-server` listens only on `127.0.0.1:8091` without an API key, so it is reachable from this PC only. Don't expose that port to a network.
- Runs on Windows 10/11 and Apple Silicon Macs. Platform-specific code (stopping `llama-server`, the llama.cpp cache path, the dark title bar, GPU Kokoro, and Parakeet on MLX or onnxruntime) checks `sys.platform` or skips what isn't installed.
