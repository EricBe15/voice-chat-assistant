# Voice Chat Assistant

A fully local voice assistant for Windows. Everything runs on your own PC, with no cloud APIs and no API keys. (Optional web search sends only search queries to DuckDuckGo; it's off by default.)

```
microphone ─► Silero VAD ─► Whistle (speech to text) ─► llama.cpp (LLM) ─► Kokoro (text to speech) ─► speakers
```

- **Whistle** (`cactus-needle`): on-device speech recognition for en, de, fr, es, it, nl and pl.
- **Silero VAD**: tells speech apart from noise, so taps and breaths don't turn into "Thank you."
- **llama.cpp**: `llama-server` runs any GGUF model. The app starts its own server on port 8091.
- **Kokoro**: natural-sounding TTS in English, Spanish, French, Italian, Portuguese and Hindi.

Replies stream into the window and are spoken sentence by sentence. Talk over the assistant (or press Esc) to interrupt it.

## Scripts

| File | What it does |
|---|---|
| `assistant.py` | The voice assistant window (main app). Add `--debug` to save every heard phrase to `logs/phrases/`. |
| `live.py` | Live transcription window (speech to text only, no LLM). |
| `demo.py` | Transcribes sample clips, your own WAV (`python demo.py clip.wav`) or the mic (`--mic`). |
| `mic_check.py` | Records 8 s and shows what the speech detector hears. Use it to debug your mic. |
| `download_models.py` | Fetches the Kokoro and Silero model files into `models/`. |

## Setup

Requirements: **Windows 10/11**, Python 3.12+ (tested on 3.14), a microphone and speakers (headphones work best for interrupting). For larger LLMs you need an NVIDIA GPU; the 7B default runs fine on 8 GB of VRAM.

```powershell
git clone https://github.com/EricBe15/voice-chat-assistant.git
cd voice-chat-assistant

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Kokoro TTS + Silero VAD (~355 MB, checked against SHA-256 hashes)
python download_models.py

# llama.cpp (puts llama-server on your PATH; open a new terminal afterwards)
winget install ggml.llamacpp
```

On an Apple Silicon Mac, use `brew install llama.cpp` instead of winget and Python with Tk (e.g. Homebrew's 3.13; its 3.14 has no tkinter). Then also install Kokoro for the GPU, without its dependencies:

```bash
pip install --no-deps mlx-audio==0.5.7
```

Check the pieces one at a time:

```powershell
python demo.py        # speech to text works (downloads the Whistle weights on first run)
python mic_check.py   # your microphone is picked up
python assistant.py   # the full assistant
```

### The LLM

By default the assistant uses **Qwen2.5 7B Instruct (Q4_K_M)**. If the file is missing, `llama-server` downloads it from Hugging Face (`bartowski/Qwen2.5-7B-Instruct-GGUF`) on first start, which takes a few minutes (~4.7 GB). Progress goes to `logs/llama-server.log`.

To use another model, open the settings (gear icon), choose **Model → Browse for a .gguf file…** and save. Any chat GGUF works. Models with "thinking" (Qwen3) have it switched off, and any stray `<think>` blocks are hidden.

The server is started with `-ngl 99 -c 8192` (all layers on the GPU, 8k context). Change this in `LlamaServer.start` in `assistant.py` if your GPU has less memory.

### Settings

The gear icon opens the settings: model, speech recognition (Whistle, or Parakeet on Apple Silicon), voice (or "auto" to match the language you speak), speed, whether you can interrupt by talking, web search, and the personality prompt. They are saved to `settings.json` next to the scripts. That file isn't committed because it holds local paths. Without it, the app uses the defaults in `assistant.py`.

### Web search (optional, off by default)

With **Let it search the web** switched on, the model can call a `web_search` tool for news and other current facts. The app sends the search query to DuckDuckGo (`html.duckduckgo.com`) and gives the model the top five result titles and snippets; it doesn't open the pages. For weather it uses a separate `weather` tool instead, because search snippets rarely contain the numbers: it sends just the place name to [Open-Meteo](https://open-meteo.com) (free, no key) and gets the current weather and a 3-day forecast. Questions with words like "latest", "news", "today" or "weather" always search first, because a small model tends to answer those from its dated knowledge. Only the queries leave your PC. DuckDuckGo can rate-limit heavy use, and then the assistant says the search failed.

## Files that stay local

`.gitignore` keeps these out of the repository:

- `models/`: large binaries; run `download_models.py` to get them.
- `logs/`: llama-server output and, with `--debug`, **recordings of your voice and their transcripts**.
- `samples/`: generated or recorded test clips (`demo.py` recreates its own).
- `settings.json`: your prompt and local model paths.

## Troubleshooting

- **"Could not load Kokoro"**: run `python download_models.py`.
- **"llama-server stopped"**: read `logs/llama-server.log`. Usually the model doesn't fit in VRAM; pick a smaller model or lower `-ngl`.
- **`llama-server` not found**: open a new terminal after `winget install`, or add its folder to PATH.
- **It hears nothing / transcribes "Thank you."**: run `python mic_check.py` and check the default input device in Windows sound settings.
- **It interrupts itself**: use headphones, or turn off "Let me interrupt by talking" (Esc still works).

## Notes

- `llama-server` listens only on `127.0.0.1:8091` without an API key, so it is reachable from this PC only. Don't expose that port to a network.
- Windows only for now: the app uses `netstat`/`taskkill`, `CREATE_NO_WINDOW` and Windows title-bar APIs.
