"""Local voice assistant: Whistle (speech to text) -> llama.cpp -> Kokoro (text to speech).

    python assistant.py            (add --debug to save every heard phrase to logs/phrases)

Press record, say something and pause: the reply streams into the window and is
spoken sentence by sentence. Talk over it (or press Esc) to interrupt. The gear
opens the settings: voice, speed, interruptions and the assistant's prompt.
Runs its own llama-server on port 8091; pick the model in the settings.
"""
import difflib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from kokoro_onnx import Kokoro

from live import ACCENT, BG, BORDER, FONT, MUTED, SURFACE, TEXT, App, dark_title_bar

ROOT = Path(__file__).parent
SETTINGS_FILE = ROOT / "settings.json"
LLAMA_PORT = 8091   # its own port, so other llama-servers (8080) are left alone
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "llama.cpp"
# (label, GGUF file, Hugging Face download used when the file is missing). For split
# models the first part is enough; llama.cpp finds the rest.
MODELS = [
    ("Qwen2.5 7B · fast", LLAMA_CACHE / "bartowski_Qwen2.5-7B-Instruct-GGUF_Qwen2.5-7B-Instruct-Q4_K_M.gguf",
     "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"),
    ("Qwen3.8 27B Uncensored", ROOT.parent / "qwen-uncensored" / "Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf", None),
    ("Qwen2.5 32B", LLAMA_CACHE / "Qwen_Qwen2.5-32B-Instruct-GGUF_qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf", None),
]
KOKORO_MODEL = ROOT / "models" / "kokoro-v1.0.onnx"
KOKORO_VOICES = ROOT / "models" / "voices-v1.0.bin"
TURN_PAUSE = 1.0     # silence that ends your turn
HISTORY = 20         # messages of conversation sent to the model

DEFAULT_PROMPT = ("You are a friendly voice assistant running locally on the user's PC. Your replies are spoken "
                  "aloud, so answer in one to three short sentences of plain conversational text: no markdown, "
                  "lists, code or emojis.")
# Always added after the prompt from the settings.
PROMPT_SUFFIX = (" The user's words come from speech recognition and may contain small mistakes. "
                 "Always reply in {language}, and always say at least one short sentence, even when the user "
                 "only says something like hmm or okay.")
DEFAULTS = {"voice": "auto", "speed": 1.0, "interrupt": True, "prompt": DEFAULT_PROMPT,
            "model": str(MODELS[0][1]), "custom_models": []}

# Kokoro voice prefix -> (espeak language, reply language, menu label). Japanese and Chinese
# voices need a phonemizer kokoro-onnx doesn't ship, so they are left out.
VOICE_LANGUAGES = {
    "a": ("en-us", "English", "English (US)"),
    "b": ("en-gb", "English", "English (UK)"),
    "e": ("es", "Spanish", "Spanish"),
    "f": ("fr-fr", "French", "French"),
    "i": ("it", "Italian", "Italian"),
    "p": ("pt-br", "Portuguese", "Portuguese (BR)"),
    "h": ("hi", "Hindi", "Hindi"),
}
# With the voice on "auto": the voice for each Whistle language Kokoro can speak.
# German, Dutch and Polish are understood but answered in English.
AUTO_VOICES = {"en": "af_heart", "es": "ef_dora", "fr": "ff_siwis", "it": "if_sara"}
PREVIEW = {"English": "Hi! This is how I sound.", "Spanish": "¡Hola! Así es como sueno.",
           "French": "Bonjour ! Voici ma voix.", "Italian": "Ciao! Questa è la mia voce.",
           "Portuguese": "Olá! Esta é a minha voz.", "Hindi": "नमस्ते! मेरी आवाज़ ऐसी है।"}
SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+")
NOT_LETTERS = re.compile(r"\W+")
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)
ECHO_MATCH = 0.65    # how closely heard text must match the reply to count as the assistant's own voice
SPEEDS = (0.8, 0.9, 1.0, 1.1, 1.25)
VOICE_COLOR = "#7c8cff"
VOICE_FADE = ("#3a4277", "#5964b8", VOICE_COLOR)


def voice_language(voice):
    """(espeak language, reply language) of a Kokoro voice such as "bf_emma"."""
    espeak, language, _ = VOICE_LANGUAGES[voice[0]]
    return espeak, language


def voice_label(voice):
    return f"{voice[3:].capitalize()} ({'female' if voice[1] == 'f' else 'male'})"


def load_settings():
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{key: value for key, value in saved.items() if key in DEFAULTS}}


class LlamaServer:
    """The llama-server this window runs on its own port; switching models restarts it."""

    def __init__(self):
        self.process = None
        self.model = None

    def ready(self):
        try:
            return requests.get(f"{LLAMA_URL}/health", timeout=1).status_code == 200
        except requests.RequestException:
            return False

    def loaded(self):
        """Path of the model a server on our port is serving, if any."""
        try:
            return requests.get(f"{LLAMA_URL}/v1/models", timeout=2).json()["data"][0]["id"]
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None

    def start(self, model):
        if self.process is None and self.ready() and self.loaded() == model:
            self.model = model  # left running by an earlier session: adopt it
            return
        self.stop()
        (ROOT / "logs").mkdir(exist_ok=True)
        log = open(ROOT / "logs" / "llama-server.log", "w")
        self.process = subprocess.Popen(["llama-server", *model_args(model), "-ngl", "99", "-c", "8192",
                                         "--port", str(LLAMA_PORT)], stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        while not self.ready():
            if self.process.poll() is not None:
                self.process = None
                raise RuntimeError("llama-server stopped, see logs/llama-server.log")
            time.sleep(0.5)
        self.model = model

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(15)
            self.process = None
        elif self.ready():  # a leftover from a session that didn't close cleanly
            free_port()
        self.model = None


def free_port():
    """Ends whatever listens on our port. Only called once it answered as a llama-server."""
    netstat = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    for line in netstat.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[1].endswith(f":{LLAMA_PORT}") and parts[2].endswith(":0"):  # listening socket
            subprocess.run(["taskkill", "/PID", parts[4], "/F"], capture_output=True)
    for _ in range(20):
        try:
            requests.get(f"{LLAMA_URL}/health", timeout=0.5)
        except requests.RequestException:
            return
        time.sleep(0.25)


def without_reasoning(text):
    """The reply with any <think>…</think> block removed, for text streamed so far.

    Thinking is switched off, but some models (the uncensored Qwen3.8) open a reasoning block anyway,
    and llama.cpp then passes it through as reply text. An unfinished block, or the start of a tag
    still arriving, is held back until it is complete.
    """
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text:  # the opening tag was in the prompt
        text = text.rsplit("</think>", 1)[1].lstrip()
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    for tag in ("</think>", "<think>"):
        for size in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:size]):
                return text[:-size]
    return text


def model_args(model):
    if Path(model).exists():
        return ["-m", model]
    for _, path, download in MODELS:
        if str(path) == model and download:
            return ["-hf", download]
    raise RuntimeError(f"model file not found: {model}")


def model_label(model):
    for label, path, _ in MODELS:
        if str(path) == model:
            return label
    return Path(model).stem


def available_models(settings):
    """(label, model) for the built-in models present on this PC (or downloadable) plus the ones you added."""
    found = [(label, str(path)) for label, path, download in MODELS if path.exists() or download]
    known = {model for _, model in found}
    return found + [(Path(m).stem, m) for m in settings["custom_models"] if m not in known and Path(m).exists()]


class Voice(threading.Thread):
    """Speaks sentences in order. Synthesis and playback overlap, so speech starts after the first sentence.

    Every item carries the turn it belongs to; next_turn() drops whatever is still queued or playing.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.events = None
        self.sentences = queue.Queue()  # (turn, text, voice, lang, speed), or (turn, None) after the last one
        self.audio = queue.Queue()
        self.turn = 0
        self.spoken = []  # sentences of the current turn that have started playing

    def next_turn(self):
        self.turn += 1
        self.spoken = []
        return self.turn

    def run(self):
        try:
            self.kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
            self.kokoro.create("Ready.", voice="af_heart")  # the first call is slow
        except Exception as error:
            self.events.put(("error", f"Could not load Kokoro: {error}"))
            return
        threading.Thread(target=self.play, daemon=True).start()
        self.events.put(("part_ready", "voice"))
        while True:
            turn, *item = self.sentences.get()
            if turn != self.turn:
                continue
            if item[0] is None:
                self.audio.put((turn, None, None))
                continue
            text, voice, lang, speed = item
            try:
                samples, _ = self.kokoro.create(text, voice=voice, lang=lang, speed=speed)
            except Exception as error:  # an odd sentence shouldn't end the conversation
                self.events.put(("status", f"Could not speak a sentence: {error}"))
                continue
            self.audio.put((turn, samples, text))

    def play(self):
        block = 960  # 40 ms at Kokoro's 24 kHz
        with sd.OutputStream(samplerate=24000, channels=1, dtype="float32") as stream:
            while True:
                turn, samples, text = self.audio.get()
                if turn != self.turn:
                    continue
                if samples is None:
                    self.events.put(("spoken", turn))
                    continue
                self.spoken.append(text)
                for start in range(0, len(samples), block):
                    if turn != self.turn:
                        break
                    chunk = samples[start:start + block]
                    stream.write(chunk.reshape(-1, 1))
                    self.events.put(("wave", np.abs(chunk).reshape(-1, 480).max(axis=1).tolist()
                                     if len(chunk) == block else [float(np.abs(chunk).max())]))


class Settings(tk.Toplevel):
    """Model, voice, speed, interruptions and prompt. Saved to settings.json; used from the next reply on."""

    def __init__(self, app):
        super().__init__(app.root, bg=BG)
        self.app = app
        self.title("Settings")
        self.geometry("560x740")
        self.minsize(460, 640)
        self.transient(app.root)
        dark_title_bar(self)
        settings = app.settings
        self.voice = settings["voice"]
        self.model = settings["model"]
        self.custom_models = list(settings["custom_models"])

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=20)
        label = dict(bg=BG, fg=TEXT, font=(FONT, 10, "bold"), anchor="w")
        note = dict(bg=BG, fg=MUTED, font=(FONT, 9), anchor="w", justify="left")

        tk.Label(body, text="Model", **label).pack(fill="x")
        self.model_picker = tk.Menubutton(body, bg=SURFACE, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
                                          relief="flat", bd=0, padx=14, pady=6, font=(FONT, 10), cursor="hand2",
                                          anchor="w", highlightthickness=0)
        self.model_picker.pack(fill="x", pady=(6, 0))
        self.build_model_menu()
        tk.Label(body, text="A different model is loaded onto the GPU when you save: a few seconds, about 10 s "
                            "for the 27B.", wraplength=500, **note).pack(fill="x", pady=(6, 0))

        tk.Label(body, text="Voice", **label).pack(fill="x", pady=(18, 0))
        row = tk.Frame(body, bg=BG)
        row.pack(fill="x", pady=(6, 0))
        self.picker = tk.Menubutton(row, text=self.voice_text(), bg=SURFACE, fg=TEXT, activebackground=BORDER,
                                    activeforeground=TEXT, relief="flat", bd=0, padx=14, pady=6, font=(FONT, 10),
                                    cursor="hand2", anchor="w", highlightthickness=0)
        menu = tk.Menu(self.picker, tearoff=0, bg=SURFACE, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
                       bd=0, font=(FONT, 10))
        menu.add_command(label="Auto: match the language I speak", command=lambda: self.pick("auto"))
        menu.add_separator()
        voices = sorted(np.load(KOKORO_VOICES).files)
        for prefix, (_, _, title) in VOICE_LANGUAGES.items():
            group = tk.Menu(menu, tearoff=0, bg=SURFACE, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
                            bd=0, font=(FONT, 10))
            for voice in (v for v in voices if v[0] == prefix):
                group.add_command(label=voice_label(voice), command=lambda v=voice: self.pick(v))
            menu.add_cascade(label=title, menu=group)
        self.picker["menu"] = menu
        self.picker.pack(side="left", fill="x", expand=True)
        self.link(row, "▶  Preview", self.preview).pack(side="left", padx=(12, 0))
        tk.Label(body, text="A specific voice also sets the reply language. Auto picks an English, Spanish, "
                            "French or Italian voice from what you say.", wraplength=500, **note).pack(fill="x", pady=(6, 0))

        tk.Label(body, text="Speed", **label).pack(fill="x", pady=(18, 0))
        pills = tk.Frame(body, bg=BG)
        pills.pack(fill="x", pady=(6, 0))
        self.speed = min(SPEEDS, key=lambda speed: abs(speed - settings["speed"]))
        self.pills = {}
        for speed in SPEEDS:
            pill = tk.Label(pills, text=f"{speed:g}×", bg=SURFACE, fg=TEXT, font=(FONT, 10), padx=14, pady=5,
                            cursor="hand2")
            pill.pack(side="left", padx=(0, 6))
            pill.bind("<Button-1>", lambda _, s=speed: self.pick_speed(s))
            self.pills[speed] = pill
        self.pick_speed(self.speed)

        self.interrupt = tk.BooleanVar(value=settings["interrupt"])
        tk.Checkbutton(body, text="Let me interrupt by talking", variable=self.interrupt, bg=BG, fg=TEXT,
                       selectcolor=SURFACE, activebackground=BG, activeforeground=TEXT, font=(FONT, 10, "bold"),
                       anchor="w", bd=0, highlightthickness=0).pack(fill="x", pady=(14, 0))
        tk.Label(body, text="Its own voice is filtered out by comparing what the mic hears with what it is "
                            "saying. Works best with headphones. Esc always interrupts.", wraplength=500,
                 **note).pack(fill="x", pady=(2, 0))

        top = tk.Frame(body, bg=BG)
        top.pack(fill="x", pady=(18, 0))
        tk.Label(top, text="Personality prompt", **label).pack(side="left")
        self.link(top, "Reset", lambda: self.set_prompt(DEFAULT_PROMPT)).pack(side="right")
        frame = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        frame.pack(fill="both", expand=True, pady=(6, 0))
        self.prompt = tk.Text(frame, wrap="word", height=7, font=(FONT, 10), bg=SURFACE, fg=TEXT,
                              insertbackground=TEXT, selectbackground=BORDER, relief="flat", bd=0, padx=12, pady=10,
                              highlightthickness=0, undo=True)
        self.prompt.pack(fill="both", expand=True)
        self.set_prompt(settings["prompt"])
        tk.Label(body, text="Added automatically: that your words come from speech recognition, and which "
                            "language to reply in.", wraplength=500, **note).pack(fill="x", pady=(6, 0))

        buttons = tk.Frame(self, bg=BG)
        buttons.pack(fill="x", padx=24, pady=(0, 20))
        tk.Button(buttons, text="Save", command=self.save, bg=ACCENT, fg="white", activebackground="#ff6b79",
                  activeforeground="white", relief="flat", bd=0, padx=22, pady=6, font=(FONT, 10, "bold"),
                  cursor="hand2").pack(side="right")
        self.link(buttons, "Cancel", self.destroy).pack(side="right", padx=12)
        self.bind("<Escape>", lambda _: self.destroy())

    def link(self, parent, text, command):
        link = tk.Label(parent, text=text, bg=BG, fg=MUTED, font=(FONT, 10), cursor="hand2")
        link.bind("<Button-1>", lambda _: command())
        link.bind("<Enter>", lambda _: link.config(fg=TEXT))
        link.bind("<Leave>", lambda _: link.config(fg=MUTED))
        return link

    def build_model_menu(self):
        menu = tk.Menu(self.model_picker, tearoff=0, bg=SURFACE, fg=TEXT, activebackground=BORDER,
                       activeforeground=TEXT, bd=0, font=(FONT, 10))
        for label, model in available_models({"custom_models": self.custom_models}):
            menu.add_command(label=label, command=lambda m=model: self.pick_model(m))
        menu.add_separator()
        menu.add_command(label="Browse for a .gguf file…", command=self.browse_model)
        self.model_picker["menu"] = menu
        self.model_picker.config(text=f"{model_label(self.model)}  ▾")

    def pick_model(self, model):
        self.model = model
        self.model_picker.config(text=f"{model_label(model)}  ▾")

    def browse_model(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self, title="Choose a GGUF model", initialdir=str(ROOT.parent),
                                          filetypes=[("GGUF models", "*.gguf")])
        if not path:
            return
        path = str(Path(path))
        if path not in self.custom_models and all(str(p) != path for _, p, _ in MODELS):
            self.custom_models.append(path)
        self.model = path
        self.build_model_menu()

    def voice_text(self):
        if self.voice == "auto":
            return "Auto: match the language I speak  ▾"
        return f"{voice_label(self.voice)} · {VOICE_LANGUAGES[self.voice[0]][2]}  ▾"

    def pick(self, voice):
        self.voice = voice
        self.picker.config(text=self.voice_text())

    def pick_speed(self, speed):
        self.speed = speed
        for value, pill in self.pills.items():
            pill.config(bg=ACCENT if value == speed else SURFACE, fg="white" if value == speed else TEXT)

    def set_prompt(self, text):
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", text)

    def preview(self):
        voice = AUTO_VOICES["en"] if self.voice == "auto" else self.voice
        self.app.say(PREVIEW[voice_language(voice)[1]], voice, self.speed)

    def save(self):
        settings = {"voice": self.voice, "speed": self.speed, "interrupt": self.interrupt.get(),
                    "prompt": self.prompt.get("1.0", "end-1c").strip() or DEFAULT_PROMPT,
                    "model": self.model, "custom_models": self.custom_models}
        switch = settings["model"] != self.app.settings["model"]
        self.app.settings = settings
        try:
            SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
            self.app.status.config(text="Settings saved", fg=MUTED)
        except OSError as error:
            self.app.status.config(text=f"Could not save settings: {error}", fg=ACCENT)
        self.destroy()
        if switch:
            self.app.switch_model()


class Assistant(App):
    subtitle = "voice assistant"
    hint = "Press record, say something, then pause. The assistant answers out loud; talk over it or press Esc to interrupt."

    def __init__(self, root):
        self.voice = Voice()
        self.server = LlamaServer()
        self.server_lock = threading.Lock()
        self.settings = load_settings()
        self.history = []
        self.answer = None  # history entry of the reply in progress
        self.replying = False
        self.waiting = {"whistle", "llama", "voice"}
        self.user_open = False
        self.detected = None
        self.settings_window = None
        super().__init__(root)
        self.voice.events = self.events
        self.worker.pause = TURN_PAUSE
        self.worker.is_echo = self.is_echo
        if "--debug" in sys.argv:
            self.worker.log_dir = str(ROOT / "logs" / "phrases")
        root.title("Whistle Assistant")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.interrupt())
        gear = tk.Label(self.header, text="⚙", bg=BG, fg=MUTED, font=(FONT, 15), cursor="hand2", padx=10)
        gear.pack(side="right")
        gear.bind("<Button-1>", lambda _: self.open_settings())
        gear.bind("<Enter>", lambda _: gear.config(fg=TEXT))
        gear.bind("<Leave>", lambda _: gear.config(fg=MUTED))
        self.text.tag_configure("you_label", foreground=MUTED, font=(FONT, 10, "bold"), spacing1=10)
        self.text.tag_configure("bot_label", foreground=VOICE_COLOR, font=(FONT, 10, "bold"), spacing1=10)
        self.text.tag_configure("bot", foreground=TEXT)
        self.text.tag_configure("cut", foreground=MUTED, font=(FONT, 11, "italic"))
        self.text.tag_configure("unsaid", foreground="#4b5262", overstrike=True)
        self.text.tag_raise("unsaid")
        self.voice.start()
        threading.Thread(target=self.boot_llama, daemon=True).start()

    def boot_llama(self):
        model = self.settings["model"]
        with self.server_lock:
            if model != self.settings["model"]:
                return  # switched again while waiting; the newer thread loads that one
            downloading = not Path(model).exists()
            self.events.put(("status", f"Loading {model_label(model)}" + (" (downloading first)..." if downloading else "...")))
            try:
                self.server.start(model)
                # The first request is seconds slower (GPU warm-up), so spend it while loading.
                requests.post(f"{LLAMA_URL}/v1/chat/completions", timeout=300, json={
                    "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}, "messages": [
                        {"role": "system", "content": self.system_prompt("English")}, {"role": "user", "content": "Hi"}]})
            except (OSError, RuntimeError, requests.RequestException, subprocess.TimeoutExpired) as error:
                self.events.put(("error", f"llama.cpp: {error}"))
                return
        self.events.put(("part_ready", "llama"))

    def switch_model(self):
        if self.state == "recording":
            self.toggle()
        self.stop_reply()
        self.state = "loading"
        self.waiting.add("llama")
        threading.Thread(target=self.boot_llama, daemon=True).start()

    def ready(self):
        self.part_ready("whistle")

    def part_ready(self, part):
        self.waiting.discard(part)
        if self.waiting:
            self.status.config(text="Loading " + ", ".join(sorted(self.waiting)) + "...")
        else:
            self.state = "idle"
            self.status.config(text=f"Ready · {model_label(self.server.model or self.settings['model'])}")

    def open_settings(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
        else:
            self.settings_window = Settings(self)

    def system_prompt(self, language):
        return self.settings["prompt"] + PROMPT_SUFFIX.format(language=language)

    def toggle(self):
        self.stop_reply()
        self.worker.paused.clear()
        self.worker.speaking.clear()
        self.worker.flush.set()
        super().toggle()

    # Interrupting -----------------------------------------------------------

    def is_echo(self, text):
        """Whether text heard while the assistant talks is just its own voice (runs on the Whistle thread).

        Compares letters rather than words: the mic often mishears its voice ("there lived" as "they're
        lifted"), but the letters still line up with some stretch of the reply. Echo scores 0.7 and up,
        someone talking over it 0.55 and below.
        """
        heard = NOT_LETTERS.sub("", text.lower())
        said = NOT_LETTERS.sub("", ((self.answer or {}).get("content") or "").lower())
        if len(heard) < 8:  # too short to compare fuzzily, e.g. "Stop"
            return heard in said
        matcher = difflib.SequenceMatcher(autojunk=False)
        matcher.set_seq2(heard)
        best = 0.0
        for start in range(max(1, len(said) - len(heard) + 1)):
            matcher.set_seq1(said[start:start + len(heard) + 4])
            if matcher.real_quick_ratio() > best and matcher.quick_ratio() > best:
                best = max(best, matcher.ratio())
        return best >= ECHO_MATCH

    def interrupt(self):
        """Esc: stop talking and listen."""
        if self.replying:
            self.stop_reply()
            self.worker.speaking.clear()
            self.worker.paused.clear()
            self.worker.flush.set()
            self.status.config(text="Interrupted. Listening...", fg=MUTED)

    def stop_reply(self):
        if not self.replying:
            return
        spoken = " ".join(self.voice.spoken)
        last = self.voice.spoken[-1] if self.voice.spoken else ""
        self.voice.next_turn()
        self.replying = False
        self.fade = App.fade
        # Dim what was written but never said: everything after the last sentence that played.
        written = self.text.get("reply", "end-1c")
        found = written.find(last) if last else -1
        said = found + len(last) if found >= 0 else (len(written) if last else 0)
        if said < len(written):
            self.text.tag_add("unsaid", f"reply + {said} chars", "end-1c")
        if self.answer is not None:  # remember only what was actually said
            self.answer["content"] = (spoken + " …" if spoken else "…") + " [interrupted by the user]"
        self.text.insert("end-1c", " (interrupted)", "cut")
        self.text.see("end")

    def say(self, text, voice, speed):
        """Speak a line outside the conversation (the settings preview)."""
        self.stop_reply()
        turn = self.voice.next_turn()
        self.voice.sentences.put((turn, text, voice, voice_language(voice)[0], speed))

    # Conversation -----------------------------------------------------------

    def open_user_block(self):
        self.drop_hint()
        self.text.mark_set("block", "end-1c")
        self.text.mark_gravity("block", "left")
        self.text.insert("end-1c", ("\n" if self.text.index("end-1c") != "1.0" else "") + "You\n", "you_label")
        self.text.mark_set("live", "end-1c")
        self.user_open = True

    def show_partial(self, text):
        if not text and not self.user_open:
            return
        if not self.user_open:
            self.open_user_block()
        super().show_partial(text)

    def add_final(self, text):
        if not text:
            if self.user_open:
                self.text.delete("block", "end-1c")
                self.user_open = False
            return
        if not self.user_open:
            self.open_user_block()
        super().add_final(text)
        self.user_open = False
        self.text.insert("end-1c", "\nAssistant\n", "bot_label")
        self.text.mark_set("reply", "end-1c")
        self.text.mark_gravity("reply", "left")
        self.text.see("end")
        self.start_reply(text)

    def start_reply(self, text):
        settings = self.settings
        if settings["voice"] == "auto":
            voice = AUTO_VOICES.get(self.worker.language or self.detected, AUTO_VOICES["en"])
        else:
            voice = settings["voice"]
        lang, language = voice_language(voice)
        self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": self.system_prompt(language)}] + self.history[-HISTORY:]
        self.answer = {"role": "assistant", "content": ""}
        self.history.append(self.answer)
        turn = self.voice.next_turn()
        self.replying = True
        self.fade = VOICE_FADE
        self.status.config(text="Thinking...", fg=MUTED)
        threading.Thread(target=self.reply, args=(turn, messages, self.answer, voice, lang, settings["speed"]),
                         daemon=True).start()

    def handle(self, kind, value):
        if kind == "part_ready":
            self.part_ready(value)
        elif kind == "language":
            self.detected = value
        elif kind == "token":
            turn, token = value
            if turn == self.voice.turn and self.replying:
                self.text.insert("end-1c", token, "bot")
                self.text.see("end")
        elif kind == "no_answer":
            if value == self.voice.turn and self.replying:
                self.text.insert("end-1c", "(no answer)", "cut")
                self.text.see("end")
                if self.answer in self.history:  # an empty reply in the history invites more of them
                    self.history.remove(self.answer)
        elif kind == "speaking":
            if value == self.voice.turn and self.replying:
                tip = "talk or press Esc to interrupt" if self.settings["interrupt"] else "Esc to interrupt"
                self.status.config(text=f"Speaking...   {tip}", fg=MUTED)
        elif kind == "interrupt":
            if self.replying:
                self.stop_reply()
                self.status.config(text="Interrupted. Listening...", fg=MUTED)
            self.show_partial(value)
        elif kind == "spoken":
            if value == self.voice.turn and self.replying:
                self.root.after(300, lambda: self.listen_again(value))  # let the room go quiet first

    def listen_again(self, turn):
        if turn != self.voice.turn or not self.replying:
            return
        self.replying = False
        self.fade = App.fade
        self.worker.flush.set()  # drop the tail of its own voice
        self.worker.speaking.clear()
        self.worker.paused.clear()
        if self.state == "recording":
            self.status.config(text="Listening...", fg=MUTED)

    def tick(self):
        # Before each phrase is reported, the Whistle thread either mutes the mic or listens for interruptions.
        self.worker.after_final = "speaking" if self.settings["interrupt"] else "pause"
        super().tick()

    # Talking to llama.cpp ---------------------------------------------------

    def reply(self, turn, messages, answer, voice, lang, speed):
        raw, shown, pending, spoke = "", "", "", False

        def speak(sentence):
            nonlocal spoke
            sentence = re.sub(r"[*#_`~]", "", sentence).strip()
            if sentence:
                self.voice.sentences.put((turn, sentence, voice, lang, speed))
                if not spoke:
                    self.events.put(("speaking", turn))
                    spoke = True

        # Models sometimes end their turn at once, e.g. after a bare "Hmm.", so an empty reply is asked
        # for once more, a little more randomly.
        for temperature in (0.7, 1.0):
            raw = ""
            try:
                with requests.post(f"{LLAMA_URL}/v1/chat/completions", stream=True, timeout=120,
                                   json={"messages": messages, "stream": True, "temperature": temperature,
                                         "max_tokens": 1024,
                                         # thinking models (Qwen3) would otherwise reason silently before every reply
                                         "chat_template_kwargs": {"enable_thinking": False}}) as response:
                    response.raise_for_status()
                    response.encoding = "utf-8"
                    for line in response.iter_lines(decode_unicode=True):
                        if turn != self.voice.turn:
                            return
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        raw += json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
                        visible = without_reasoning(raw)
                        if not visible.startswith(shown):
                            # A late </think>: what came before was reasoning. It can't be unsaid, but the
                            # answer from here on is shown and spoken, and only it is remembered.
                            shown, pending, answer["content"] = "", "", ""
                        token, shown = visible[len(shown):], visible
                        if not token:
                            continue
                        answer["content"] += token
                        pending += token
                        self.events.put(("token", (turn, token)))
                        *sentences, pending = SENTENCE_END.split(pending)
                        for sentence in sentences:
                            speak(sentence)
            except requests.RequestException as error:
                self.events.put(("error", f"llama.cpp: {error}"))
                self.events.put(("spoken", turn))
                return
            if shown.strip():
                break
        if turn == self.voice.turn:
            if not shown.strip():
                self.events.put(("no_answer", turn))
            speak(pending)
            self.voice.sentences.put((turn, None))

    def close(self):
        self.voice.next_turn()
        self.server.stop()
        self.root.destroy()


if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    Assistant(root)
    root.mainloop()
