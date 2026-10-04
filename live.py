"""Live transcription window for Cactus Whistle.

    python live.py

Click the record button and speak: the current phrase updates as you talk
(dimmed) and is committed when you pause. Needs: pip install "cactus-needle[mic]"
"""
import os
import queue
import re
import threading
import time
import tkinter as tk
from collections import deque

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode

import numpy as np
import sounddevice as sd
import soxr

from needle.agent.whistle import LANGUAGES, SAMPLE_RATE, Whistle

REFRESH = 0.4        # seconds between live re-transcriptions
PAUSE = 0.7          # silence that ends a phrase
MIN_PHRASE = 0.8     # shorter phrases are not committed on a pause
MAX_PHRASE = 20.0    # force a cut well under Whistle's 30 s limit
NOT_LETTERS = re.compile(r"\W+")
INTERRUPT_SPEECH = 0.6  # speech needed before checking whether someone talks over an assistant
# Whistle turns sounds that aren't speech (taps, breaths, desk rumble) into "Thank you.", so a
# speech detector decides what counts as speech: Silero VAD (MIT, github.com/snakers4/silero-vad).
# On recorded false triggers it never scored above 0.08; real speech scored 1.0.
VAD_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "silero_vad.onnx")
VAD_WINDOW = 512     # 32 ms at 16 kHz, the size Silero expects
SPEECH_PROB = 0.5    # windows at or above this are speech
MIN_SPEECH = 0.16    # seconds of speech before a phrase counts (a short "no" has ~0.25)
# What Whistle says about non-speech; a backup filter for when the detector lets one through.
FILLERS = {"thank you", "thank you very much", "thanks for watching", "thanks", "bye", "you",
           "vielen dank", "danke", "danke schön", "merci", "gracias", "grazie", "dank je wel", "dziękuję"}
FILLER_SPEECH = 0.4  # seconds of speech ...
FILLER_PEAK = 0.012  # ... and loudness a filler needs to be kept; the ones made from faint sounds peaked at 0.002-0.007


class SpeechDetector:
    """Silero VAD on onnxruntime: the probability that each 32 ms window is speech."""

    def __init__(self):
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(VAD_MODEL, options, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(64, np.float32)  # the model looks 64 samples back

    def __call__(self, window):
        x = np.concatenate([self.context, window])[None, :]
        probability, self.state = self.session.run(None, {"input": x, "state": self.state,
                                                          "sr": np.array(SAMPLE_RATE, np.int64)})
        self.context = x[0, -64:]
        return float(probability[0, 0])


class Transcriber(threading.Thread):
    """Owns the model and the microphone; everything Whistle runs on this one thread."""

    def __init__(self, events):
        super().__init__(daemon=True)
        self.events = events
        self.chunks = queue.Queue()
        self.recording = threading.Event()
        self.paused = threading.Event()    # set: the mic is ignored (e.g. while an assistant speaks)
        self.speaking = threading.Event()  # set: an assistant is talking, only listen for interruptions
        self.flush = threading.Event()     # set: drop the audio heard so far
        self.after_final = None            # "pause" or "speaking": set that event before a phrase is reported
        self.is_echo = lambda text: False  # tells the assistant's own voice apart from an interruption
        self.log_dir = None                # a folder: save every phrase's audio and transcript there
        self.pause = PAUSE
        self.language = None

    def run(self):
        self.events.put(("status", "Loading Whistle..."))
        try:
            self.whistle = Whistle()
            self.detector = SpeechDetector()
        except Exception as error:
            self.events.put(("error", f"Could not load Whistle: {error}"))
            return
        self.events.put(("ready", None))
        while True:
            self.recording.wait()
            try:
                self.session()
            except Exception as error:
                self.recording.clear()
                self.events.put(("error", str(error)))

    def transcribe(self, audio):
        if len(audio) < SAMPLE_RATE // 4:
            return ""
        result = self.whistle.transcribe(audio, language=self.language)
        self.detected = result["language"]
        return result["text"].strip()

    def session(self):
        rate = int(sd.query_devices(kind="input")["default_samplerate"])
        resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32") if rate != SAMPLE_RATE else None
        phrase = np.zeros(0, np.float32)
        pending = np.zeros(0, np.float32)  # the tail of the phrase the detector hasn't seen yet
        silent_for = 0.0
        heard_speech = False
        voiced = 0           # samples of speech in the phrase
        speech_at = 0        # sample in the phrase where speech began
        last_pass = 0.0

        step = rate // 50  # one waveform bar per 20 ms

        def callback(data, *_):
            if self.paused.is_set():
                return
            samples = data[:, 0].copy()
            self.chunks.put(samples)
            usable = len(samples) // step * step
            if usable and not self.speaking.is_set():  # drawn from here so it stays smooth; shows the assistant meanwhile
                self.events.put(("wave", np.abs(samples[:usable]).reshape(-1, step).max(axis=1).tolist()))

        with sd.InputStream(samplerate=rate, channels=1, dtype="float32", callback=callback):
            self.events.put(("status", "Listening..."))
            while self.recording.is_set():
                try:
                    chunk = self.chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if resampler is not None:
                    chunk = resampler.resample_chunk(chunk)
                if self.paused.is_set() or self.flush.is_set():
                    self.flush.clear()
                    phrase, pending = np.zeros(0, np.float32), np.zeros(0, np.float32)
                    heard_speech, silent_for, voiced = False, 0.0, 0
                    continue
                phrase = np.concatenate([phrase, chunk])
                pending = np.concatenate([pending, chunk])

                # The speech detector marks speech and the pauses between phrases.
                while len(pending) >= VAD_WINDOW:
                    window, pending = pending[:VAD_WINDOW], pending[VAD_WINDOW:]
                    if self.detector(window) >= SPEECH_PROB:
                        silent_for = 0.0
                        if voiced == 0:
                            first_speech = len(phrase) - len(pending) - VAD_WINDOW
                        voiced += VAD_WINDOW
                        if not heard_speech and voiced >= MIN_SPEECH * SAMPLE_RATE:
                            heard_speech, speech_at = True, max(0, first_speech)
                    else:
                        silent_for += VAD_WINDOW / SAMPLE_RATE

                seconds = len(phrase) / SAMPLE_RATE
                if self.speaking.is_set():
                    # Speech heard now is usually the assistant's own voice coming back through the mic.
                    # Transcribe it and treat it as an interruption only if it says something else.
                    if (heard_speech and len(phrase) - speech_at >= INTERRUPT_SPEECH * SAMPLE_RATE
                            and time.perf_counter() - last_pass >= REFRESH):
                        text = self.transcribe(phrase)
                        last_pass = time.perf_counter()
                        if text and not is_filler(text) and not self.is_echo(text):
                            self.log(phrase, voiced, "interrupt", text)
                            self.speaking.clear()
                            self.events.put(("interrupt", text))
                            continue  # keep the phrase: it carries on as the next turn
                    if silent_for >= self.pause:
                        phrase, heard_speech, silent_for, voiced = pending.copy(), False, 0.0, 0
                    elif seconds > 3.0:
                        phrase, speech_at = phrase[-SAMPLE_RATE * 3 // 2:], 0  # judge only recent audio
                    continue
                if heard_speech and silent_for >= self.pause and seconds >= MIN_PHRASE or seconds >= MAX_PHRASE:
                    self.commit(phrase, voiced)
                    phrase, heard_speech, silent_for, voiced = pending.copy(), False, 0.0, 0
                elif not heard_speech and seconds > 2.0:
                    phrase, voiced = phrase[-SAMPLE_RATE // 2:], 0  # drop leading silence and stray noises
                elif (heard_speech and len(phrase) - speech_at >= SAMPLE_RATE // 2
                      and time.perf_counter() - last_pass >= REFRESH):  # tiny slivers make up words
                    started = time.perf_counter()
                    text = self.transcribe(phrase)
                    if is_noise(text, phrase, voiced):
                        text = ""
                    self.events.put(("partial", text))
                    last_pass = time.perf_counter()
                    self.events.put(("status", f"Listening...   {(last_pass - started) * 1000:.0f} ms per pass"))

        if heard_speech:
            self.commit(phrase, voiced)
        while not self.chunks.empty():
            self.chunks.get_nowait()
        self.events.put(("status", "Stopped"))

    def commit(self, phrase, voiced):
        self.detected = None
        text = self.transcribe(phrase)
        self.log(phrase, voiced, "final", text)
        if is_noise(text, phrase, voiced):
            text = ""
        if text and self.after_final == "pause":
            self.paused.set()
        elif text and self.after_final == "speaking":
            self.speaking.set()
        self.events.put(("language", self.detected))
        self.events.put(("final", text))

    def log(self, phrase, voiced, kind, text):
        if self.log_dir is None:
            return
        import json
        import wave
        os.makedirs(self.log_dir, exist_ok=True)
        name = time.strftime("%H%M%S") + f"_{kind}"
        with wave.open(os.path.join(self.log_dir, name + ".wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes((np.clip(phrase, -1, 1) * 32767).astype("<i2").tobytes())
        entry = {"file": name + ".wav", "kind": kind, "text": text, "seconds": round(len(phrase) / SAMPLE_RATE, 2),
                 "speech_s": round(voiced / SAMPLE_RATE, 2), "peak": round(float(np.abs(phrase).max()), 4),
                 "assistant_speaking": self.speaking.is_set()}
        with open(os.path.join(self.log_dir, "phrases.jsonl"), "a", encoding="utf-8") as out:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")


def is_filler(text):
    """What Whistle tends to say about sounds that aren't speech."""
    return NOT_LETTERS.sub(" ", text.lower()).strip() in FILLERS


def is_noise(text, phrase, voiced):
    """A filler made up from a faint sound or one with barely any speech in it."""
    return is_filler(text) and (voiced < FILLER_SPEECH * SAMPLE_RATE
                                or float(np.abs(phrase).max()) < FILLER_PEAK)


BG = "#0e1014"
SURFACE = "#161a21"
BORDER = "#252b36"
TEXT = "#e6e8ec"
MUTED = "#7a8291"
ACCENT = "#ff4d5e"
ACCENT_HOVER = "#ff6b79"
ACCENT_GLOW = "#4a1d25"
IDLE_WAVE = "#2e3440"
FONT = "Segoe UI"
NAMES = {"auto": "Auto detect", "en": "English", "de": "Deutsch", "fr": "Français", "es": "Español",
         "it": "Italiano", "nl": "Nederlands", "pl": "Polski"}


def rounded(canvas, x1, y1, x2, y2, r, **options):
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **options)


def dark_title_bar(window):
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        on = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
    except (AttributeError, OSError):
        pass


class App:
    subtitle = "live transcription"
    hint = "Press record and start talking."
    fade = ("#7a2e37", "#b8404d", "#ff4d5e")  # oldest to newest wave bars while recording

    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.worker = Transcriber(self.events)
        self.state = "loading"  # loading | idle | recording
        self.language = "auto"
        self.peaks = deque([0.0] * 400, maxlen=400)
        self.palettes = deque([None] * 400, maxlen=400)  # each bar keeps the colours it was drawn in
        self.level = 0.0
        self.hovering = False
        self.empty = True

        root.title("Whistle Live")
        root.geometry("760x600")
        root.minsize(480, 420)
        root.configure(bg=BG)

        self.header = header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 12))
        tk.Label(header, text="Whistle", bg=BG, fg=TEXT, font=(FONT, 18, "bold")).pack(side="left")
        tk.Label(header, text=self.subtitle, bg=BG, fg=MUTED, font=(FONT, 11)).pack(side="left", padx=(10, 0), pady=(6, 0))
        self.picker = tk.Menubutton(header, text=f"{NAMES['auto']}  ▾", bg=SURFACE, fg=TEXT, activebackground=BORDER,
                                    activeforeground=TEXT, disabledforeground=MUTED, relief="flat", bd=0, padx=14,
                                    pady=6, font=(FONT, 10), cursor="hand2", highlightthickness=0)
        menu = tk.Menu(self.picker, tearoff=0, bg=SURFACE, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
                       bd=0, font=(FONT, 10))
        for code in ("auto",) + LANGUAGES:
            menu.add_command(label=NAMES[code], command=lambda c=code: self.pick(c))
        self.picker["menu"] = menu
        self.picker.pack(side="right")

        self.wave = tk.Canvas(root, height=110, bg=BG, highlightthickness=0)
        self.wave.pack(fill="x", padx=24)

        footer = tk.Frame(root, bg=BG)
        footer.pack(side="bottom", fill="x", padx=24, pady=(0, 18))  # packed first so it is never squeezed out
        footer.columnconfigure(0, weight=1, uniform="side")
        footer.columnconfigure(2, weight=1, uniform="side")
        self.status = tk.Label(footer, text="", bg=BG, fg=MUTED, font=(FONT, 9), anchor="w")
        self.status.grid(row=0, column=0, sticky="w")
        self.button = tk.Canvas(footer, width=84, height=84, bg=BG, highlightthickness=0, cursor="hand2")
        self.button.grid(row=0, column=1)
        self.button.bind("<Button-1>", lambda _: self.toggle())
        self.button.bind("<Enter>", lambda _: setattr(self, "hovering", True))
        self.button.bind("<Leave>", lambda _: setattr(self, "hovering", False))
        actions = tk.Frame(footer, bg=BG)
        actions.grid(row=0, column=2, sticky="e")
        for label, command in (("Copy", self.copy), ("Clear", self.clear)):
            link = tk.Label(actions, text=label, bg=BG, fg=MUTED, font=(FONT, 10), cursor="hand2", padx=8)
            link.pack(side="left")
            link.bind("<Button-1>", lambda _, c=command: c())
            link.bind("<Enter>", lambda e: e.widget.config(fg=TEXT))
            link.bind("<Leave>", lambda e: e.widget.config(fg=MUTED))

        card = tk.Frame(root, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="both", expand=True, padx=24, pady=16)
        self.text = tk.Text(card, wrap="word", font=(FONT, 14), bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                            selectbackground=BORDER, relief="flat", bd=0, padx=20, pady=16, spacing3=6,
                            highlightthickness=0)
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("partial", foreground=MUTED)
        self.text.tag_configure("hint", foreground=MUTED)
        self.clear()

        self.worker.start()
        self.dark_title_bar()
        self.tick()

    def dark_title_bar(self):
        dark_title_bar(self.root)

    def pick(self, code):
        self.language = code
        self.picker.config(text=f"{NAMES[code]}  ▾")

    def toggle(self):
        if self.state == "recording":
            self.worker.recording.clear()
            self.state = "idle"
            self.picker.config(state="normal")
        elif self.state == "idle":
            self.worker.language = None if self.language == "auto" else self.language
            self.worker.recording.set()
            self.state = "recording"
            self.picker.config(state="disabled")

    def draw_wave(self):
        canvas = self.wave
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        rounded(canvas, 1, 1, width - 1, height - 1, 18, fill=SURFACE, outline=BORDER)
        bar, gap, pad = 3, 3, 22
        count = max(1, (width - 2 * pad) // (bar + gap))
        values, palettes = list(self.peaks)[-count:], list(self.palettes)[-count:]
        middle = height / 2
        loudest = max(0.08, max(values))  # auto gain: quiet mics still fill the strip
        x = width - pad - len(values) * (bar + gap)
        for index, (value, palette) in enumerate(zip(values, palettes)):
            size = (value / loudest) ** 0.8 * (middle - 16)
            if palette and size > 2:
                color = palette[min(2, int(3 * index / len(values) + 0.6))]
            else:
                color = IDLE_WAVE
            canvas.create_line(x, middle - size - 1, x, middle + size + 1, width=bar, fill=color, capstyle="round")
            x += bar + gap

    def draw_button(self):
        canvas = self.button
        canvas.delete("all")
        c = 42
        if self.state == "recording":
            glow = 30 + self.level * 12
            canvas.create_oval(c - glow, c - glow, c + glow, c + glow, fill=ACCENT_GLOW, outline="")
            canvas.create_oval(c - 28, c - 28, c + 28, c + 28, fill=ACCENT_HOVER if self.hovering else ACCENT, outline="")
            rounded(canvas, c - 10, c - 10, c + 10, c + 10, 5, fill="white", outline="")
        else:
            ready = self.state == "idle"
            canvas.create_oval(c - 32, c - 32, c + 32, c + 32, outline=BORDER if ready else SURFACE, width=2)
            fill = (ACCENT_HOVER if self.hovering else ACCENT) if ready else BORDER
            canvas.create_oval(c - 24, c - 24, c + 24, c + 24, fill=fill, outline="")

    def show_partial(self, text):
        self.drop_hint()
        self.text.delete("live", "end-1c")
        self.text.insert("live", text, "partial")
        self.text.see("end")

    def add_final(self, text):
        self.drop_hint()
        self.text.delete("live", "end-1c")
        if text:
            self.text.insert("end-1c", text + " ")
        self.text.mark_set("live", "end-1c")
        self.text.see("end")

    def drop_hint(self):
        if self.empty:
            self.text.delete("1.0", "end")
            self.text.mark_set("live", "end-1c")
            self.text.mark_gravity("live", "left")
            self.empty = False

    def tick(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "wave":
                    self.peaks.extend(value)
                    self.palettes.extend([self.fade] * len(value))
                    self.level = max(self.level, min(1.0, max(value) * 6))
                elif kind == "partial":
                    self.show_partial(value)
                elif kind == "final":
                    self.add_final(value)
                elif kind == "status":
                    self.status.config(text=value, fg=MUTED)
                elif kind == "ready":
                    self.ready()
                elif kind == "error":
                    if self.state == "recording":
                        self.state = "idle"
                    self.picker.config(state="normal")
                    self.status.config(text=value, fg=ACCENT)
                else:
                    self.handle(kind, value)
        except queue.Empty:
            pass
        if self.state != "recording":
            self.peaks.append(0.0)  # let the wave settle to a flat line
            self.palettes.append(None)
        self.level *= 0.85
        self.draw_wave()
        self.draw_button()
        self.root.after(33, self.tick)

    def ready(self):
        self.state = "idle"
        self.status.config(text="Ready")

    def handle(self, kind, value):
        """Events this window doesn't know; subclasses add their own."""

    def clear(self):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.hint, "hint")
        self.empty = True

    def copy(self):
        if self.empty:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.text.get("1.0", "end-1c").strip())
        self.status.config(text="Copied to clipboard", fg=MUTED)


if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on scaled displays
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


