"""Sets up everything the assistant needs, then starts it. start.bat / start.sh run this for you.

    python bootstrap.py [--install-only] [arguments for assistant.py, e.g. --debug]

Needs Python 3.12+ with tkinter. Creates .venv, installs requirements.txt into it (again whenever that
file changes), fetches the Kokoro and Silero models, finds llama-server (and offers to install it), then
runs assistant.py with the venv's Python. Every step is skipped once it's done, so later starts are quick.
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
STAMP = VENV / "requirements.sha256"  # what was installed last, so a changed requirements.txt is noticed
MIN_PYTHON = (3, 12)
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
# Kokoro on the GPU. --no-deps: its full dependencies would pull in transformers and more (see requirements.txt).
MLX_AUDIO = "mlx-audio==0.5.7"
if sys.platform == "win32":
    SYSTEM_HINT = "Install Python from https://www.python.org/downloads/ (tkinter is included)."
elif sys.platform == "darwin":
    SYSTEM_HINT = "Install Python with Tk: brew install python@3.13 python-tk@3.13"
else:
    SYSTEM_HINT = "Install the system packages: sudo apt install python3-venv python3-tk libportaudio2"


def main():
    install_only = "--install-only" in sys.argv[1:]
    app_args = [arg for arg in sys.argv[1:] if arg != "--install-only"]
    check_python()
    python = ensure_venv()
    install_requirements(python)
    download_models(python)
    llama_server = find_llama_server() or install_llama_server()
    if not llama_server:
        print("\nllama.cpp (llama-server) is needed to run the language model. Install it, then run this again:")
        print("  Windows: winget install ggml.llamacpp")
        print("  macOS / Linux: brew install llama.cpp")
        print("  or download a release from https://github.com/ggml-org/llama.cpp/releases")
        return 1
    if install_only:
        print("\nAll set. Start the assistant with " + ("start.bat" if os.name == "nt" else "./start.sh") + ".")
        return 0
    env = dict(os.environ)
    env["PATH"] = str(Path(llama_server).parent) + os.pathsep + env.get("PATH", "")  # a fresh winget install
    print("\nStarting the assistant...", flush=True)
    return subprocess.call([str(python), str(ROOT / "assistant.py"), *app_args], cwd=ROOT, env=env)


def check_python():
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed, this is {platform.python_version()}. "
                 + SYSTEM_HINT)
    try:
        import tkinter  # noqa: F401  (the window toolkit; some Python builds leave it out)
    except ImportError:
        sys.exit(f"This Python ({sys.executable}) has no tkinter. " + SYSTEM_HINT)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_venv():
    """The venv's Python, creating the venv, or creating it again when it no longer runs."""
    python = venv_python()
    if python.exists() and subprocess.run([str(python), "-c", "import tkinter"], capture_output=True).returncode == 0:
        return python
    if Path(sys.prefix).resolve() == VENV.resolve():
        sys.exit("The .venv folder is broken. Delete it and run this again.")
    if VENV.exists():
        print("The .venv folder doesn't work any more (was Python moved or updated?). Creating it again...")
        shutil.rmtree(VENV)
    print(f"Creating .venv with Python {platform.python_version()}...", flush=True)
    try:
        venv.create(VENV, with_pip=True)
    except (OSError, subprocess.CalledProcessError) as error:
        shutil.rmtree(VENV, ignore_errors=True)
        sys.exit(f"Could not create .venv: {error}. " + SYSTEM_HINT)
    return python


def wanted_packages():
    """A fingerprint of what should be installed: requirements.txt, the extras below and the Python version."""
    digest = hashlib.sha256((ROOT / "requirements.txt").read_bytes())
    digest.update(f"{MLX_AUDIO if APPLE_SILICON else ''} {platform.python_version()}".encode())
    return digest.hexdigest()


def install_requirements(python):
    wanted = wanted_packages()
    if STAMP.exists() and STAMP.read_text().strip() == wanted:
        return
    print("Installing the Python packages into .venv (the first time takes a few minutes)...", flush=True)
    pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if subprocess.call([*pip, "-r", str(ROOT / "requirements.txt")]) != 0:
        sys.exit("Installing the packages failed, see the messages above. " + SYSTEM_HINT)
    if APPLE_SILICON and subprocess.call([*pip, "--no-deps", MLX_AUDIO]) != 0:
        print("mlx-audio didn't install; Kokoro will run on the CPU instead of the GPU.")
    # sounddevice needs the PortAudio library and is the usual thing missing on Linux
    check = subprocess.run([str(python), "-c", "import sounddevice, tkinter, kokoro_onnx, needle"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"The packages installed but don't load:\n{check.stderr.strip()}\n" + SYSTEM_HINT)
    STAMP.write_text(wanted)


def download_models(python):
    sys.path.insert(0, str(ROOT))
    import download_models
    if any(not (download_models.MODELS / name).exists() for name, _, _ in download_models.FILES):
        print("Downloading the voice (Kokoro) and speech detector (Silero) models, ~355 MB...", flush=True)
        if subprocess.call([str(python), str(ROOT / "download_models.py")]) != 0:
            sys.exit("Downloading the models failed, see the messages above.")


def find_llama_server():
    found = shutil.which("llama-server")
    if found:
        return found
    # Installed, but not on this terminal's PATH yet (winget and brew update it for new terminals only)
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates = [local / "Microsoft" / "WinGet" / "Links" / "llama-server.exe",
                      *sorted((local / "Microsoft" / "WinGet" / "Packages").glob("ggml.llamacpp*/llama-server.exe"))]
    else:
        candidates = [Path(p) / "llama-server" for p in ("/opt/homebrew/bin", "/usr/local/bin",
                                                          "/home/linuxbrew/.linuxbrew/bin")]
    return next((str(path) for path in candidates if path.exists()), None)


def install_llama_server():
    if os.name == "nt" and shutil.which("winget"):
        command = ["winget", "install", "--id", "ggml.llamacpp", "-e",
                   "--accept-source-agreements", "--accept-package-agreements"]
    elif shutil.which("brew"):
        command = ["brew", "install", "llama.cpp"]
    else:
        return None
    if ask(f"llama.cpp (llama-server) runs the language model and isn't installed. Install it now ({command[0]})?"):
        subprocess.call(command)
    return find_llama_server()


def ask(question):
    """Yes unless the answer is no; with no one at the keyboard, no."""
    try:
        return input(f"{question} [Y/n] ").strip().lower() not in ("n", "no")
    except EOFError:
        return False


if __name__ == "__main__":
    sys.exit(main())
