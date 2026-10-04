"""Downloads the model files into models/ and checks their SHA-256.

    python download_models.py
"""
import hashlib
import urllib.request
from pathlib import Path

MODELS = Path(__file__).parent / "models"
FILES = [
    ("kokoro-v1.0.onnx", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
     "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    ("voices-v1.0.bin", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
     "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
    ("silero_vad.onnx", "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
     "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    MODELS.mkdir(exist_ok=True)
    for name, url, expected in FILES:
        path = MODELS / name
        if path.exists() and sha256(path) == expected:
            print(f"  {name} already there")
            continue
        print(f"  downloading {name}...", flush=True)
        urllib.request.urlretrieve(url, path)
        if sha256(path) != expected:
            # silero_vad.onnx comes from the repo's master branch and may change; Kokoro files are fixed releases
            print(f"  warning: {name} differs from the tested version")
    print("Done.")


if __name__ == "__main__":
    main()
