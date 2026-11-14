"""Download the voice models that are not stored in git into models/<Name>/.

Kafka ships in git. Narrator (48 kHz RVC v2, MIT, by 7TearZZZ on HuggingFace) is too big for git
(the index is 180 MB), so it is attached to this repo's GitHub release "models-v1" as .safetensors.

Usage (from the repo root, with the server's Python):  python setup/download_models.py
Standard library only; every file is checked against its SHA-256 before it is moved into place.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/sapandeep31/rvc-lan-voice/releases/download/models-v1"

MODELS = {
    "Narrator": [
        ("narrator.safetensors", "8ed139a1ef6aa742c5f6cd65b1753597db680bc876b837122dc2f9937e16e602"),
        ("narrator.index", "8d01c28caedff6879ca250647cf9bd150fc6b8ff9130e7177b9e26c44f96d9c6"),
    ],
}

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str) -> None:
    if dest.exists() and sha256(dest) == expected:
        print(f"  ok      {dest.relative_to(ROOT)} (already downloaded)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get     {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r          {done * 100 // total:3d}% of {total >> 20} MB", end="", flush=True)
        print()
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        sys.exit(f"Checksum mismatch for {dest.name}: expected {expected}, got {actual}")
    tmp.replace(dest)
    print(f"  ok      {dest.relative_to(ROOT)}")


def main() -> None:
    for name, files in MODELS.items():
        print(f"{name}:")
        folder = ROOT / "models" / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, digest in files:
            fetch(f"{RELEASE}/{filename}", folder / filename, digest)
    print("Models ready.")


if __name__ == "__main__":
    main()
