"""
Downloads the TrackNetV3 weights (142 MB, too big for git) into models/tracknetv3/ from this
repository's GitHub release, and checks them. Without them the pipeline falls back to the much
less accurate YOLO shuttle tracker.

  python download_weights.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "tracknetv3"
RELEASE_URL = "https://github.com/Agastyasingh1106/Badminton-AI/releases/download/tracknetv3-weights"
FILES = {
    "TrackNet_best.pt": "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a",
    "InpaintNet_best.pt": "5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {dest.name}: {done / total:.0%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = WEIGHTS_DIR / name
        if path.is_file() and sha256(path) == digest:
            print(f"{name}: already there")
            continue
        print(f"Downloading {name}")
        download(f"{RELEASE_URL}/{name}", path)
        if sha256(path) != digest:
            path.unlink()
            sys.exit(f"{name}: checksum mismatch, deleted. Try again.")
        print(f"{name}: ok")
    print(f"TrackNetV3 weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
