"""Download Spider 1.0 (questions + SQLite databases, ~200 MB) into data/spider_data."""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

URL = "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
DATA = Path(__file__).resolve().parents[1] / "data"


def _download(url: str, dest: Path) -> None:
    # The openai SDK ships an HTTP client with its own certificate bundle; urllib may lack one on macOS.
    try:
        import httpx2 as httpx
    except ImportError:
        try:
            import httpx
        except ImportError:
            import urllib.request

            urllib.request.urlretrieve(url, dest)
            return
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as resp, dest.open("wb") as f:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            f.write(chunk)


def main() -> int:
    target = DATA / "spider_data"
    if (target / "dev.json").exists():
        print(f"Already present: {target}")
        return 0
    DATA.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "spider_data.zip"
        print("Downloading Spider…")
        _download(URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        shutil.move(str(Path(tmp) / "spider_data"), target)
    print(f"OK: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
