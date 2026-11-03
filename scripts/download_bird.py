"""Download the BIRD dev set (1,534 questions, 11 SQLite databases, ~1.5 GB unpacked) into data/bird."""
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

from download_spider import _download

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
TARGET = Path(__file__).resolve().parents[1] / "data" / "bird"


def main() -> int:
    if list(TARGET.glob("**/dev.json")) and list(TARGET.glob("**/dev_databases/*/*.sqlite")):
        print(f"Already present: {TARGET}")
        return 0
    TARGET.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dev.zip"
        print("Downloading BIRD dev (~350 MB)…")
        _download(URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(TARGET)
    # The databases come as a zip inside the zip.
    for inner in TARGET.glob("**/dev_databases.zip"):
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
        inner.unlink()
    print(f"OK: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
