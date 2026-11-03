"""Benchmark loading (Spider, BIRD): questions, gold query, database path, difficulty and schema prompt."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .hardness import hardness
from .sandbox import execute

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "spider_data"
BIRD_ROOT = REPO_ROOT / "data" / "bird"
DATASETS = ["spider", "bird"]

# Question files and database folder for each split.
SPLITS = {
    "train": (["train_spider.json", "train_others.json"], "database"),
    "dev": (["dev.json"], "database"),
    "test": (["test.json"], "test_database"),
}


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    gold_sql: str
    db_id: str
    db_path: Path
    hardness: str | None = None
    evidence: str | None = None  # BIRD's external-knowledge hint, given to the model with the question
    metric: str = "spider"  # comparison rule used by the reward (see reward.py)


def load_dataset(dataset: str, split: str, root: str | Path | None = None) -> list[Example]:
    if dataset == "bird":
        if split != "dev":
            raise ValueError("Only BIRD's dev split is supported (its test set is hidden).")
        return load_bird_dev(root or BIRD_ROOT)
    return load_split(split, root or DEFAULT_ROOT)


def load_bird_dev(root: str | Path = BIRD_ROOT) -> list[Example]:
    """BIRD dev (1,534 questions, 11 databases). Run `python scripts/download_bird.py` first."""
    found = sorted(Path(root).glob("**/dev.json"))
    if not found:
        raise FileNotFoundError(f"BIRD dev.json not found under {root}: run `python scripts/download_bird.py` first.")
    base = found[0].parent
    return [Example(
        id=f"bird-dev-{i}",
        question=ex["question"].strip(),
        gold_sql=ex["SQL"].strip(),
        db_id=ex["db_id"],
        db_path=base / "dev_databases" / ex["db_id"] / f"{ex['db_id']}.sqlite",
        hardness=ex.get("difficulty"),
        evidence=(ex.get("evidence") or "").strip() or None,
        metric="bird",
    ) for i, ex in enumerate(json.loads(found[0].read_text()))]


def load_split(split: str, root: str | Path = DEFAULT_ROOT) -> list[Example]:
    root = Path(root)
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split} (expected one of: {', '.join(SPLITS)})")
    files, db_dir = SPLITS[split]
    examples = []
    for name in files:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `python scripts/download_spider.py` first.")
        for ex in json.loads(path.read_text()):
            examples.append(Example(
                id=f"{split}-{len(examples)}",
                question=ex["question"].strip(),
                gold_sql=ex["query"].strip(),
                db_id=ex["db_id"],
                db_path=root / db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite",
                hardness=hardness(ex["sql"]) if "sql" in ex else None,
            ))
    return examples


def repo_relative(path: Path) -> str:
    """Path relative to the repo, so that files produced locally stay valid once copied to Colab."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_db(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=1024)
def schema_prompt(db_path: Path, sample_rows: int = 3) -> str:
    """Compact CREATE TABLE statements, each followed by a few sample rows.

    Sample rows show the actual value formats ('F' or 'female', dates, casing), where many text-to-SQL errors
    happen. They lengthen the prompt: `sample_rows=0` removes them.
    """
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    parts = []
    for name, sql in tables:
        parts.append(re.sub(r"\s+", " ", sql).strip() + ";")
        if sample_rows:
            res = execute(db_path, f'SELECT * FROM "{name}" LIMIT {sample_rows}', timeout_s=2.0)
            if res.ok and res.rows:
                cells = (" | ".join("NULL" if v is None else _short(v) for v in row) for row in res.rows)
                parts.append(f"/* {len(res.rows)} rows from {name}: " + " ; ".join(cells) + " */")
    return "\n".join(parts)


def _short(v, n: int = 30) -> str:
    s = re.sub(r"\s+", " ", str(v))
    return s if len(s) <= n else s[: n - 1] + "…"
