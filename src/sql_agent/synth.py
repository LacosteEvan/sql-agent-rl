"""Training data built without any paid call.

- Direct answer: the question and schema, then the gold query.
- Correction: the gold query is broken (misnamed column or table), the sandbox returns the REAL SQLite error,
  and the target is the correct query. The model learns to read an error and fix it, without ever being
  trained to produce the broken query: it only appears in the prompt.

Examples are split turn by turn in prompt / completion format: the loss only covers the expected reply,
and the same prompts are reused for GRPO.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from .agent import feedback_message, initial_messages
from .data import Example, repo_relative
from .reward import GoldError, score
from .sandbox import ExecResult, execute

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_QUOTED = re.compile(r"'(?:[^']|'')*'|\"[^\"]*\"")
_FAILED_PREFIXES = ("FORMAT ERROR", "Your answer failed", "<result>\nERROR")


def clean_sql(sql: str) -> str:
    """Spider gold queries are tokenized ("T1.name ,  T2.age"): restore normal spacing."""
    sql = re.sub(r"\s+", " ", sql).strip()
    return re.sub(r"\s+,", ",", sql)


@lru_cache(maxsize=512)
def _identifiers(db_path: Path) -> frozenset[str]:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        cols = [r[1] for t in tables for r in conn.execute(f'PRAGMA table_info("{t}")')]
    finally:
        conn.close()
    return frozenset(n.lower() for n in tables + cols)


def _mutations(name: str) -> list[str]:
    out = [name + "s", name + "_name", name.replace("_", "")]
    if len(name) > 3:
        out.append(name[:-1])
    return [m for m in out if m and m != name]


def corrupt(sql: str, db_path: Path, rng: random.Random) -> tuple[str, ExecResult] | None:
    """Replace one identifier of the query with a non-existent variant. The broken query is only kept if
    SQLite actually rejects it, so that the error message in the prompt is genuine."""
    known = _identifiers(db_path)
    quoted = [m.span() for m in _QUOTED.finditer(sql)]
    spans = [m for m in _IDENT.finditer(sql)
             if m.group().lower() in known and not any(a <= m.start() < b for a, b in quoted)]
    rng.shuffle(spans)
    for m in spans[:3]:
        candidates = [c for c in _mutations(m.group()) if c.lower() not in known]
        if not candidates:
            continue
        broken = sql[: m.start()] + rng.choice(candidates) + sql[m.end():]
        res = execute(db_path, broken, timeout_s=2.0)
        if not res.ok:
            return broken, res
    return None


def build_examples(examples: list[Example], max_turns: int = 4, sample_rows: int = 3,
                   correction_rate: float = 0.3, seed: int = 0) -> list[dict]:
    """One row per turn to learn: {prompt, completion, gold_sql, db_path, kind, example_id}."""
    rng, rows = random.Random(seed), []
    for ex in examples:
        try:
            gold = clean_sql(ex.gold_sql)
            if not score(gold, ex.gold_sql, ex.db_path).correct:
                gold = ex.gold_sql  # cleaning touched a string literal: keep the original
        except GoldError:
            continue  # gold query does not run (3 cases in Spider's train set)
        prompt = initial_messages(ex, max_turns, sample_rows)
        meta = {"gold_sql": gold, "db_path": repo_relative(ex.db_path), "example_id": ex.id}
        rows.append({"prompt": prompt, "completion": [_answer(gold)], "kind": "direct", **meta})
        if rng.random() < correction_rate and (broken := corrupt(gold, ex.db_path, rng)):
            bad_sql, res = broken
            fix_prompt = prompt + [{"role": "assistant", "content": f"<answer>{bad_sql}</answer>"},
                                   feedback_message(res, max_turns - 1, after_answer=True)]
            rows.append({"prompt": fix_prompt, "completion": [_answer(gold)], "kind": "correction", **meta})
    return rows


def rows_from_run(records: list[dict]) -> list[dict]:
    """Split the SUCCESSFUL episodes of a run (e.g. a strong model on the train set) into turns to learn.
    Turns followed by an error stay in the context but are not learned."""
    rows = []
    for rec in records:
        if not rec.get("correct") or not rec.get("messages"):
            continue
        msgs = rec["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            nxt = msgs[i + 1]["content"] if i + 1 < len(msgs) else ""
            if nxt.startswith(_FAILED_PREFIXES):
                continue
            rows.append({"prompt": msgs[:i], "completion": [m], "kind": "distilled",
                         "gold_sql": rec["gold_sql"], "db_path": rec.get("db_path", ""),
                         "example_id": rec["example_id"]})
    return rows


def _answer(sql: str) -> dict:
    return {"role": "assistant", "content": f"<answer>{sql}</answer>"}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
