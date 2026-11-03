"""Verifiable reward: does the predicted query return the same result as the gold query?

Two rules, each identical to its benchmark's official evaluation:
- "spider": row order only matters when the gold query has an ORDER BY; column order and names never matter.
- "bird": the two results are compared as sets of rows (order and duplicates ignored, column order kept).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .sandbox import ExecResult, execute

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


@dataclass
class Verdict:
    correct: bool
    executed: bool  # the predicted query ran without error
    error: str | None = None


class GoldError(RuntimeError):
    """The gold query itself fails: the example is unusable."""


def _norm(v):
    # 3 and 3.0 are the same result; rounding absorbs floating-point noise.
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, bytes):
        return v.hex()
    return v


def results_match(pred: ExecResult, gold: ExecResult, order_matters: bool) -> bool:
    if pred.truncated or gold.truncated:
        return False  # incomplete result: cannot conclude
    p_rows = [tuple(_norm(v) for v in r) for r in pred.rows]
    g_rows = [tuple(_norm(v) for v in r) for r in gold.rows]
    if len(p_rows) != len(g_rows):
        return False
    if not g_rows:
        return len(pred.columns) == len(gold.columns)
    n_cols = len(g_rows[0])
    if len(p_rows[0]) != n_cols:
        return False

    # Map each predicted column to a gold column with the same content,
    # so that "SELECT name, age" matches "SELECT age, name".
    def key(rows, j):
        col = [r[j] for r in rows]
        return tuple(col) if order_matters else frozenset(Counter(col).items())

    gold_keys = [key(g_rows, j) for j in range(n_cols)]
    used, mapping = set(), []
    for j in range(n_cols):
        k = key(p_rows, j)
        match = next((i for i in range(n_cols) if i not in used and gold_keys[i] == k), None)
        if match is None:
            return False
        used.add(match)
        mapping.append(match)
    inverse = [mapping.index(i) for i in range(n_cols)]
    p_rows = [tuple(r[j] for j in inverse) for r in p_rows]
    return p_rows == g_rows if order_matters else Counter(p_rows) == Counter(g_rows)


def result_signature(res: ExecResult) -> frozenset | None:
    """Fingerprint of a result, insensitive to row and column order, used for majority voting.
    Coarse (two different results can share it) but good enough to group votes."""
    if not res.ok or res.truncated:
        return None
    cols = zip(*[tuple(_norm(v) for v in r) for r in res.rows], strict=True) if res.rows else []
    return frozenset(Counter(frozenset(Counter(c).items()) for c in cols).items()) | {("n_rows", len(res.rows))}


MAX_ROWS = 1_000_000  # some BIRD gold queries return several hundred thousand rows


def score(pred_sql: str | None, gold_sql: str, db_path: str | Path, timeout_s: float = 10.0,
          metric: str = "spider", gold_timeout_s: float = 30.0) -> Verdict:
    """`gold_timeout_s` is more generous than `timeout_s`: a few BIRD gold queries take over 10s."""
    gold = execute(db_path, gold_sql, timeout_s=gold_timeout_s, max_rows=MAX_ROWS)
    if not gold.ok:
        raise GoldError(f"{gold_sql!r}: {gold.error}")
    if not pred_sql or not pred_sql.strip():
        return Verdict(correct=False, executed=False, error="No query")
    pred = execute(db_path, pred_sql, timeout_s=timeout_s, max_rows=MAX_ROWS)
    if not pred.ok:
        return Verdict(correct=False, executed=False, error=pred.error)
    if metric == "bird":
        complete = not (pred.truncated or gold.truncated)
        return Verdict(correct=complete and set(pred.rows) == set(gold.rows), executed=True)
    order_matters = bool(_ORDER_BY.search(gold_sql))
    return Verdict(correct=results_match(pred, gold, order_matters), executed=True)
