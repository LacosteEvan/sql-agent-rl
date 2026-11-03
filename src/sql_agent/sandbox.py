"""Isolated SQL execution: read-only database, one statement, bounded time and row count."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False  # more than max_rows rows: `rows` keeps only the first max_rows
    elapsed_s: float = 0.0


# ATTACH would let a model-written query read any file on disk.
_DENIED = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}


def _authorizer(action, *_):
    return sqlite3.SQLITE_DENY if action in _DENIED else sqlite3.SQLITE_OK


def execute(db_path: str | Path, sql: str, timeout_s: float = 5.0, max_rows: int = 10_000) -> ExecResult:
    """Run `sql` against the SQLite database at `db_path`. Never raises: errors are returned to the model."""
    start = time.monotonic()
    deadline = start + timeout_s
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error as exc:
        return ExecResult(ok=False, error=f"Cannot open database: {exc}")
    # Some Spider databases contain invalid UTF-8.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    conn.set_authorizer(_authorizer)
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)  # sqlite3 rejects several statements separated by ";"
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return ExecResult(ok=True, columns=columns, rows=rows[:max_rows], truncated=truncated,
                          elapsed_s=time.monotonic() - start)
    except sqlite3.OperationalError as exc:
        msg = f"Timeout after {timeout_s:g}s" if str(exc) == "interrupted" else str(exc)
        return ExecResult(ok=False, error=msg, elapsed_s=time.monotonic() - start)
    except (sqlite3.Error, ValueError, OverflowError) as exc:
        return ExecResult(ok=False, error=f"{type(exc).__name__}: {exc}", elapsed_s=time.monotonic() - start)
    finally:
        conn.close()


def format_observation(res: ExecResult, max_rows: int = 10, max_chars: int = 60) -> str:
    """Render a result for the model: the error, or a header and the first rows."""
    if not res.ok:
        return f"ERROR: {res.error}"
    if not res.columns:
        return "OK (no rows returned)"

    def cell(v) -> str:
        s = "NULL" if v is None else str(v)
        return s if len(s) <= max_chars else s[: max_chars - 1] + "…"

    n = len(res.rows)
    count = f"{n}+ rows" if res.truncated else f"{n} row{'s' if n != 1 else ''}"
    lines = [f"{count}, showing {min(n, max_rows)}", " | ".join(res.columns)]
    lines += [" | ".join(cell(v) for v in row) for row in res.rows[:max_rows]]
    return "\n".join(lines)
