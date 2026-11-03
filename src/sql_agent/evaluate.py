"""Evaluate a model on a split: execution accuracy with a 95% CI, per-difficulty breakdown,
self-correction gain, optional majority voting, cost."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import Episode, run_episode
from .data import Example, repo_relative
from .llm import LLM, BudgetExceeded, LLMError
from .reward import GoldError, result_signature, score
from .sandbox import execute
from .stats import bootstrap_ci

_BUDGET = "budget"  # example not played: spending cap reached
LEVELS = ["easy", "medium", "hard", "extra", "simple", "moderate", "challenging"]  # Spider, then BIRD


@dataclass
class Record:
    example_id: str
    db_id: str
    db_path: str  # repo-relative: reusable by `build-sft --from-run`
    hardness: str | None
    question: str
    gold_sql: str
    final_sql: str | None
    correct: bool
    first_correct: bool  # the very first query attempted was already right
    any_correct: bool  # at least one sample is right (pass@k; equals `correct` with a single sample)
    answered: bool
    n_turns: int
    n_exec_errors: int
    n_format_errors: int
    tokens: int
    cost_usd: float
    error: str | None = None  # API error: the example counts as a miss
    messages: list[dict] | None = None  # conversation of the first sample


def majority_vote(db_path: Path, sqls: list[str | None]) -> str | None:
    """Keep the query whose RESULT comes up most often: two queries written differently but equivalent
    vote together. Ties go to the earliest sample."""
    sigs = {}
    for sql in sqls:
        if sql and sql not in sigs:
            sigs[sql] = result_signature(execute(db_path, sql, timeout_s=5.0, max_rows=10_000))
    counts = Counter(sigs[s] for s in sqls if s and sigs[s] is not None)
    if not counts:
        return next((s for s in sqls if s), None)
    best = counts.most_common(1)[0][0]
    return next(s for s in sqls if s and sigs[s] == best)


def evaluate_one(llm: LLM, ex: Example, max_turns: int, sample_rows: int = 3, samples: int = 1):
    """A Record; None if the gold query fails (example dropped); or _BUDGET."""
    episodes, err = [], None
    try:
        for _ in range(samples):
            episodes.append(run_episode(llm, ex, max_turns=max_turns, sample_rows=sample_rows))
    except BudgetExceeded:
        if not episodes:
            return _BUDGET
    except LLMError as exc:
        err = str(exc)
    if not episodes:
        episodes = [Episode(example_id=ex.id, messages=[])]
    first = episodes[0]
    final_sql = first.final_sql if len(episodes) == 1 else majority_vote(ex.db_path, [e.final_sql for e in episodes])

    try:
        verdicts = {sql: score(sql, ex.gold_sql, ex.db_path, metric=ex.metric).correct
                    for sql in {final_sql, first.first_sql, *(e.final_sql for e in episodes)}}
    except GoldError:
        return None
    usage = [c for e in episodes for c in e.usage]
    return Record(
        example_id=ex.id, db_id=ex.db_id, db_path=repo_relative(ex.db_path), hardness=ex.hardness,
        question=ex.question, gold_sql=ex.gold_sql, final_sql=final_sql, correct=verdicts[final_sql],
        first_correct=verdicts[first.first_sql], any_correct=any(verdicts[e.final_sql] for e in episodes),
        answered=first.answered, n_turns=first.n_turns, n_exec_errors=first.n_exec_errors,
        n_format_errors=first.n_format_errors, tokens=sum(c.prompt_tokens + c.completion_tokens for c in usage),
        cost_usd=sum(c.cost_usd for c in usage), error=err, messages=first.messages,
    )


def evaluate(llm: LLM, examples: list[Example], max_turns: int = 4, sample_rows: int = 3, samples: int = 1,
             workers: int = 8, out: Path | None = None, on_done: Callable[[int], None] | None = None) -> dict:
    records: list[Record] = []
    skipped_gold = skipped_budget = 0
    out_f = out.open("w") if out else None
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = pool.map(lambda ex: evaluate_one(llm, ex, max_turns, sample_rows, samples), examples)
            for i, rec in enumerate(jobs, 1):
                if rec is None:
                    skipped_gold += 1
                elif rec is _BUDGET:
                    skipped_budget += 1
                else:
                    records.append(rec)
                    if out_f:
                        out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                if on_done:
                    on_done(i)
    finally:
        if out_f:
            out_f.close()
    return summarize(records) | {"skipped_gold_errors": skipped_gold, "not_run_budget": skipped_budget}


def summarize(records: list[Record]) -> dict:
    n = len(records) or 1
    correct = [float(r.correct) for r in records]
    by_level = defaultdict(list)
    for r in records:
        by_level[r.hardness or "unknown"].append(r.correct)
    return {
        "n": len(records),
        "execution_accuracy": sum(correct) / n,
        "execution_accuracy_ci95": bootstrap_ci(correct),
        "first_attempt_accuracy": sum(r.first_correct for r in records) / n,
        "pass_at_k": sum(r.any_correct for r in records) / n,
        "by_hardness": {k: {"n": len(by_level[k]), "accuracy": sum(by_level[k]) / len(by_level[k])}
                        for k in [*LEVELS, *sorted(set(by_level) - set(LEVELS))] if by_level[k]},
        "answered_rate": sum(r.answered for r in records) / n,
        "avg_turns": sum(r.n_turns for r in records) / n,
        "exec_errors_per_episode": sum(r.n_exec_errors for r in records) / n,
        "api_errors": sum(r.error is not None for r in records),
        "total_tokens": sum(r.tokens for r in records),
        "total_cost_usd": round(sum(r.cost_usd for r in records), 4),
    }


def load_records(path: Path) -> dict[str, dict]:
    return {r["example_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}
