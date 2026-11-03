"""Reward functions for TRL's GRPOTrainer (signature: completions + the dataset's columns).

Deliberately free of torch: they are tested locally and imported as-is on Colab.
"""
from __future__ import annotations

from .agent import parse_action
from .data import resolve_db
from .reward import GoldError, score


def _text(completion) -> str:
    # Conversational format: [{"role": "assistant", "content": …}]; standard format: a string.
    return completion[-1]["content"] if isinstance(completion, list) else completion


def correctness_reward(completions, gold_sql, db_path, **kwargs) -> list[float | None]:
    """1 if the final answer returns the right result, 0 otherwise. Binary and checked against the database:
    no partial credit for "it runs", which would teach the model to answer SELECT 1."""
    rewards = []
    for completion, gold, db in zip(completions, gold_sql, db_path, strict=True):
        action = parse_action(_text(completion))
        if action.kind != "answer":
            rewards.append(0.0)
            continue
        try:
            rewards.append(float(score(action.query, gold, resolve_db(db)).correct))
        except GoldError:
            rewards.append(None)  # unusable example: TRL leaves it out of the computation
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """Small bonus for a well-formed reply, mostly useful at the very start of training."""
    return [0.1 if parse_action(_text(c)).kind != "invalid" else 0.0 for c in completions]
