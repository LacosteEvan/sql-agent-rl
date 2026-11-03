"""Bootstrap confidence intervals: on 200 questions, a 2-point gap can be pure noise."""
from __future__ import annotations

import random


def bootstrap_ci(values: list[float], n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    return means[int(alpha / 2 * n_resamples)], means[int((1 - alpha / 2) * n_resamples) - 1]


def paired_bootstrap(a: list[float], b: list[float], n_resamples: int = 2000, seed: int = 0) -> dict:
    """Compare two models evaluated on the SAME questions: pairing removes the variance due to the questions,
    which is much larger than the variance due to the models. `p_not_better` estimates P(b does not beat a)."""
    if len(a) != len(b) or not a:
        raise ValueError("Expected two non-empty lists of the same length.")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rng, n = random.Random(seed), len(diffs)
    deltas = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    return {
        "n": n,
        "delta": sum(diffs) / n,
        "ci95": (deltas[int(0.025 * n_resamples)], deltas[int(0.975 * n_resamples) - 1]),
        "p_not_better": sum(d <= 0 for d in deltas) / n_resamples,
    }
