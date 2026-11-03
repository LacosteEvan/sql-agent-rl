"""README chart of the GRPO training reward, from the CSV exported from Weights & Biases.

    python scripts/plot_reward.py results/grpo/train_reward.csv docs/grpo-reward
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from sql_agent.report import reward_curve


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader(src.open()))
    steps, rewards = [int(r["step"]) for r in rows], [float(r["reward"]) for r in rows]
    for path in reward_curve(steps, rewards, out):
        print(f"chart → {path}", file=sys.stderr)
    first, last = rewards[: len(rewards) // 4], rewards[-len(rewards) // 4:]
    print(f"mean reward, first quarter {sum(first) / len(first):.3f}, last quarter {sum(last) / len(last):.3f}")


if __name__ == "__main__":
    main()
