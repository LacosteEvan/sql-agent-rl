"""Keep the prompts at the model's frontier before GRPO.

GRPO compares the replies within a group: if the model always solves (or always fails) a question, every
reward in the group is equal, the advantage is zero and the compute is wasted. We sample k replies per prompt
with vLLM and keep only the prompts whose success rate is strictly between 0 and 1.

    python train/filter_prompts.py --model outputs/sft/merged --data data/sft.jsonl --out data/grpo.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from vllm import LLM, SamplingParams

from sql_agent.rl import correctness_reward


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/sft.jsonl")
    p.add_argument("--out", default="data/grpo.jsonl")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--limit", type=int, default=4000, help="prompts tested (randomly drawn)")
    p.add_argument("--max-model-len", type=int, default=6144)
    args = p.parse_args()

    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.85)
    params = SamplingParams(n=args.k, temperature=1.0, max_tokens=384)
    outputs = llm.chat([r["prompt"] for r in rows], params)

    kept, rates = [], []
    for row, out in zip(rows, outputs, strict=True):
        completions = [o.text for o in out.outputs]
        rewards = correctness_reward(completions, [row["gold_sql"]] * args.k, [row["db_path"]] * args.k)
        if None in rewards:
            continue
        rate = sum(rewards) / args.k
        rates.append(rate)
        if 0 < rate < 1:
            kept.append({k: row[k] for k in ("prompt", "gold_sql", "db_path", "kind", "example_id")})

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    solved = sum(r == 1 for r in rates) / len(rates)
    never = sum(r == 0 for r in rates) / len(rates)
    print(f"{len(kept)}/{len(rates)} prompts kept → {args.out}")
    print(f"always solved: {solved:.0%}, never solved: {never:.0%}, mean success rate: {sum(rates) / len(rates):.0%}")


if __name__ == "__main__":
    main()
