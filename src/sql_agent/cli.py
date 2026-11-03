"""CLI: eval, demo, compare, report, estimate, check-gold, build-sft. Only OpenRouter runs cost credits."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

from .data import DATASETS, load_dataset
from .reward import GoldError, score


def _sample(examples, limit: int | None, seed: int):
    if limit and limit < len(examples):
        return random.Random(seed).sample(examples, limit)
    return examples


def _make_llm(args, temperature: float):
    from dotenv import load_dotenv

    from .llm import Budget, ChatLLM

    load_dotenv()
    llm = ChatLLM(model=args.model, base_url=args.base_url, temperature=temperature, budget=Budget(args.max_cost))
    if llm.is_paid and args.max_cost is None:
        raise SystemExit("Paid run without --max-cost: refused. Estimate it first with `estimate`.")
    return llm


def cmd_eval(args) -> int:
    from .evaluate import evaluate
    from .llm import LLMError

    examples = _sample(load_dataset(args.dataset, args.split, args.data), args.limit, args.seed)
    temperature = args.temperature if args.temperature is not None else (0.7 if args.samples > 1 else 0.0)
    try:
        llm = _make_llm(args, temperature)
    except LLMError as exc:
        print(exc, file=sys.stderr)
        return 1
    name = re.sub(r"[^\w.-]", "_", llm.model)
    out = args.out or Path("runs") / f"{name}_{args.split}_{len(examples)}_t{args.max_turns}_k{args.samples}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def progress(done):
        print(f"\r{done}/{len(examples)}  ${llm.budget.spent_usd:.4f}", end="", file=sys.stderr, flush=True)

    start = time.monotonic()
    summary = evaluate(llm, examples, max_turns=args.max_turns, sample_rows=args.sample_rows,
                       samples=args.samples, workers=args.workers, out=out, on_done=progress)
    summary |= {"model": llm.model, "base_url": llm.base_url, "dataset": args.dataset, "split": args.split,
                "limit": args.limit,
                "seed": args.seed, "max_turns": args.max_turns, "sample_rows": args.sample_rows,
                "samples": args.samples, "temperature": temperature,
                "elapsed_s": round(time.monotonic() - start, 1), "records": str(out)}
    print(file=sys.stderr)
    print(json.dumps(summary, indent=2))
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def cmd_demo(args) -> int:
    """Play one question live, turn by turn, in the terminal."""
    from .agent import run_episode
    from .reward import score as score_sql

    examples = {ex.id: ex for ex in load_dataset(args.dataset, args.split, args.data)}
    ex = examples.get(args.id) if args.id else random.Random(args.seed).choice(list(examples.values()))
    if ex is None:
        print(f"Unknown example id: {args.id}", file=sys.stderr)
        return 1
    llm = _make_llm(args, temperature=0.0)
    bold, dim, green, red, reset = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"
    print(f"{bold}{ex.id} · {ex.db_id} · {ex.hardness}{reset}\n{bold}Q:{reset} {ex.question}\n")

    def show(message: dict) -> None:
        text = message["content"]
        if message["role"] == "assistant":
            print(f"{bold}model ›{reset} {text}\n")
        else:
            color = red if "ERROR" in text else dim
            print(f"{color}{text}{reset}\n")
        time.sleep(args.pause)

    ep = run_episode(llm, ex, max_turns=args.max_turns, on_message=show)
    verdict = score_sql(ep.final_sql, ex.gold_sql, ex.db_path, metric=ex.metric)
    mark = f"{green}✔ correct" if verdict.correct else f"{red}✘ wrong"
    print(f"{mark}{reset} after {ep.n_turns} turn(s) · gold: {dim}{ex.gold_sql}{reset}")
    return 0


def cmd_compare(args) -> int:
    """Gain of B over A, paired on the questions both runs share."""
    from .evaluate import load_records
    from .stats import paired_bootstrap

    a, b = load_records(args.a), load_records(args.b)
    ids = sorted(set(a) & set(b))
    if not ids:
        print("The two runs share no question.", file=sys.stderr)
        return 1
    res = paired_bootstrap([float(a[i]["correct"]) for i in ids], [float(b[i]["correct"]) for i in ids])
    lo, hi = res["ci95"]
    acc_a = sum(a[i]["correct"] for i in ids) / len(ids)
    acc_b = sum(b[i]["correct"] for i in ids) / len(ids)
    print(f"{res['n']} shared questions")
    print(f"A: {acc_a:.1%}   B: {acc_b:.1%}")
    print(f"B - A = {res['delta']:+.1%}  (95% CI {lo:+.1%} to {hi:+.1%}, P(B not better) = {res['p_not_better']:.3f})")
    return 0


def cmd_report(args) -> int:
    from .report import chart, load_runs, markdown_table

    runs = load_runs(args.runs)
    print(markdown_table(runs))
    if args.chart:
        for path in chart(runs, args.chart):
            print(f"chart → {path}", file=sys.stderr)
    return 0


def cmd_estimate(args) -> int:
    """Order of magnitude of a run's tokens and cost, without any call. ~3.5 characters per token."""
    from .agent import initial_messages

    examples = _sample(load_dataset(args.dataset, args.split, args.data), args.limit, args.seed)
    prompt = sum(sum(len(m["content"]) for m in initial_messages(ex, args.max_turns, args.sample_rows))
                 for ex in examples) / 3.5
    n, step = len(examples), args.out_tokens + 250  # each turn adds a reply and an observation to the context
    # Each turn resends the whole history; a fractional average counts the last turn partially.
    tokens_in = args.samples * sum(min(1.0, args.avg_turns - i) * (prompt + i * step * n)
                                   for i in range(math.ceil(args.avg_turns)))
    tokens_out = args.samples * n * args.avg_turns * args.out_tokens
    cost = (tokens_in * args.price_in + tokens_out * args.price_out) / 1e6
    print(f"{n} questions × {args.samples} sample(s), ~{args.avg_turns} turns")
    print(f"≈ {tokens_in / 1e6:.2f}M input tokens, {tokens_out / 1e6:.2f}M output tokens")
    print(f"≈ ${cost:.2f} at the given prices (${args.price_in} / ${args.price_out} per M)")
    return 0


def cmd_check_gold(args) -> int:
    """Score every gold query against itself: must give 100%, otherwise the reward has a bug."""
    examples = _sample(load_dataset(args.dataset, args.split, args.data), args.limit, args.seed)
    ok, broken = 0, []
    for ex in examples:
        try:
            ok += score(ex.gold_sql, ex.gold_sql, ex.db_path, metric=ex.metric).correct
        except GoldError as exc:
            broken.append(str(exc))
    print(f"{ok}/{len(examples) - len(broken)} gold queries recognized as correct, {len(broken)} do not run")
    for b in broken[:10]:
        print("  -", b)
    return 0 if ok == len(examples) - len(broken) else 1


def cmd_build_sft(args) -> int:
    """SFT/GRPO data: free from the gold queries, or from the successful episodes of a run."""
    from collections import Counter

    from .synth import build_examples, rows_from_run, write_jsonl

    if args.from_run:
        rows = rows_from_run([json.loads(line) for line in args.from_run.read_text().splitlines() if line])
    else:
        examples = _sample(load_dataset(args.dataset, args.split, args.data), args.limit, args.seed)
        rows = build_examples(examples, max_turns=args.max_turns, sample_rows=args.sample_rows,
                              correction_rate=args.correction_rate, seed=args.seed)
    write_jsonl(rows, args.out)
    print(f"{len(rows)} rows → {args.out}  {dict(Counter(r['kind'] for r in rows))}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sql_agent")
    p.add_argument("--data", type=Path, help="dataset folder (default: data/spider_data or data/bird)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, split="dev"):
        sp.add_argument("--dataset", default="spider", choices=DATASETS)
        sp.add_argument("--split", default=split, choices=["train", "dev", "test"])
        sp.add_argument("--limit", type=int, help="random subsample (reproducible with --seed)")
        sp.add_argument("--seed", type=int, default=0)

    def episode(sp):
        sp.add_argument("--max-turns", type=int, default=4, help="1 = no exploration, no self-correction")
        sp.add_argument("--sample-rows", type=int, default=3, help="sample rows per table in the prompt")

    def model(sp):
        sp.add_argument("--model", help="model id (default: $SQL_AGENT_MODEL)")
        sp.add_argument("--base-url", help="OpenAI-compatible API, e.g. http://localhost:8000/v1 (default: OpenRouter)")
        sp.add_argument("--max-cost", type=float, help="spending cap in $, required on OpenRouter")

    e = sub.add_parser("eval", help="evaluate a model (OpenRouter or vLLM server)")
    common(e)
    episode(e)
    model(e)
    e.add_argument("--samples", type=int, default=1, help="> 1: majority vote over results")
    e.add_argument("--temperature", type=float, help="default: 0, or 0.7 when --samples > 1")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--out", type=Path)
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("demo", help="play one question live in the terminal")
    common(d)
    model(d)
    d.add_argument("--id", help="example id, e.g. dev-530 (default: random)")
    d.add_argument("--max-turns", type=int, default=4)
    d.add_argument("--pause", type=float, default=0.8, help="seconds between messages, for screen recordings")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("compare", help="paired gain of B over A, with a 95%% CI")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("report", help="results table (markdown) and per-difficulty chart")
    r.add_argument("runs", nargs="+", help="label=path/to/run.summary.json")
    r.add_argument("--chart", type=Path, help="output stem, e.g. docs/results → results-light.png, -dark.png")
    r.set_defaults(func=cmd_report)

    s = sub.add_parser("estimate", help="estimate a run's tokens and cost, without any call")
    common(s)
    episode(s)
    s.add_argument("--samples", type=int, default=1)
    s.add_argument("--avg-turns", type=float, default=1.5)
    s.add_argument("--out-tokens", type=int, default=150, help="generated tokens per turn")
    s.add_argument("--price-in", type=float, required=True, help="$ per million input tokens")
    s.add_argument("--price-out", type=float, required=True, help="$ per million output tokens")
    s.set_defaults(func=cmd_estimate)

    g = sub.add_parser("check-gold", help="check the reward on the gold queries")
    common(g)
    g.set_defaults(func=cmd_check_gold)

    b = sub.add_parser("build-sft", help="build SFT/GRPO data (free)")
    common(b, split="train")
    episode(b)
    b.add_argument("--correction-rate", type=float, default=0.3, help="share of correction examples added")
    b.add_argument("--from-run", type=Path, help="JSONL of an eval run: keeps its successful episodes")
    b.add_argument("--out", type=Path, default=Path("data/sft.jsonl"))
    b.set_defaults(func=cmd_build_sft)

    args = p.parse_args(argv)
    return args.func(args)
