"""Merge a LoRA adapter into its base model, to serve it with vLLM or publish it on the Hub.

    python train/merge.py --adapter outputs/sft --out outputs/sft/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, dtype=torch.bfloat16)
    model.merge_and_unload().save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged model: {args.out}")


if __name__ == "__main__":
    main()
