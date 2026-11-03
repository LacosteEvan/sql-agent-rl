"""LoRA SFT on the build-sft data (prompt / completion format: the loss only covers the reply).

    python train/sft.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --data data/sft.jsonl --out outputs/sft
"""
from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from _common import precision


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--data", default="data/sft.jsonl")
    p.add_argument("--out", default="outputs/sft")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs (smoke tests)")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    mixed, dtype = precision()
    ds = load_dataset("json", data_files=args.data, split="train").select_columns(["prompt", "completion"])
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        max_length=args.max_length,
        **mixed,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb" if args.wandb else "none",
        model_init_kwargs={"dtype": dtype},
    )
    peft = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=ds, peft_config=peft)
    trainer.train()
    trainer.save_model(args.out)
    print(f"LoRA adapter saved to {args.out}: merge it with train/merge.py before serving it with vLLM.")


if __name__ == "__main__":
    main()
