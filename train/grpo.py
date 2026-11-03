"""LoRA GRPO with the execution reward, on the prompts kept by filter_prompts.py.

Turn-level training: each prompt is either a fresh question or a conversation state where the previous answer
failed with a real SQLite error. The model thus learns to answer right the first time AND to correct itself,
with the standard GRPOTrainer (stable API), without multi-turn rollouts.

    python train/grpo.py --model outputs/sft/merged --data data/grpo.jsonl --out outputs/grpo --wandb
"""
from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from _common import precision
from sql_agent.rl import correctness_reward, format_reward


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ideally the merged SFT model")
    p.add_argument("--data", default="data/grpo.jsonl")
    p.add_argument("--out", default="outputs/grpo")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--num-generations", type=int, default=8, help="size of the group GRPO compares")
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=384)
    p.add_argument("--vllm-memory", type=float, default=0.35, help="share of GPU memory given to vLLM (colocate)")
    p.add_argument("--no-vllm", action="store_true", help="generate with transformers (slow; smoke tests)")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    mixed, dtype = precision()
    ds = load_dataset("json", data_files=args.data, split="train").select_columns(["prompt", "gold_sql", "db_path"])
    completions_per_step = args.num_generations * args.prompts_per_step
    config = GRPOConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=completions_per_step // args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        reward_weights=[1.0, 1.0],
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_max_model_length=6144,
        gradient_checkpointing=True,
        **mixed,
        logging_steps=5,
        log_completions=True,
        num_completions_to_print=2,
        save_steps=100,
        report_to="wandb" if args.wandb else "none",
        model_init_kwargs={"dtype": dtype},
    )
    peft = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=args.model, reward_funcs=[correctness_reward, format_reward],
                          args=config, train_dataset=ds, peft_config=peft)
    trainer.train()
    trainer.save_model(args.out)
    print(f"GRPO adapter saved to {args.out}: merge it with train/merge.py before serving it with vLLM.")


if __name__ == "__main__":
    main()
