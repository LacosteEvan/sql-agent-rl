PY := .venv/bin/python

.PHONY: setup data test lint sft-data check-gold report report-bird report-test report-grpo

setup:            ## virtualenv + dependencies
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:             ## download Spider (~200 MB) and BIRD dev (~350 MB)
	$(PY) scripts/download_spider.py
	$(PY) scripts/download_bird.py

test:             ## unit tests (no data, no API)
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

check-gold:       ## the reward must accept every gold query
	$(PY) -m sql_agent check-gold --split dev
	$(PY) -m sql_agent check-gold --dataset bird --split dev

sft-data:         ## free SFT/GRPO data from the gold queries
	$(PY) -m sql_agent build-sft --split train --out data/sft.jsonl

SPIDER := results/spider-dev-200
BIRD := results/bird-dev-200
TEST := results/spider-test

report:           ## README table + chart from the committed results (the chart holds 6 runs at most)
	$(PY) -m sql_agent report \
		"Qwen2.5-Coder-1.5B, 1 turn=$(SPIDER)/coder1.5b_t1.summary.json" \
		"Qwen2.5-Coder-1.5B, 4 turns=$(SPIDER)/coder1.5b_t4.summary.json" \
		"Qwen2.5-Coder-1.5B, 4 turns, majority of 5=$(SPIDER)/coder1.5b_t4_k5.summary.json" \
		"Qwen2.5-Coder-1.5B + SFT=$(SPIDER)/coder1.5b_sft_t4.summary.json" \
		"Qwen2.5-Coder-1.5B + SFT + GRPO=$(SPIDER)/coder1.5b_grpo_t4.summary.json" \
		"Qwen2.5-7B, 1 turn=$(SPIDER)/qwen7b_t1.summary.json" \
		"Qwen2.5-7B, 4 turns=$(SPIDER)/qwen7b_t4.summary.json" \
		"Claude Sonnet 5.5, 4 turns=$(SPIDER)/sonnet55_t4.summary.json"
	$(PY) -m sql_agent report \
		"Qwen2.5-Coder-1.5B, 1 turn=$(SPIDER)/coder1.5b_t1.summary.json" \
		"Qwen2.5-Coder-1.5B, 4 turns=$(SPIDER)/coder1.5b_t4.summary.json" \
		"Qwen2.5-Coder-1.5B + SFT=$(SPIDER)/coder1.5b_sft_t4.summary.json" \
		"Qwen2.5-Coder-1.5B + SFT + GRPO=$(SPIDER)/coder1.5b_grpo_t4.summary.json" \
		"Qwen2.5-7B, 4 turns=$(SPIDER)/qwen7b_t4.summary.json" \
		"Claude Sonnet 5.5, 4 turns=$(SPIDER)/sonnet55_t4.summary.json" \
		--chart docs/results > /dev/null

report-bird:      ## same, on BIRD
	$(PY) -m sql_agent report \
		"Qwen2.5-Coder-1.5B, 4 turns=$(BIRD)/coder1.5b_t4.summary.json" \
		"Qwen2.5-Coder-1.5B + SFT + GRPO=$(BIRD)/coder1.5b_grpo_t4.summary.json" \
		"Qwen2.5-7B, 1 turn=$(BIRD)/qwen7b_t1.summary.json" \
		"Qwen2.5-7B, 4 turns=$(BIRD)/qwen7b_t4.summary.json" \
		"Claude Sonnet 5.5, 4 turns=$(BIRD)/sonnet55_t4.summary.json" \
		--chart docs/results-bird

report-test:      ## final Spider test set (2,147 questions), the trained models only
	$(PY) -m sql_agent report \
		"Qwen2.5-Coder-1.5B, 4 turns=$(TEST)/coder1.5b_t4.summary.json" \
		"Qwen2.5-Coder-1.5B + SFT=$(TEST)/coder1.5b_sft_t4.summary.json" \
		"Qwen2.5-Coder-1.5B + SFT + GRPO=$(TEST)/coder1.5b_grpo_t4.summary.json"

report-grpo:      ## GRPO training reward, from the CSV exported from Weights & Biases
	$(PY) scripts/plot_reward.py results/grpo/train_reward.csv docs/grpo-reward
