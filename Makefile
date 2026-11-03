PY := .venv/bin/python

.PHONY: setup data test lint sft-data check-gold report report-bird

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

report:           ## README table + chart from the committed results
	$(PY) -m sql_agent report \
		"Qwen2.5-7B, 1 turn=results/spider-dev-200/qwen7b_t1.summary.json" \
		"Qwen2.5-7B, 4 turns=results/spider-dev-200/qwen7b_t4.summary.json" \
		"Claude Sonnet 5.5, 4 turns=results/spider-dev-200/sonnet55_t4.summary.json" \
		--chart docs/results

report-bird:      ## same, on BIRD
	$(PY) -m sql_agent report \
		"Qwen2.5-7B, 1 turn=results/bird-dev-200/qwen7b_t1.summary.json" \
		"Qwen2.5-7B, 4 turns=results/bird-dev-200/qwen7b_t4.summary.json" \
		"Claude Sonnet 5.5, 4 turns=results/bird-dev-200/sonnet55_t4.summary.json" \
		--chart docs/results-bird
