# RAG Facts Check — Makefile
#
# Requires .env with LLM config (copy .env.example to .env).
#
# Usage:
#   make gen TOPIC=climate_change              # Generate dataset
#   make gen-all                                # Generate all topics
#   make check DATASET=climate_change_hallucinated
#   make check-all                              # Check all datasets
#   make test                                   # Run tests
#   make example                                # Run example_usage.py

PYTHON := python3
GEN := .agents/skills/synth-rag-dataset/scripts/generate_dataset.py
CHK := scripts/check_dataset.py
DATASETS := mock_datasets

TOPIC ?= pollution
DATASET ?= climate_change_hallucinated

# ─── Dataset Generation ──────────────────────────────────────────────────────

.PHONY: gen gen-all

gen:
	$(PYTHON) $(GEN) --topics $(TOPIC) --hallucination-rate 0.5 -n 1 \
		--num-docs 6 --doc-chunk-size 80 \
		-o $(DATASETS)/$(TOPIC)_generated.jsonl --verbose

gen-all:
	$(PYTHON) $(GEN) -n 10 --hallucination-rate 0.3 --num-docs 6 \
		--doc-chunk-size 80 -o $(DATASETS)/all_generated.jsonl --verbose

# ─── Fact Checking ───────────────────────────────────────────────────────────

.PHONY: check check-all

check:
	$(PYTHON) $(CHK) $(DATASETS)/$(DATASET).json

check-all:
	$(PYTHON) $(CHK) $(DATASETS)/*.json

# ─── Testing ─────────────────────────────────────────────────────────────────

.PHONY: test test-verbose test-coverage

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-verbose:
	$(PYTHON) -m pytest tests/ -vv --tb=long

test-coverage:
	$(PYTHON) -m pytest tests/ -v --cov=rag_facts_check --cov-report=term-missing

# ─── Examples ────────────────────────────────────────────────────────────────

.PHONY: example

example:
	$(PYTHON) example_usage.py

# ─── Utility ─────────────────────────────────────────────────────────────────

.PHONY: clean list

clean:
	rm -f $(DATASETS)/*_generated.jsonl $(DATASETS)/all_*.jsonl
	rm -f report.json
	rm -rf .pytest_cache .coverage htmlcov

list:
	@echo "Datasets:"; ls -1 $(DATASETS)/*.json 2>/dev/null | xargs -I{} basename {} .json
