# RAG Facts Check — Makefile
#
# Usage:
#   make gen TOPIC=climate_change              # Generate with mock LLM
#   make gen-real TOPIC=eutrophication          # Generate with real LLM (.env)
#   make check DATASET=climate_change_hallucinated
#   make check-real DATASET=climate_change_hallucinated
#   make check-all                              # Check all datasets (mock)
#   make test                                   # Run all tests
#   make example                                # Run example_usage.py

PYTHON := python3
GEN := .agents/skills/synth-rag-dataset/scripts/generate_dataset.py
CHK := scripts/check_dataset.py
DATASETS := mock_datasets

TOPIC ?= pollution
DATASET ?= climate_change_hallucinated

# ─── Dataset Generation ──────────────────────────────────────────────────────

.PHONY: gen gen-real gen-all-mock gen-all-real

gen:
	$(PYTHON) $(GEN) --topics $(TOPIC) --hallucination-rate 0.5 -n 1 \
		--num-docs 6 --doc-chunk-size 80 --mock \
		-o $(DATASETS)/$(TOPIC)_mock.jsonl --verbose

gen-real:
	$(PYTHON) $(GEN) --topics $(TOPIC) --hallucination-rate 0.5 -n 1 \
		--num-docs 6 --doc-chunk-size 80 --llm-backend env \
		-o $(DATASETS)/$(TOPIC)_real.jsonl --verbose

gen-all-mock:
	$(PYTHON) $(GEN) -n 10 --hallucination-rate 0.3 --num-docs 6 \
		--doc-chunk-size 80 --mock -o $(DATASETS)/all_mock.jsonl --verbose

gen-all-real:
	$(PYTHON) $(GEN) -n 10 --hallucination-rate 0.3 --num-docs 6 \
		--doc-chunk-size 80 --llm-backend env -o $(DATASETS)/all_real.jsonl --verbose

# ─── Fact Checking ───────────────────────────────────────────────────────────

.PHONY: check check-real check-all check-all-real

check:
	$(PYTHON) $(CHK) $(DATASETS)/$(DATASET).json

check-real:
	$(PYTHON) $(CHK) --real $(DATASETS)/$(DATASET).json

check-all:
	$(PYTHON) $(CHK) $(DATASETS)/*.json

check-all-real:
	$(PYTHON) $(CHK) --real $(DATASETS)/*.json

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
	rm -f $(DATASETS)/*_mock.jsonl $(DATASETS)/*_real.jsonl $(DATASETS)/all_*.jsonl
	rm -f report.json
	rm -rf .pytest_cache .coverage htmlcov

list:
	@echo "Datasets:"; ls -1 $(DATASETS)/*.json 2>/dev/null | xargs -I{} basename {} .json
