# RAG Facts Check — Makefile
#
# Requires .env with LLM config (copy .env.example to .env).

PYTHON := python3
GEN := .agents/skills/synth-rag-dataset/scripts/generate_dataset.py
CHK := scripts/check_dataset.py
DATASETS := mock_datasets

TOPIC ?= pollution
DATASET ?= climate_change_hallucinated

## help: Show this help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -hE '^## [a-zA-Z]' $(MAKEFILE_LIST) | \
		sed 's/^## \([^:]*\):\(.*\)/  \1  \2/' | \
		awk '{printf "  %-20s %s\n", $$1, substr($$0, index($$0,$$2))}' | \
		sort

## gen: Generate a dataset (TOPIC=, default: pollution)
gen:
	$(PYTHON) $(GEN) --topics $(TOPIC) --hallucination-rate 0.5 -n 1 \
		--num-docs 6 --doc-chunk-size 80 \
		-o $(DATASETS)/$(TOPIC)_generated.json --verbose

## gen-all: Generate datasets for all topics
gen-all:
	$(PYTHON) $(GEN) -n 10 --hallucination-rate 0.3 --num-docs 6 \
		--doc-chunk-size 80 -o $(DATASETS)/all_generated.jsonl --verbose

## check: Check a dataset (DATASET=, default: climate_change_hallucinated)
check:
	$(PYTHON) $(CHK) $(DATASETS)/$(DATASET).json

## check-v: Check a dataset with detailed evidence (DATASET=)
check-v:
	$(PYTHON) $(CHK) --verbose $(DATASETS)/$(DATASET).json

## check-all: Check all datasets
check-all:
	$(PYTHON) $(CHK) $(DATASETS)/*.json

## check-all-v: Check all datasets with detailed evidence
check-all-v:
	$(PYTHON) $(CHK) --verbose $(DATASETS)/*.json

## test: Run tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

## test-verbose: Run tests with full output
test-verbose:
	$(PYTHON) -m pytest tests/ -vv --tb=long

## test-coverage: Run tests with coverage report
test-coverage:
	$(PYTHON) -m pytest tests/ -v --cov=rag_facts_check --cov-report=term-missing

## example: Run example_usage.py
example:
	$(PYTHON) example_usage.py

## list: List available datasets
list:
	@echo "Datasets:"; ls -1 $(DATASETS)/*.json 2>/dev/null | xargs -I{} basename {} .json

## clean: Remove generated files
clean:
	rm -f $(DATASETS)/*_generated.jsonl $(DATASETS)/all_*.jsonl
	rm -f report.json
	rm -rf .pytest_cache .coverage htmlcov
