# RAG Facts Check — Makefile
#
# Requires .env with LLM config (copy .env.example to .env).

PYTHON := python3
GEN    := .agents/skills/synth-rag-dataset/scripts/generate_dataset.py
CHK    := scripts/check_dataset.py
DATASETS := mock_datasets

TOPIC ?= pollution
DATASET ?= climate_change_hallucinated

# Dev tool paths
UV     := uv
VENV   := .venv
RUFF   := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest

.DEFAULT_GOAL := help

## help: Show this help
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════╗"
	@echo "  ║           RAG Facts Check — Fact-Checking            ║"
	@echo "  ╚══════════════════════════════════════════════════════╝"
	@echo ""
	@grep -hE '^## [a-zA-Z]' $(MAKEFILE_LIST) | \
		sed 's/^## \([^:]*\):\(.*\)/  \1  \2/' | \
		awk '{printf "  %-20s %s\n", $$1, substr($$0, index($$0,$$2))}' | \
		sort

# --- Setup ---

## setup-dev: Create venv and install all dependencies (dev + test + server)
setup-dev:
	@if [ ! -d $(VENV) ]; then $(UV) venv --python python3 $(VENV); else echo "Using existing $(VENV)"; fi
	$(UV) pip install -p $(PYTHON) -e ".[test,dev,server]"

## setup: Create venv and install project with dev deps (alias for setup-dev)
setup: setup-dev

## setup-server: Create venv and install project with server deps (alias for setup-dev)
setup-server: setup-dev

## install-hooks: Install git pre-commit hook (lint + format-check)
install-hooks:
	@cp scripts/hooks/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Git hooks installed."

# --- Testing ---

## test: Run tests
test:
	$(PYTEST) tests/

## test-verbose: Run tests with full output
test-verbose:
	$(PYTEST) tests/ -vv --tb=long

## test-coverage: Run tests with coverage report
test-coverage:
	$(PYTEST) tests/ -v --cov=rag_facts_check --cov-report=term-missing

# --- Linting ---

## lint: Run ruff linter
lint:
	$(RUFF) check rag_facts_check/ tests/ scripts/

## lint-fix: Auto-fix lint issues
lint-fix:
	$(RUFF) check --fix rag_facts_check/ tests/ scripts/

## format: Format code with ruff
format:
	$(RUFF) format rag_facts_check/ tests/ scripts/

## format-check: Check formatting (no writes)
format-check:
	$(RUFF) format --check rag_facts_check/ tests/ scripts/

# --- Running ---

## example: Run example_usage.py
example:
	$(PYTHON) example_usage.py

# --- Server ---

## serve: Start the FastAPI server (localhost:8000, verbose)
serve: setup-dev
	PYTHONDONTWRITEBYTECODE=1 $(VENV)/bin/uvicorn rag_facts_check.server:app --reload --host 0.0.0.0 --port 8000 --log-level info

## serve-prod: Start the server in production mode
serve-prod:
	$(VENV)/bin/uvicorn rag_facts_check.server:app --host 0.0.0.0 --port 8000

## gen: Generate a dataset (TOPIC=, default: pollution)
gen:
	$(PYTHON) $(GEN) --topics $(TOPIC) --hallucination-rate 0.5 -n 1 \
		--num-docs 6 --doc-chunk-size 80 \
		-o $(DATASETS)/$(TOPIC)_generated.json --verbose

## gen-all: Generate datasets for all topics
gen-all:
	$(PYTHON) $(GEN) -n 10 --hallucination-rate 0.3 --num-docs 6 \
		--doc-chunk-size 80 -o $(DATASETS)/all_generated.jsonl --verbose

## check: Check a dataset (make check DATASET_PATH=path/to/file.json)
check:
	$(PYTHON) $(CHK) $(or $(DATASET_PATH),$(DATASETS)/$(DATASET).json)

## check-v: Check a dataset with detailed evidence (make check-v DATASET_PATH=path/to/file.json)
check-v:
	$(PYTHON) $(CHK) --verbose $(or $(DATASET_PATH),$(DATASETS)/$(DATASET).json)

## check-all: Check all datasets
check-all:
	$(PYTHON) $(CHK) $(DATASETS)/*.json

## check-all-v: Check all datasets with detailed evidence
check-all-v:
	$(PYTHON) $(CHK) --verbose $(DATASETS)/*.json

## list: List available datasets
list:
	@echo "Datasets:"; ls -1 $(DATASETS)/*.json 2>/dev/null | xargs -I{} basename {} .json

# --- Maintenance ---

## clean: Remove all build artifacts and caches
clean:
	rm -rf $(VENV)/ .pytest_cache/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f $(DATASETS)/*_generated.jsonl $(DATASETS)/all_*.jsonl
	rm -f report.json
