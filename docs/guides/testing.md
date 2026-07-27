---
type: Framework
title: Testing
description: Test suite, MockLLM, and mock datasets.
tags: [testing, mockllm, datasets]
timestamp: '2025-01-01T00:00:00Z'
---

# Testing

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_checker.py -v

# Run with coverage
python -m pytest tests/ --cov=rag_facts_check --cov-report=term-missing

# Run integration tests only
python -m pytest tests/test_integration.py -v
```

### Markers

- `@pytest.mark.llm` — tests that require a live LLM. Skipped by default. Run with `pytest -m llm`.

## MockLLM

The `MockLLM` class (in `rag_facts_check/testing/mocks.py`) provides deterministic responses based on keyword matching, enabling testing without a real LLM.

```python
from rag_facts_check.testing import MockLLM

llm = MockLLM()
# llm.generate(prompt) returns predefined responses based on prompt content
# Tracks call_count for test assertions
```

## Test Datasets

Mock datasets in `mock_datasets/` provide realistic test cases:

| Dataset | Documents | Claims | Notes |
|---|---|---|---|
| `climate_change_hallucinated.json` | 6 chunks | 5 | Hallucinated values: 5.7°C vs 2-4°C, IPCC 2024 vs 2023, Arctic ice-free by 2035 vs 2040-2060 |
| `renewable_energy_supported.json` | 6 chunks | 6 | All claims supported: 30%, 42%, 89%, 340 GW |
| `phosphorus_eutrophication.json` | 6 EEA documents | 22+ | Real production data from the climate adapt chatbot |

## Running Examples

```bash
python example_usage.py
```
