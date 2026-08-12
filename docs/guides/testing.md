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

## LLM Mocking

Tests use `unittest.mock.AsyncMock` for deterministic LLM responses. Fixtures are defined in `tests/conftest.py`:

### Fixtures

| Fixture | Description |
|---|---|
| `mock_llm` | Returns parseable responses — CLAIM-format for extraction prompts, JSON with `supported` verdict for verification |
| `mock_llm_contradicted` | Same as `mock_llm` but returns `contradicted` verdict for verification |
| `live_llm` | Real LLM for `@pytest.mark.llm` tests. Reads config from `.env` |

### Writing tests

```python
from unittest.mock import AsyncMock
import pytest

@pytest.fixture
def my_mock_llm():
    llm = AsyncMock()
    async def _respond(prompt: str, **kwargs) -> str:
        return '{"verdict": "SUPPORTED", "evidence": "...", "explanation": "..."}'
    llm.generate = AsyncMock(side_effect=_respond)
    return llm
```

### Live LLM tests

Tests that require a real LLM are marked with `@pytest.mark.llm` and skipped by default:

```python
@pytest.mark.llm
async def test_something(live_llm):
    checker = RAGFactsChecker(live_llm)
    ...
```

Run them with `pytest -m llm`.

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
