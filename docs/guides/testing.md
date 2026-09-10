---
type: Framework
title: Testing
description: Test suite layout, module coverage, MockLLM fixtures, and mock datasets.
tags: [testing, mockllm, datasets, pytest]
timestamp: '2026-09-09T00:00:00Z'
---

# Testing

RAG Facts Check includes an extensive automated test suite covering chunking, claim deduplication, span alignment across line breaks, Halloumi adapter conversions, and FastAPI web endpoints.

---

## Running Tests

```bash
# Run the complete test suite
pytest

# Run with verbose output
pytest -v

# Run specific test modules
pytest tests/test_checker.py -v
pytest tests/test_spans.py -v
pytest tests/test_halloumi_adapter.py -v

# Run with test coverage report
pytest --cov=rag_facts_check --cov-report=term-missing

# Run code linter
ruff check tests/ rag_facts_check/
```

### Coverage Overview

The current test suite contains **190 tests** with a **100% pass rate**:

| Module | Purpose | Key Test Areas | Coverage |
|---|---|---|---|
| `tests/test_checker.py` | Core pipeline orchestration | `split_answer_into_chunks` (markdown tables, row integrity, header replication), claim extraction, span deduplication, batch verification, aggregation | **87%** |
| `tests/test_spans.py` | Span matching algorithms | Quote/ellipsis stripping, exact search, whitespace regex (`\s+`), punctuation regex (`[\W_]+`), fuzzy alignment | **88%** |
| `tests/test_halloumi_adapter.py` | Adapter & coordinate mapping | `_to_halloumi_format`, `joinedSources` coordinate mapping, score translation, segment creation | **92%** |
| `tests/test_models.py` | Data structures | `Claim`, `VerificationResult`, `CheckReport`, `score_label`, serialization | **100%** |
| `tests/test_retriever.py` | Evidence retrieval | Lexical keyword retrieval, `LLMEvidenceRetriever`, chunking | **95%** |
| `tests/test_server.py` | FastAPI endpoints | `POST /check`, `POST /halloumi/generate`, `GET /health` | **84%** |
| `tests/test_integration.py` | End-to-end execution | Full pipeline from input answer to final report | **85%** |

---

## Markers

- `@pytest.mark.llm` — Tests requiring a live LLM endpoint. Skipped by default. Run with:
  ```bash
  pytest -m llm
  ```

---

## LLM Mocking Architecture

Tests use `unittest.mock.AsyncMock` for deterministic, offline testing. Shared fixtures live in `tests/conftest.py`:

### Standard Fixtures

| Fixture | Description |
|---|---|
| `mock_llm` | Returns structured JSON responses: valid claims for extraction, and `supported` verdicts with verbatim evidence for verification |
| `mock_llm_contradicted` | Same as `mock_llm` but returns `contradicted` verdicts for verification |
| `mock_llm_not_enough_info` | Returns `not_enough_info` verdicts with `"N/A"` evidence |
| `live_llm` | Instantiates a real `AsyncAPILLM` configured via environment variables from `.env` |

### Writing Custom Mocks

```python
from unittest.mock import AsyncMock
import pytest
from rag_facts_check.checker import RAGFactsChecker

@pytest.fixture
def custom_mock_llm():
    llm = AsyncMock()
    async def _respond(prompt: str, **kwargs) -> str:
        if "Extract atomic factual claims" in prompt:
            return '[{"claim": "Test claim", "original_text": "verbatim text"}]'
        return '{"verdict": "SUPPORTED", "evidence": "verbatim text", "document_index": 0}'
    llm.generate = AsyncMock(side_effect=_respond)
    return llm

@pytest.mark.asyncio
async def test_pipeline(custom_mock_llm):
    checker = RAGFactsChecker(custom_mock_llm)
    report = await checker.check(
        answer="This is verbatim text.",
        documents=["This is verbatim text in source."]
    )
    assert report.answer_score > 7.0
```

---

## Test Datasets

Realistic mock datasets in `mock_datasets/` provide ground-truth benchmarks:

| Dataset | Documents | Claims | Scenarios Tested |
|---|---|---|---|
| `climate_change_hallucinated.json` | 6 chunks | 5 | Numeric and milestone hallucinations (5.7°C vs 2–4°C, 2035 vs 2050) |
| `renewable_energy_supported.json` | 6 chunks | 6 | 100% grounded assertions with explicit numerical evidence |
| `phosphorus_eutrophication.json` | 6 EEA docs | 22+ | Real-world production data captured from the Climate-ADAPT chatbot |

---

## Verification Commands

To verify everything is in order before opening pull requests:

```bash
# Verify formatting and linting
ruff check tests/ rag_facts_check/

# Run the complete test suite
pytest
```
