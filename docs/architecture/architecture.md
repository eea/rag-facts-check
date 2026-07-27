---
type: Architecture
title: System Architecture
description: Module layout and component responsibilities.
tags: [architecture, modules]
timestamp: '2025-01-01T00:00:00Z'
---

# System Architecture

## Module Layout

```
rag_facts_check/
├── __init__.py       # Package exports
├── models.py         # Data classes: Claim, VerificationResult, CheckReport, Span
├── llm.py            # LLM interface + adapters (HF, API, Chat, AsyncAPI)
├── prompts.py        # Prompt templates for extraction & verification
├── retriever.py      # Evidence retrieval (chunk-based lexical matching)
├── spans.py          # Span matching (claim → answer, evidence → document)
├── checker.py        # Core pipeline: Extractor → Verifier → Aggregator
└── server.py         # FastAPI web service (POST /check, POST /halloumi/generate)
```

```
mock_datasets/       # Synthetic test datasets (JSON)
tests/               # Pytest test suite
├── conftest.py       # Shared fixtures
├── test_models.py    # Data model tests
├── test_retriever.py # Evidence retrieval tests
├── test_checker.py   # Core pipeline tests
└── test_integration.py # End-to-end tests
```

## Components

### `checker.py` — Core Pipeline

Orchestrates the full fact-checking flow:

- **`ClaimExtractor`** — sends the answer to the LLM and parses returned claims
- **`ClaimVerifier`** — verifies each claim against documents (with optional self-consistency and evidence-first prompting)
- **`RAGFactsChecker._aggregate`** — combines per-claim results into a `CheckReport`

### `models.py` — Data Classes

- **`Claim`** — a single extracted factual assertion with an index and span
- **`VerificationResult`** — per-claim verdict, confidence, evidence, explanation
- **`CheckReport`** — the final aggregated report with dimensions and hallucination flags
- **`Span`** — character-level `{start, end}` offsets

### `llm.py` — LLM Interface

Abstract `LLM` base class with a single `generate(prompt) -> str` method. Built-in adapters:

- `HuggingFaceLLM` — local transformers models
- `APILLM` — HTTP completion endpoints (vLLM, Ollama, llama.cpp)
- `ChatLLM` — chat-completion format APIs
- `AsyncAPILLM` — async HTTP API client

### `retriever.py` — Evidence Retrieval

Splits documents into chunks and retrieves the most relevant ones per claim using lexical (keyword overlap) matching. Configurable `chunk_size` and `top_k`.

### `spans.py` — Span Matching

Maps extracted claims back to character offsets in the original answer, and evidence quotes back to offsets in source documents. Enables clickable highlighting in UI clients.

### `prompts.py` — Prompt Templates

System and user prompts for claim extraction and claim verification phases, including evidence-first multi-step variants.

### `server.py` — FastAPI Web Service

HTTP endpoints for async fact-checking. See [Web Service](/guides/web-service.md).
