---
type: EntryPoint
title: Web Service
description: FastAPI endpoints for async fact-checking.
tags: [server, fastapi, api]
timestamp: '2025-01-01T00:00:00Z'
---

# Web Service

The package includes a FastAPI web service for async fact-checking from any client.

## Installation

```bash
make setup-server    # Installs fastapi, uvicorn, httpx, python-dotenv
```

## Running the Server

```bash
# Development (auto-reload)
make serve

# Production
make serve-prod
```

The server reads LLM configuration from `.env`:

```env
LLM_API_BASE=http://localhost:4002/v1
LLM_API_KEY=not-needed
LLM_MODEL=gemma
LLM_TEMPERATURE=0.1
# Reasoning models spend most of a judge call on chain-of-thought. Turn it off:
LLM_DISABLE_REASONING=1
```

`LLM_DISABLE_REASONING` accepts `1` (default strategy: `chat_template_kwargs.enable_thinking=false`,
honoured by vLLM and llama.cpp server), `reasoning_effort` (`reasoning_effort: "none"`), or `both`.
`LLM_EXTRA_BODY` still works for endpoint-specific tuning and overrides the flag on conflicting keys.

Source budgets are also read from the environment:

```env
# Total characters of source text the judge may see across all documents
CHECKER_MAX_DOCS_CHARS=100000
# Fairness cap per document — applied only when the corpus does not fit the total
CHECKER_MAX_CHARS_PER_DOC=10000
```

A document that fits inside `CHECKER_MAX_DOCS_CHARS` is passed whole. The per-document
cap only kicks in when the corpus overflows the total, so one very long source cannot
starve the others. See [Onyx Evidence Contract](../architecture/onyx-evidence-contract.md#character-budgets).

## Endpoints

### `POST /check` — Full fact-checking report

**Request:**

```json
{
  "answer": "Paris is the capital of France.",
  "documents": [
    { "doc_id": "doc_1", "title": "Paris overview", "text": "Paris is the capital..." },
    { "doc_id": "doc_2", "title": "Eiffel Tower", "text": "The Eiffel Tower..." }
  ],
  "options": {
    "num_consistency_runs": 1,
    "evidence_first": true,
    "use_evidence_retrieval": true
  }
}
```

**Response:** Full `CheckReport` with `overall_verdict`, `dimensions`, `claims` (with `span` offsets), `results` (with `evidence_spans` offsets), and `hallucination_flags`.

### `POST /halloumi/generate` — Halloumi-compatible endpoint

Drop-in replacement for the existing halloumi middleware. Accepts the same request format and returns a response compatible with the frontend's `ClaimModal`, `ClaimSegments`, and `Citation` components.

**Request:**

```json
{
  "answer": "Paris is the capital of France.",
  "sources": [
    { "text": "Paris is the capital...", "title": "Paris overview" },
    { "text": "The Eiffel Tower...", "title": "Eiffel Tower" }
  ]
}
```

Sources accept either plain strings or structured objects. Structured sources
(`HalloumiSource`) carry document metadata that improves verification accuracy:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Document text (required) |
| `title` | `str \| null` | Document title or semantic identifier |
| `source_type` | `str \| null` | Source type (e.g. `web`, `file`) |
| `link` | `str \| null` | Source URL |

Plain strings are still accepted for backward compatibility but the LLM will
not have document title context for verification.

**Response:**

```json
{
  "claims": [
    {
      "startOffset": 0,
      "endOffset": 32,
      "segmentIds": ["0"],
      "score": 1.0,
      "rationale": "Document states this explicitly."
    }
  ],
  "segments": {
    "0": { "startOffset": 0, "endOffset": 22 }
  }
}
```

Claim scores are categorical: `1.0` (supported), `0.4` (not enough info),
`0.0` (contradicted). The frontend renders these as `High`, `Low`, `Failed`.

### `GET /health` — Health check

Returns `{"status": "ok", "version": "0.2.0"}`.

## Span-Level Grounding

Both endpoints return character offsets for clickable highlighting:

- **`claims[].span`**: `{start, end}` offsets in the original answer text
- **`results[].evidence_spans[]`**: `{quote, start, end, document_index}` — one entry per evidence quote located in a source document

The client can use these to render clickable spans in the answer that link to highlighted evidence in the source documents.
