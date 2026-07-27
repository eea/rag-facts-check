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
```

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

**Response:** Full `CheckReport` with `overall_verdict`, `dimensions`, `claims` (with `span` offsets), `results` (with `evidence_span` offsets), and `hallucination_flags`.

### `POST /halloumi/generate` — Halloumi-compatible endpoint

Drop-in replacement for the existing halloumi middleware. Accepts the same request format and returns a response compatible with the frontend's `ClaimModal`, `ClaimSegments`, and `Citation` components.

**Request:**

```json
{
  "answer": "Paris is the capital of France.",
  "sources": ["Paris is the capital...", "The Eiffel Tower..."]
}
```

**Response:**

```json
{
  "claims": [
    {
      "startOffset": 0,
      "endOffset": 32,
      "segmentIds": ["0"],
      "score": 0.95,
      "rationale": "Document states this explicitly."
    }
  ],
  "segments": {
    "0": { "startOffset": 0, "endOffset": 22 }
  }
}
```

### `GET /health` — Health check

Returns `{"status": "ok", "version": "0.2.0"}`.

## Span-Level Grounding

Both endpoints return character offsets for clickable highlighting:

- **`claims[].span`**: `{start, end}` offsets in the original answer text
- **`results[].evidence_span`**: `{start, end}` offsets in the source document text

The client can use these to render clickable spans in the answer that link to highlighted evidence in the source documents.
