---
type: Guide
title: Project Overview
description: What RAG Facts Check is and how it works at a high level.
tags: [overview, rag, fact-checking]
timestamp: '2025-01-01T00:00:00Z'
---

# Project Overview

RAG Facts Check is a standalone verification service for EEA chatbot answers. It runs as a FastAPI server that accepts RAG-generated answers and their source documents, extracts factual claims, verifies each against the sources, and returns a detailed report with per-claim scores and evidence highlights.

## The Problem

RAG (Retrieval-Augmented Generation) systems can hallucinate — generating answers that aren't grounded in the retrieved documents. This service detects those hallucinations by systematically checking each factual claim.

## How It Works

The pipeline has four stages:

1. **Extract** — Atomic factual claims are extracted from the generated answer.
2. **Retrieve** — Relevant document chunks are retrieved for each claim (optional).
3. **Verify** — Each claim is verified against the source documents.
4. **Aggregate** — Results are combined into a confidence score, verdict, and detailed report.

## Running the Service

```bash
# Development (auto-reload, bootstraps venv)
make serve

# Production
docker build -t rag-fact-check .
docker run -p 8000:8000 rag-fact-check
```

The service exposes `POST /halloumi/generate` (halloumi-compatible) and `POST /check` (full report). See [Web Service](/guides/web-service.md) for endpoint details.

## Key Features

- **Claim-level granularity** — each factual assertion is scored independently
- **Document titles in prompts** — structured sources carry metadata for better LLM context
- **Self-consistency** — multiple verification runs with majority voting
- **Evidence-first prompting** — LLM extracts evidence before deciding
- **Span-level grounding** — character offsets for clickable highlighting in the frontend
- **Categorical scores** — High / Low / Failed per claim

See [Fact-Checking Approaches](approaches.md) for how this strategy compares to alternatives.
See [System Architecture](/architecture/architecture.md) for module-level details.
