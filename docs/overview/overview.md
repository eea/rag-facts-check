---
type: Guide
title: Project Overview
description: What RAG Facts Check is and how it works at a high level.
tags: [overview, rag, fact-checking]
timestamp: '2025-01-01T00:00:00Z'
---

# Project Overview

RAG Facts Check is a modular system for verifying RAG-generated answers against their source documents using claim extraction and per-claim verification.

## The Problem

RAG (Retrieval-Augmented Generation) systems can hallucinate — generating answers that aren't grounded in the retrieved documents. This system detects those hallucinations by systematically checking each factual claim.

## How It Works

The pipeline has four stages:

1. **Extract** — Atomic factual claims are extracted from the generated answer.
2. **Retrieve** — Relevant document chunks are retrieved for each claim (optional).
3. **Verify** — Each claim is verified against the source documents.
4. **Aggregate** — Results are combined into a confidence score, verdict, and detailed report.

## Quick Start

```python
from rag_facts_check import RAGFactsChecker, MockLLM

llm = MockLLM()
checker = RAGFactsChecker(llm)

answer = "Paris is the capital of France. The Eiffel Tower was built in 1889."
documents = [
    "Paris is the capital of France. It is known for the Eiffel Tower.",
    "The Eiffel Tower was constructed between 1887 and 1889.",
]

report = checker.check(answer, documents)
print(report.to_dict())
```

## Key Features

- **Claim-level granularity** — each factual assertion is scored independently
- **Evidence retrieval** — chunk-based lexical matching to find relevant context
- **Self-consistency** — multiple verification runs with majority voting
- **Evidence-first prompting** — LLM extracts evidence before deciding
- **Span-level grounding** — character offsets for clickable highlighting
- **FastAPI web service** — async fact-checking from any client

See [Fact-Checking Approaches](approaches.md) for how this strategy compares to alternatives.
See [System Architecture](/architecture/architecture.md) for module-level details.
