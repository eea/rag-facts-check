---
type: Configuration
title: Configuration
description: Checker options and tuning parameters.
tags: [configuration, tuning]
timestamp: '2025-01-01T00:00:00Z'
---

# Configuration

## RAGFactsChecker Parameters

```python
checker = RAGFactsChecker(
    llm=llm,
    max_claims=10,                    # Limit claims for latency control
    max_new_tokens=512,               # LLM generation length (verification)
    max_extraction_tokens=2048,       # LLM generation length (extraction)
    max_docs_chars=8000,              # Truncate docs to fit context
    max_chars_per_doc=2000,           # Truncate individual documents
    num_consistency_runs=3,           # Self-consistency: run 3 times, majority vote
    evidence_first=True,              # Use evidence-first multi-step prompting
    use_evidence_retrieval=True,      # Retrieve relevant chunks per claim
    retriever=EvidenceRetriever(      # Custom retriever
        chunk_size=200,
        top_k=3,
    ),
)
```

## Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `llm` | *(required)* | LLM instance implementing `generate(prompt)` |
| `max_claims` | `None` | Maximum number of claims to extract and verify |
| `max_new_tokens` | `512` | Maximum tokens for LLM generation (verification phase) |
| `max_extraction_tokens` | `2048` | Maximum tokens for claim extraction (allows thorough decomposition) |
| `max_docs_chars` | `8000` | Total document characters before truncation |
| `max_chars_per_doc` | `2000` | Per-document character limit |
| `num_consistency_runs` | `1` | Number of self-consistency verification runs |
| `evidence_first` | `True` | Use evidence-first multi-step prompting |
| `use_evidence_retrieval` | `False` | Enable chunk-based evidence retrieval (disabled by default — verifier sees all documents to avoid false positives from missed chunks) |
| `retriever` | `EvidenceRetriever()` | Custom retriever instance |

## Evidence-First Prompting

When `evidence_first=True`, the verification prompt uses a multi-step format:

```
Step 1: Extract relevant evidence
Step 2: Compare evidence to claim
Step 3: Verdict
Step 4: Output (VERDICT/CONFIDENCE/EVIDENCE/EXPLANATION)
```

This reduces hallucinated evaluations by forcing the LLM to cite evidence before deciding.

## Self-Consistency

When `num_consistency_runs > 1`, verification runs multiple times with increasing temperatures (0.1, 0.2, 0.3, ...). The majority verdict is used, and a `consistency_score` is computed per claim.

| Runs | Behavior |
|---|---|
| 1 (default) | Single verification per claim |
| 3 | Majority vote + consistency score |
| 5+ | More robust majority vote at higher compute cost |

## Evidence Retrieval

When `use_evidence_retrieval=True`, the `EvidenceRetriever` splits documents into chunks and retrieves only the most relevant ones per claim. This reduces context window usage and improves verification accuracy.

```python
from rag_facts_check import EvidenceRetriever

retriever = EvidenceRetriever(chunk_size=200, top_k=3)
checker = RAGFactsChecker(llm, retriever=retriever, use_evidence_retrieval=True)
```
