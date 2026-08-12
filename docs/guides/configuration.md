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
    max_docs_chars=100000,            # Truncate docs to fit context
    max_chars_per_doc=10000,          # Truncate individual documents
    num_consistency_runs=3,           # Self-consistency: run 3 times, majority vote
    evidence_first=True,              # Use evidence-first multi-step prompting
    use_evidence_retrieval=True,      # Retrieve relevant chunks per claim (default: True)
    batch_size=1,                     # Claims per LLM call (1=sequential, >1=batched)
    retriever=LLMEvidenceRetriever(   # LLM-based retrieval (default when enabled)
        llm=llm,
        chunk_size=1000,
        top_k=5,
    ),
)
```

## Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `llm` | *(required)* | LLM instance implementing `async generate(prompt)` |
| `max_claims` | `None` | Maximum number of claims to extract and verify |
| `max_new_tokens` | `512` | Maximum tokens for LLM generation (verification phase) |
| `max_extraction_tokens` | `2048` | Maximum tokens for claim extraction (allows thorough decomposition) |
| `max_docs_chars` | `100000` | Total document characters before truncation |
| `max_chars_per_doc` | `10000` | Per-document character limit |
| `num_consistency_runs` | `1` | Number of self-consistency verification runs |
| `evidence_first` | `True` | Use evidence-first multi-step prompting |
| `use_evidence_retrieval` | `True` | Enable evidence retrieval (LLM-based by default) |
| `retriever` | `LLMEvidenceRetriever(llm)` | Custom retriever instance (LLM-based when enabled, keyword-based when disabled) |
| `batch_size` | `1` | Number of claims to verify in a single LLM call. `1` = sequential (one claim per call). `>1` = batched (multiple claims per call, fewer LLM calls but no per-claim chunk narrowing) |

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

When `use_evidence_retrieval=True` (the default), documents are split into chunks and only the most relevant chunks are passed to each claim verification. This reduces context window usage and improves verification accuracy.

Two retrieval strategies are available:

### LLM-Based Retrieval (default)

`LLMEvidenceRetriever` uses the LLM to judge semantic relevance. It understands paraphrases, synonyms, and domain terminology. Uses larger chunks (default 1000 words) so the LLM has enough context.

```python
from rag_facts_check import LLMEvidenceRetriever

retriever = LLMEvidenceRetriever(llm=llm, chunk_size=1000, top_k=5)
checker = RAGFactsChecker(llm, retriever=retriever)
```

### Keyword-Based Retrieval

`EvidenceRetriever` uses lexical matching (keyword overlap / Jaccard similarity). Fast — no extra LLM calls — but less accurate for paraphrased content.

```python
from rag_facts_check import EvidenceRetriever

retriever = EvidenceRetriever(chunk_size=200, top_k=3)
checker = RAGFactsChecker(llm, retriever=retriever)
```

## Batch Verification

When `batch_size > 1`, multiple claims are verified in a single LLM call. Documents are sent once per batch (benefiting from KV cache reuse), and all claims in the batch are verified together. This reduces the number of LLM calls from N to ceil(N/batch_size).

```python
checker = RAGFactsChecker(llm, batch_size=10)
```

**Trade-offs:**

| | Sequential (`batch_size=1`) | Batched (`batch_size=N`) |
|---|---|---|
| LLM calls | N (one per claim) | ceil(N/batch_size) |
| Context per call | docs + 1 claim | docs + N claims |
| Per-claim retrieval | Yes (chunks narrowed per claim) | No (full documents sent) |
| Failure mode | 1 bad claim = 1 bad result | 1 bad batch = N bad results |

Batch mode skips evidence retrieval — all documents are sent directly to the LLM. Use `batch_size=1` (default) when you need per-claim chunk narrowing.
