---
type: Reference
title: Fact-Checking Approaches
description: Survey of possible RAG fact-checking strategies and where this project fits.
tags: [approaches, survey, rag, verification]
timestamp: '2025-01-01T00:00:00Z'
---

# Fact-Checking Approaches

There are many possible strategies for verifying that a RAG-generated answer is grounded in its source documents. The table below surveys **possible approaches** — not all of them are implemented here.

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Single-Prompt Verification** | Feed answer + docs to LLM, ask "is this supported?" | Simple, fast | No per-claim breakdown |
| **Claim Extraction + Verification** | Extract claims, verify each against docs | Granular, cites evidence, per-claim scores | More compute |
| **NLI-based** | Use Natural Language Inference models | Fast, model-based | Requires specialized NLI model |
| **Two-Agent Verification** | Separate LLM verifies the answer | Independent check | More compute |
| **Reverse QA** | Re-answer the question from docs, compare | Catches hallucinations | Indirect comparison |
| **Evidence-First Prompting** | LLM extracts evidence before deciding | Reduces hallucinated evaluations | More prompt tokens |
| **Self-Consistency** | Run verification N times, majority vote | More robust | More compute |
| **Span-Level Verification** | Cite specific document/paragraph IDs | Precise, enterprise-ready | Requires structured docs |

## What This Project Implements

This project's core strategy is **Claim Extraction + Verification**. On top of that core, it optionally layers several complementary techniques from the table:

- **Evidence Retrieval** — chunk-based lexical retrieval to reduce context window usage
- **Evidence-First Prompting** — multi-step prompts that extract evidence before deciding
- **Self-Consistency** — configurable number of verification runs with majority voting
- **Multi-Dimensional Scoring** — groundedness, contradiction rate, hallucination rate, completeness

## What This Project Does Not Implement

The following approaches from the table are **not implemented** but remain viable alternatives:

- **Single-Prompt Verification** — deliberately avoided in favor of claim-level granularity
- **NLI-based** — would require integrating a specialized NLI model
- **Two-Agent Verification** — possible future direction for independent verification
- **Reverse QA** — would require re-generating answers and comparing outputs
- **Span-Level Verification** — partially supported via character offsets; full paragraph-level citation would require structured documents
