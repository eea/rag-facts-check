---
type: DataModel
title: Output Format
description: CheckReport structure, verdicts, and dimensions.
tags: [output, schema, report]
timestamp: '2025-01-01T00:00:00Z'
---

# Output Format

The `CheckReport` is the final output of the fact-checking pipeline.

## Fields

| Field | Type | Description |
|---|---|---|
| `answer_score` | `float (0-10)` | Overall answer quality grade — see [Answer Quality Score](/architecture/answer-quality-score.md) |
| `overall_confidence` | `int (0-100)` | Weighted confidence score across all claims |
| `overall_verdict` | `str` | `fully_supported`, `mostly_supported`, `partially_supported`, `largely_unsupported`, `no_claims` |
| `dimensions` | `dict` | Multi-dimensional scores (see below) |
| `claims` | `list[Claim]` | Extracted claims with indices and `span` offsets |
| `results` | `list[VerificationResult]` | Per-claim verification results |
| `hallucination_flags` | `list` | Claims that are contradicted or lack evidence |
| `summary` | `str` | Human-readable summary |

## Dimensions

| Dimension | Description |
|---|---|
| `groundedness` | % of claims supported by documents |
| `contradiction_rate` | % of claims contradicted by documents |
| `hallucination_rate` | % of claims unsupported or contradicted |
| `completeness` | % of claims covered (same as groundedness without coverage analysis) |

## Per-Claim Verdicts

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Source documents contain clear evidence supporting the claim |
| `CONTRADICTED` | Source documents contain clear evidence contradicting the claim |
| `NOT ENOUGH INFO` | Source documents don't contain sufficient information |

## Span-Level Grounding

Both claims and results include character offsets for clickable highlighting:

- **`claims[].span`**: `{start, end}` offsets in the original answer text
- **`results[].evidence_span`**: `{start, end}` offsets in the source document text

## VerificationResult Fields

| Field | Type | Description |
|---|---|---|
| `verdict` | `str` | `SUPPORTED`, `CONTRADICTED`, `NOT ENOUGH INFO` |
| `confidence` | `int (0-100)` | Confidence in this verdict |
| `evidence` | `str` | Quoted evidence from source documents |
| `explanation` | `str` | Reasoning for the verdict |
| `document_id` | `str` | ID of the source document (when using evidence retrieval) |
| `chunk_id` | `str` | ID of the retrieved chunk (when using evidence retrieval) |
| `consistency_score` | `float` | Agreement across self-consistency runs (when `num_consistency_runs > 1`) |
| `evidence_span` | `Span` | Character offsets of the evidence in the source document |
