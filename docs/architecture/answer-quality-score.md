---
type: DataModel
title: Answer Quality Score
description: 0-10 numeric grade for overall RAG answer quality.
tags: [scoring, quality, output]
timestamp: '2026-07-28T00:00:00Z'
---

# Answer Quality Score

A single 0-10 numeric grade summarising how well a RAG-generated answer is
grounded in its source documents. Computed deterministically from per-claim
verification results — no extra LLM call required.

## Scale

| Score | Label | Meaning |
|---|---|---|
| 9–10 | Excellent | Fully grounded, all claims supported with evidence |
| 7–8 | Good | Mostly grounded, minor gaps or low-confidence claims |
| 5–6 | Acceptable | Partially grounded — some claims lack evidence |
| 3–4 | Poor | Significant hallucination or uncited content |
| 1–2 | Failing | Mostly contradicted or fabricated |
| 0 | No claims | Answer contains no verifiable claims |

The label is derived from the score via `score_label(score)`:

```python
def score_label(score: float) -> str:
    if score >= 9:
        return "Excellent"
    elif score >= 7:
        return "Good"
    elif score >= 5:
        return "Acceptable"
    elif score >= 3:
        return "Poor"
    elif score > 0:
        return "Failing"
    else:
        return "No claims"
```

## Formula

The score composes three factors multiplicatively:

```
raw = groundedness_base × citation_penalty × contradiction_penalty
score = round(clamp(raw, 0, 10), 1)
```

### 1. Groundedness base (0-10)

Weighted average of per-claim verdicts, scaled to 0-10:

```
groundedness = Σ(verdict_weight × confidence) / Σ(confidence) × 10
```

Verdict weights:

| Verdict | Weight |
|---|---|
| `supported` | 1.0 |
| `not_enough_info` | 0.4 |
| `contradicted` | 0.0 |

If all claims are supported at high confidence, this approaches 10.
Contradicted claims contribute zero, dragging the average down.

### 2. Citation penalty

Claims whose evidence quote was not matched to a source document
(empty `segmentIds`) are flagged as "uncited". Even when the LLM
verdicts a claim as SUPPORTED, absence of a matched evidence span
suggests the claim may rely on model knowledge rather than sources.

```
citation_ratio = claims_with_segments / total_claims
citation_penalty = 1.0 - (0.3 × (1 - citation_ratio))
```

- All cited → penalty = 1.0 (no reduction)
- None cited → penalty = 0.7 (30% reduction)
- Partially cited → proportional reduction

### 3. Contradiction penalty

Contradicted claims are severe — the answer asserts something the
sources explicitly disagree with. This penalty drives the score
toward zero quickly.

```
contradiction_ratio = contradicted_claims / total_claims
contradiction_penalty = max(0, 1.0 - (1.5 × contradiction_ratio))
```

- 0 contradictions → penalty = 1.0
- 10% contradicted → penalty = 0.85
- 33%+ contradicted → penalty = 0 (score floors to 0)

## Worked examples

### All supported, all cited

5 claims, all supported at 90-100% confidence, all have evidence segments.

- groundedness = 9.5
- citation_penalty = 1.0
- contradiction_penalty = 1.0
- **score = 9.5** (Excellent)

### Mixed — some uncited model knowledge

10 claims: 7 supported (90% confidence), 3 not_enough_info (60%).
7 have segments, 3 do not (citation_ratio = 0.7).

- groundedness = (7×0.9×1.0 + 3×0.6×0.4) / (7×0.9 + 3×0.6) × 10 = 7.17
- citation_penalty = 1.0 - 0.3 × 0.3 = 0.91
- contradiction_penalty = 1.0
- **score = 7.17 × 0.91 = 6.5** (Acceptable)

### Hallucination — contradicted claims

6 claims: 3 supported (90%), 2 contradicted (80%), 1 not_enough_info (60%).
3 have segments (citation_ratio = 0.5).

- groundedness = (3×0.9×1.0 + 2×0.8×0.0 + 1×0.6×0.4) / (3×0.9 + 2×0.8 + 1×0.6) × 10 = 3.82
- citation_penalty = 1.0 - 0.3 × 0.5 = 0.85
- contradiction_penalty = 1.0 - 1.5 × (2/6) = 0.50
- **score = 3.82 × 0.85 × 0.50 = 1.6** (Failing)

## Tuning knobs

| Parameter | Default | Effect |
|---|---|---|
| `not_enough_info` weight | 0.4 | Higher = more forgiving of missing evidence |
| Citation max reduction | 0.3 (30%) | Higher = harsher penalty for uncited claims |
| Contradiction multiplier | 1.5 | Higher = contradictions drive score to zero faster |

These are currently hardcoded in `checker.py`. To adjust them, modify
the constants in `RAGFactsChecker._compute_answer_score()`.

## Output locations

The score appears in two places:

- **`CheckReport.answer_score`** — float 0-10 on the full report
- **Halloumi response** — top-level `answer_score` field alongside `claims` and `segments`

## Design rationale

### Why multiplicative, not additive?

Multiplicative composition ensures that a failure in any dimension
(citations, contradictions) meaningfully reduces the final score.
An additive formula would allow a high groundedness to mask a total
lack of citations.

### Why penalise uncited claims?

The LLM verifier can mark a claim as SUPPORTED based on its internal
knowledge even when the source documents do not contain the evidence.
The citation penalty surfaces this gap: a claim is only truly grounded
if its evidence quote can be located in the provided sources.

### Why is `not_enough_info` weighted at 0.4?

A claim that cannot be verified is not necessarily wrong — the sources
may simply be incomplete. A weight of 0 reflects "known to be false"
(contradicted), while 1.0 reflects "known to be true" (supported).
0.4 sits between these, reflecting uncertainty without full penalty.
