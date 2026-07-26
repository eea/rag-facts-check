---
name: rag-facts-check
description: >
  Verify RAG-generated answers against source documents for factual accuracy,
  hallucinations, and grounding. Extracts atomic claims from the answer,
  retrieves relevant document evidence, verifies each claim, and produces
  a quality report with confidence scores, evidence citations, and
  hallucination flags. Use when asked to check RAG answer quality,
  verify a generated answer, or audit an LLM-generated response for
  hallucinations and factual errors.
---

# RAG Facts Check

Manually review a RAG-generated answer for hallucinations, factual accuracy,
and grounding against source documents. Uses a claim-extraction + per-claim
verification pipeline with evidence retrieval, evidence-first prompting,
self-consistency, and multi-dimensional scoring.

## Prerequisites

- A RAG-generated answer to verify
- Source documents (retrieved by the RAG system) as a list of strings
- A local LLM accessible via the `LLM` interface (HuggingFaceLLM, APILLM, or custom)
- Python 3.10+ with `rag_facts_check` installed or on the Python path
- Virtual environment activated (if applicable)

## Quick Start

Run the consolidated check script:

```bash
python .agents/skills/rag-facts-check/scripts/run_check.py \
    --answer "Paris is the capital of France." \
    --documents "Paris is the capital of France. It is known for the Eiffel Tower." \
    --output report.json
```

This single command produces:
1. Extracted claims from the answer
2. Per-claim verification results (verdict, confidence, evidence, explanation)
3. Overall confidence score and verdict
4. Multi-dimensional scores (groundedness, contradiction_rate, hallucination_rate)
5. Hallucination flags (contradicted or unsupported claims)
6. Span-level citations (document_id, chunk_id) when evidence retrieval is enabled

Save this output — you will reference it throughout the review.

## Workflow

### Step 1: Prepare inputs

Gather:
- The RAG-generated answer text
- The source documents that were retrieved for the answer (as a list of strings)
- Your local LLM backend (implement the `LLM` interface)

### Step 2: Run the check

```bash
python .agents/skills/rag-facts-check/scripts/run_check.py \
    --answer-file <answer-file> \
    --documents-file <documents-file> \
    --output <report-file> \
    --num-consistency-runs 3 \
    --evidence-first \
    --use-evidence-retrieval
```

Or use the Python API directly:

```python
from rag_facts_check import RAGFactsChecker, HuggingFaceLLM

llm = HuggingFaceLLM(model, tokenizer, chat_format=True)
checker = RAGFactsChecker(
    llm,
    num_consistency_runs=3,
    evidence_first=True,
    use_evidence_retrieval=True,
)

report = checker.check(answer, documents)
print(report.to_dict())
```

### Step 3: Review the results

Examine the report for:

1. **Overall verdict** — `fully_supported`, `mostly_supported`, `partially_supported`, `largely_unsupported`, or `no_claims`
2. **Overall confidence** — weighted score 0–100
3. **Multi-dimensional scores** — groundedness, contradiction_rate, hallucination_rate, completeness
4. **Per-claim results** — each claim's verdict, confidence, evidence, and explanation
5. **Hallucination flags** — claims that are contradicted or lack evidence

### Step 4: Cross-reference every claim

Systematically verify each claim using the report:

1. **Supported claims** — verify the evidence quote actually appears in the source documents
2. **Contradicted claims** — verify the evidence actually contradicts the claim (not just unrelated)
3. **Not enough info claims** — verify the documents genuinely lack information about the claim
4. **Evidence quality** — check that evidence quotes are exact and relevant

### Step 5: Check for common hallucination patterns

- **Phantom facts** — claims about entities, dates, or events not present in any source document
- **Numeric hallucinations** — numbers that don't match any source data
- **Cross-document conflation** — facts from one document attributed to another
- **Temporal hallucinations** — dates or timeframes not supported by the documents
- **Causal hallucinations** — cause-effect relationships not stated in the documents
- **Invented entities** — names, organizations, or concepts not found in any source
- **Overgeneralization** — broad claims that aren't supported by specific evidence
- **Negative claims** — statements that something "did not happen" or "is not the case" when the documents are silent

### Step 6: Write the quality report

Write a quality report to the run folder with the following structure:

```markdown
# RAG Facts Check Report — <Answer ID>

**Date:** ...
**Model:** ...
**Documents:** ...

## Executive Assessment
Overall quality judgment and key findings.

## 1. Claim-by-Claim Verification
Table of every claim vs verification result.

| # | Claim | Verdict | Confidence | Evidence | Status |
|---|-------|---------|------------|----------|--------|
| 1 | ... | supported | 95 | "..." | ✅ |
| 2 | ... | contradicted | 85 | "..." | ❌ |
| 3 | ... | not_enough_info | 60 | N/A | ⚠️ |

## 2. Hallucination Analysis
Detailed analysis of contradicted and unsupported claims.

## 3. Evidence Quality Assessment
Review of evidence citations — are they exact quotes? Relevant? Sufficient?

## 4. Multi-Dimensional Scores
- Groundedness: X%
- Contradiction rate: X%
- Hallucination rate: X%
- Completeness: X%

## 5. Confidence Calibration
Review of self-consistency scores — are confidences well-calibrated?

## 6. Summary of Issues
Numbered list of all issues found with severity.

## 7. Recommendations
Actionable improvements.
```

## Important Notes

- **Claim extraction is LLM-based.** The extractor splits the answer into atomic claims. Compound sentences may be split into multiple claims. Verify that each claim is truly atomic.

- **Evidence retrieval reduces context.** When `use_evidence_retrieval=True`, only the top-k most relevant document chunks are passed to the verifier per claim. This improves efficiency but may miss evidence if the retriever's lexical matching is insufficient. For better retrieval, replace `EvidenceRetriever` with an embedding-based retriever.

- **Self-consistency improves robustness.** When `num_consistency_runs > 1`, verification runs multiple times with different temperatures. The majority verdict is used, and the consistency score indicates agreement. Low consistency scores suggest the claim is ambiguous or the model is uncertain.

- **Evidence-first prompting reduces hallucinated evaluations.** The evidence-first prompt explicitly asks the verifier to extract evidence before deciding, reducing cases where the verifier makes up evidence.

- **Local models may need prompt tuning.** The default prompts are designed for instruction-tuned models. If your model produces inconsistent output formats, adjust the prompts in `prompts.py`.

- **Do not trust the LLM's confidence blindly.** Confidence scores are model-generated and may be overconfident. Cross-reference with self-consistency scores and evidence quality.

## Scripts

All scripts live in `.agents/skills/rag-facts-check/scripts/`.
They are designed to be run from the project root with the venv activated.

| Script | Purpose |
|--------|---------|
| `.agents/skills/rag-facts-check/scripts/run_check.py` | **Primary tool** — runs the full fact-checking pipeline and outputs a JSON report |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_claims` | `None` | Maximum number of claims to verify (limits latency) |
| `max_new_tokens` | `512` | Max tokens for LLM generation |
| `max_docs_chars` | `8000` | Maximum total characters of documents to include |
| `max_chars_per_doc` | `2000` | Maximum characters per individual document |
| `num_consistency_runs` | `1` | Number of verification runs for self-consistency |
| `evidence_first` | `True` | Use evidence-first multi-step prompting |
| `use_evidence_retrieval` | `True` | Retrieve relevant document chunks per claim |
| `retriever` | `EvidenceRetriever(top_k=3)` | Custom evidence retriever |

## RAG Modules You May Need

If the review uncovers issues with the fact-checking pipeline itself:

| Area | File | What it does |
|------|------|-------------|
| Claim extraction | `rag_facts_check/checker.py` | `ClaimExtractor` — splits answer into atomic claims |
| Claim verification | `rag_facts_check/checker.py` | `ClaimVerifier` — verifies each claim against documents |
| Evidence retrieval | `rag_facts_check/retriever.py` | `EvidenceRetriever` — retrieves relevant chunks per claim |
| Prompt templates | `rag_facts_check/prompts.py` | Prompt templates for extraction and verification |
| LLM interface | `rag_facts_check/llm.py` | Abstract `LLM` class + adapters (HF, API, Chat, Mock) |
| Data models | `rag_facts_check/models.py` | `Claim`, `VerificationResult`, `CheckReport` dataclasses |
