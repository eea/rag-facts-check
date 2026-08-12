# Plan: Batch Verification

**Date:** 2025-08-12
**Status:** Draft — awaiting review
**Owner:** Tiberiu Ichim

---

## Context

Our pipeline runs ~30 sequential LLM calls for verification (one per claim), each sending ~100KB of documents. KV cache reuse on llama.cpp means the document prefix is cached after the first call, so each subsequent call computes ~50 new tokens. Wall time is ~110s for 30 claims (~4s/call).

**Batch verification** groups N claims into a single LLM call. Documents are sent once, all claims are listed after, and the LLM returns a JSON array of verdicts. This reduces HTTP overhead and allows parallel token decoding.

## Current Architecture

```
RAGFactsChecker.check()
  → ClaimExtractor.extract(answer)        # 1 LLM call, small context
  → for claim in claims:                  # N sequential iterations
      → EvidenceRetriever.retrieve()      # optional, BM25 or LLM-based
      → ClaimVerifier.verify(claim, docs) # 1 LLM call per claim, ~100KB context
  → _aggregate(results)                   # pure Python
```

**Key files:**
- `rag_facts_check/checker.py` — `ClaimExtractor`, `ClaimVerifier`, `RAGFactsChecker`
- `rag_facts_check/prompts.py` — prompt templates and formatters
- `rag_facts_check/agents.py` — atomic-agents schemas (`VerificationInput`, etc.)
- `prompts/claim-verification-*.txt` — prompt text files
- `scripts/check.py` — CLI tool for ad-hoc runs

**Prompt ordering (documents-first for KV cache):**
```
{system_prompt}
Source Documents:
{documents}          ← static, ~100KB, cached after first call

Claim:
{claim}              ← varies per call, ~30 tokens

Return JSON: ...
```

**Agent path:** `VerificationInput` has `documents` before `claim` in Pydantic field order so `model_dump_json()` serializes docs first.

## Proposed Changes

### 1. New prompt templates (done)

- `prompts/claim-verification-batch-system.txt` — system prompt for batch mode
- `prompts/claim-verification-batch-prompt.txt` — user prompt with `{documents}` then `{claims}`

Batch prompt structure:
```
{system_prompt}

Source Documents:
{documents}          ← static, ~100KB

Claims to verify:
  Claim 1: ...
  Claim 2: ...
  ...
  Claim N: ...

Return a JSON array with one object per claim.
```

### 2. `prompts.py` — new formatter (done)

```python
def format_claim_verification_batch_prompt(
    claims: list[tuple[int, str]],
    documents: list[str] | list[dict[str, str]],
) -> str:
```

### 3. `ClaimVerifier` — batch mode

Add `batch_size` parameter (default `1` = current behavior):

```python
class ClaimVerifier:
    def __init__(
        self,
        llm: LLM,
        batch_size: int = 1,          # NEW: 1 = sequential, >1 = batched
        ...
    ):
```

When `batch_size > 1`, the `verify_batch()` method:
1. Groups claims into batches of `batch_size`
2. For each batch, builds a single prompt with all claims + documents
3. Sends one LLM call per batch
4. Parses the JSON array response into individual `VerificationResult`s

New methods:
- `async def verify_batch(claims, documents) -> list[VerificationResult]` — public API
- `async def _batch_verify(self, claims, documents) -> list[VerificationResult]` — single batch call
- `def _parse_batch_result(self, response: str) -> dict[int, VerificationResult]` — parse JSON array

Existing `verify(claim, documents)` remains unchanged for single-claim use.

### 4. `RAGFactsChecker` — wire through

Add `batch_size` parameter:

```python
class RAGFactsChecker:
    def __init__(
        self,
        llm: LLM,
        batch_size: int = 1,          # NEW
        ...
    ):
```

In `check()`, after claim extraction:
- If `batch_size > 1`: call `verifier.verify_batch(claims, docs)` instead of the per-claim loop
- If `batch_size == 1`: existing per-claim loop (unchanged)

Evidence retrieval still runs per-claim (BM25 is fast, no LLM). The retrieved chunks are merged per batch.

### 5. Agent path

The atomic-agents `VerificationInput` schema is single-claim. For batch mode, we skip the agent path and use the raw prompt + LLM. This is fine because:
- Batch mode is a performance optimization, not a quality improvement
- The agent path adds retry/structured-output overhead that doesn't scale to N claims
- JSON array parsing is simple and reliable

### 6. `scripts/check.py` — CLI flag

```bash
python scripts/check.py --batch-size 10 data/generate-request.json
```

### 7. Tests

- `test_checker.py`: add `TestClaimVerifier.test_verify_batch` and `test_parse_batch_result`
- `test_integration.py`: add E2E test with `batch_size=5`
- Mock LLM fixture: add batch response handling to `mock_llm`

## Trade-offs

| | Sequential (batch_size=1) | Batched (batch_size=N) |
|---|---|---|
| LLM calls | N | ceil(N/batch_size) |
| Context per call | ~100KB + 1 claim | ~100KB + N claims |
| KV cache hit | ~99.8% | ~99% (slightly less, claims vary) |
| Response parsing | 1 verdict per call | N verdicts per call |
| Failure mode | 1 bad claim = 1 bad result | 1 bad batch = N bad results |
| Agent path | Supported | Skipped (raw prompt) |

**Risk:** If the LLM fails to return valid JSON for a batch, all claims in that batch fail. Mitigation: retry at the batch level, or fall back to sequential for failed batches.

**Risk:** Large batches may exceed the model's output token limit. Mitigation: cap batch_size (default 10) and validate response has all claim indices.

## Open Questions

1. **What's a good default batch_size?** 5-10 seems reasonable. Too small = no benefit, too large = output overflow risk.
2. **Should batch mode be opt-in or auto-detected?** Opt-in via `batch_size=N` is safest. Could auto-enable when `len(claims) > 5` but that's surprising.
3. **Evidence retrieval per claim or per batch?** Per-claim retrieval (BM25) is cheap and gives better context narrowing. Keep per-claim.
4. **Self-consistency + batching?** These are orthogonal. Self-consistency runs N times with different temps. Batching groups claims. Could combine but adds complexity. Keep separate for now.
