# Debug Command: `debug-check`

**Phase:** Full fact-checking pipeline (extraction → retrieval → verification → aggregation)  
**LLM calls:** 1 per claim (or 1 × `num_consistency_runs` per claim)  
**Purpose:** Run the complete RAG fact-checking pipeline for a single answer with debug checks and per-claim diagnostics

---

## When to Use

- Verify the full extraction → verification → aggregation chain end-to-end
- Check that all claims are properly extracted and verified
- Detect claims with low confidence or low consistency
- Debug evidence retrieval quality (are the right chunks being retrieved?)
- Compare verification results across model versions
- Inspect LLM prompts for each phase

---

## Command

```bash
# Run full pipeline with all debug checks
python .agents/skills/rag-facts-check/scripts/run_check.py \
    --answer "Paris is the capital of France. The Eiffel Tower was built in 1889." \
    --documents "Paris is the capital of France. It is known for the Eiffel Tower." \
    --documents "The Eiffel Tower was constructed between 1887 and 1889." \
    --num-consistency-runs 3 \
    --evidence-first \
    --use-evidence-retrieval \
    --mock \
    --verbose

# Run from a cached answer file
python .agents/skills/rag-facts-check/scripts/run_check.py \
    --answer-file output/answers/test-001/answer.txt \
    --documents-file output/answers/test-001/documents.json \
    --output output/reports/test-001.json \
    --num-consistency-runs 3 \
    --evidence-first \
    --verbose
```

---

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--answer TEXT` | Yes (or `--answer-file`) | The RAG-generated answer to verify (inline string) |
| `--answer-file PATH` | Yes (or `--answer`) | Path to a file containing the answer text |
| `--documents TEXT` | Yes (or `--documents-file`) | Source document text (inline string, single document) |
| `--documents-file PATH` | Yes (or `--documents`) | Path to a JSON file containing documents as a list of strings |
| `--output PATH` | No | Output file path for the JSON report (default: `report.json`) |
| `--num-consistency-runs N` | No | Number of verification runs for self-consistency (default: 1) |
| `--evidence-first` | No | Use evidence-first multi-step prompting (default: off) |
| `--no-evidence-retrieval` | No | Disable evidence retrieval — pass all documents to verifier |
| `--max-claims N` | No | Maximum number of claims to verify (limits latency) |
| `--max-new-tokens N` | No | Maximum tokens for LLM generation (default: 512) |
| `--mock` | No | Use MockLLM for testing (no real LLM required) |
| `--verbose`, `-v` | No | Enable verbose output |

---

## Output Example

```
Answer: Paris is the capital of France. The Eiffel Tower was built in 1889...
Documents: 3 document(s)
Consistency runs: 3
Evidence-first: True
Evidence retrieval: True

Running fact-checking pipeline...

Report written to: report.json
Overall confidence: 92.5%
Overall verdict: partially_supported
Claims: 3
Results: 3
Hallucination flags: 1

Dimensions:
  groundedness: 66.7
  contradiction_rate: 33.3
  hallucination_rate: 33.3
  completeness: 66.7
```

---

## How to Interpret Results

### Green flags ✅
- **All checks PASS**: All claims are supported with high confidence and consistency
- **Overall verdict `fully_supported`**: Every claim is supported by the source documents
- **High consistency scores**: Self-consistency scores ≥ 0.8 indicate the verifier is confident
- **Evidence present**: Every supported claim has a quoted evidence snippet
- **No hallucination flags**: No claims are contradicted or lack evidence

### Red flags ❌
- **Overall verdict `largely_unsupported`**: Most claims lack evidence — the answer may be largely hallucinated
- **Low consistency scores** (< 0.5): The verifier is inconsistent — claims may be ambiguous or the model is uncertain
- **Contradicted claims**: Claims that directly conflict with source documents
- **Not enough info claims**: Claims the documents don't address — the answer may be speculating
- **Missing evidence**: Supported claims without evidence quotes — the verifier may be making claims without citing sources
- **Low confidence scores** (< 50): The verifier is uncertain about the claim

### Yellow flags ⚠️
- **Overall verdict `partially_supported`**: Some claims are supported, others are not — the answer is partially grounded
- **Overall verdict `mostly_supported`**: Most claims are supported but some lack evidence
- **Consistency scores 0.5–0.8**: Moderate agreement — claims may be borderline

---

## Debug Checks

The following checks are available (controlled by flags):

| Check | Flag | What it verifies |
|-------|------|-----------------|
| Claim extraction | (always) | Claims are properly extracted from the answer |
| Evidence retrieval | `--evidence-first` | Relevant document chunks are retrieved per claim |
| Verification | (always) | Each claim is verified with verdict, confidence, evidence |
| Self-consistency | `--num-consistency-runs N` | Multiple runs produce consistent results |
| Multi-dimensional scoring | (always) | Groundedness, contradiction rate, hallucination rate computed |
| Hallucination detection | (always) | Contradicted and unsupported claims are flagged |

---

## Troubleshooting

### "No claims extracted"

```
Claims: 0
Overall verdict: no_claims
```

**Possible causes:**
- The answer contains no factual claims (e.g., it's a question, greeting, or apology)
- The claim extractor failed to parse the LLM output
- The answer is too short or vague

**Debug:** Run with `--verbose` to see the raw LLM response. Check if the answer contains factual statements. If using a real LLM, verify the prompt format matches your model's expected input.

### "All claims are 'not_enough_info'"

```
Overall verdict: largely_unsupported
Hallucination flags: N
```

**Possible causes:**
- The source documents don't contain information about the claims
- Evidence retrieval is filtering out relevant documents
- The documents are too short or truncated

**Debug:** Run with `--no-evidence-retrieval` to pass all documents. Check `max_docs_chars` and `max_chars_per_doc` settings. If using evidence retrieval, inspect which chunks were retrieved.

### "Low consistency scores"

```
Consistency: 33%
```

**Possible causes:**
- The claim is ambiguous or borderline
- The model is uncertain about the evidence
- Temperature is too high (try lowering)

**Debug:** Run with `--num-consistency-runs 5` for more samples. Inspect the per-run results in the JSON output. Consider adjusting the `temperature` parameter.

### "Contradicted claims that seem correct"

```
Claim: "Revenue increased 20%."
Verdict: CONTRADICTED
Evidence: "Revenue increased 15%."
```

**Possible causes:**
- The claim is actually wrong (the answer hallucinated a different number)
- The evidence quote doesn't fully address the claim
- The model misread the evidence

**Debug:** Manually verify the evidence quote against the source documents. Check if the claim is genuinely contradicted.

### "Evidence quotes don't match source documents"

**Possible causes:**
- The model is paraphrasing instead of quoting
- The evidence was retrieved from the wrong chunk
- The documents were truncated

**Debug:** Run with `--verbose` and inspect the evidence field. Manually search for the evidence text in the source documents.

---

## `--num-consistency-runs` Explained

The `--num-consistency-runs` option controls self-consistency:

| Runs | Phases executed | Output |
|------|----------------|--------|
| 1 (default) | Extract → Verify → Aggregate | Single verification per claim |
| 3 | Extract → Verify ×3 → Aggregate | Majority vote + consistency score |
| 5 | Extract → Verify ×5 → Aggregate | More robust majority vote |

With `num_consistency_runs > 1`, the verifier runs multiple times with increasing temperatures (0.1, 0.2, 0.3, ...). The majority verdict is used, and the consistency score indicates agreement.

---

## Known Limitations

- **Requires answer and documents**: Must provide both the RAG answer and the source documents
- **MockLLM is deterministic**: With `--mock`, all consistency runs return the same result. Use a real LLM for meaningful self-consistency
- **Lexical retrieval only**: The default `EvidenceRetriever` uses keyword overlap. For better retrieval, implement an embedding-based retriever
- **Single answer only**: This command processes one answer at a time — not a batch evaluator
- **No `--prompt-file`**: Prompt iteration is done by editing `prompts.py` directly

---

## Debug Output Files

```
<output>/
├── report.json              # Full JSON report with all results
└── (per-phase artifacts are embedded in the JSON)
```

The JSON report contains:
- `claims`: Extracted claims with indices
- `results`: Per-claim verification results (verdict, confidence, evidence, explanation, document_id, chunk_id, consistency_score)
- `dimensions`: Multi-dimensional scores (groundedness, contradiction_rate, hallucination_rate, completeness)
- `hallucination_flags`: Claims that are contradicted or lack evidence
- `summary`: Human-readable summary
