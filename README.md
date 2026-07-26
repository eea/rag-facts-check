# RAG Facts Check

A modular system for verifying RAG-generated answers against their source documents using claim extraction + per-claim verification.

## Overview

RAG (Retrieval-Augmented Generation) systems can hallucinate — generating answers that aren't grounded in the retrieved documents. This system checks RAG answers by:

1. **Extracting** atomic factual claims from the generated answer
2. **Retrieving** relevant document chunks for each claim (optional)
3. **Verifying** each claim against the source documents
4. **Aggregating** results into a confidence score, verdict, and detailed report

## Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Single-Prompt Verification** | Feed answer + docs to LLM, ask "is this supported?" | Simple, fast | No per-claim breakdown |
| **Claim Extraction + Verification** ⭐ | Extract claims, verify each against docs | Granular, cites evidence, per-claim scores | More compute |
| **NLI-based** | Use Natural Language Inference models | Fast, model-based | Requires specialized NLI model |
| **Two-Agent Verification** | Separate LLM verifies the answer | Independent check | More compute |
| **Reverse QA** | Re-answer the question from docs, compare | Catches hallucinations | Indirect comparison |
| **Evidence-First Prompting** | LLM extracts evidence before deciding | Reduces hallucinated evaluations | More prompt tokens |
| **Self-Consistency** | Run verification N times, majority vote | More robust | More compute |
| **Span-Level Verification** | Cite specific document/paragraph IDs | Precise, enterprise-ready | Requires structured docs |

This implementation uses **Claim Extraction + Verification** with optional **Evidence Retrieval**, **Evidence-First Prompting**, **Self-Consistency**, and **Multi-Dimensional Scoring**.

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

## Architecture

```
rag_facts_check/
├── __init__.py       # Package exports
├── models.py         # Data classes: Claim, VerificationResult, CheckReport
├── llm.py            # LLM interface + adapters (HF, API, Chat, Mock)
├── prompts.py        # Prompt templates for extraction & verification
├── retriever.py      # Evidence retrieval (chunk-based lexical matching)
└── checker.py        # Core pipeline: Extractor → Verifier → Aggregator
```

### Data Flow

```
RAG Answer ──► ClaimExtractor ──► [Claim 1, Claim 2, ...]
                                       │
                                       ▼
                              EvidenceRetriever (optional)
                              Retrieves relevant chunks per claim
                                       │
                                       ▼
                              ClaimVerifier ──► [Result 1, Result 2, ...]
                              (with self-consistency + evidence-first)
                                       │
                                       ▼
                              RAGFactsChecker._aggregate
                                       │
                                       ▼
                              CheckReport (confidence, verdict, evidence,
                                          dimensions, hallucination_flags)
```

## LLM Integration

The system uses an abstract `LLM` interface with a single `generate(prompt) -> str` method. You can plug in your model via:

### Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_facts_check import HuggingFaceLLM, RAGFactsChecker

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm = HuggingFaceLLM(model, tokenizer, chat_format=True)
checker = RAGFactsChecker(llm)
```

### HTTP API (vLLM, Ollama, llama.cpp server)

```python
from rag_facts_check import APILLM, RAGFactsChecker
llm = APILLM("http://localhost:8000/v1/completions", model_name="my-model")
checker = RAGFactsChecker(llm)
```

### Custom Local Model

```python
from rag_facts_check import LLM, RAGFactsChecker

class MyLocalLLM(LLM):
    def generate(self, prompt, max_new_tokens=512, temperature=0.1, **kwargs):
        # Your model inference here
        return generated_text

llm = MyLocalLLM()
checker = RAGFactsChecker(llm)
```

## Configuration

```python
checker = RAGFactsChecker(
    llm=llm,
    max_claims=10,                # Limit claims for latency control
    max_new_tokens=512,           # LLM generation length
    max_docs_chars=8000,          # Truncate docs to fit context
    max_chars_per_doc=2000,       # Truncate individual documents
    num_consistency_runs=3,       # Self-consistency: run 3 times, majority vote
    evidence_first=True,          # Use evidence-first multi-step prompting
    use_evidence_retrieval=True,  # Retrieve relevant chunks per claim
    retriever=EvidenceRetriever(  # Custom retriever
        chunk_size=200,
        top_k=3,
    ),
)
```

## Output Format

The `CheckReport` contains:

- **`overall_confidence`** (0-100): Weighted confidence score
- **`overall_verdict`**: `fully_supported`, `mostly_supported`, `partially_supported`, `largely_unsupported`, `no_claims`
- **`dimensions`**: Multi-dimensional scores:
  - `groundedness`: % of claims supported by documents
  - `contradiction_rate`: % of claims contradicted by documents
  - `hallucination_rate`: % of claims unsupported or contradicted
  - `completeness`: % of claims covered (same as groundedness without coverage analysis)
- **`claims`**: List of extracted claims with indices
- **`results`**: Per-claim verification results (verdict, confidence, evidence, explanation, document_id, chunk_id, consistency_score)
- **`hallucination_flags`**: Claims that are contradicted or lack evidence
- **`summary`**: Human-readable summary

### Verdicts

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Source documents contain clear evidence supporting the claim |
| `CONTRADICTED` | Source documents contain clear evidence contradicting the claim |
| `NOT ENOUGH INFO` | Source documents don't contain sufficient information |

## Advanced Features

### Evidence Retrieval

Instead of passing all documents to each claim verification, the retriever splits documents into chunks and retrieves only the most relevant ones per claim. This:
- Reduces context window usage
- Improves verification accuracy
- Speeds up inference

```python
from rag_facts_check import EvidenceRetriever

retriever = EvidenceRetriever(chunk_size=200, top_k=3)
checker = RAGFactsChecker(llm, retriever=retriever, use_evidence_retrieval=True)
```

### Evidence-First Prompting

The evidence-first prompt explicitly asks the LLM to extract evidence before deciding, reducing hallucinated evaluations:

```
Step 1: Extract relevant evidence
Step 2: Compare evidence to claim
Step 3: Verdict
Step 4: Output (VERDICT/CONFIDENCE/EVIDENCE/EXPLANATION)
```

### Self-Consistency

Run verification multiple times with different temperatures and aggregate via majority vote:

```python
checker = RAGFactsChecker(llm, num_consistency_runs=3)
```

### Span-Level Verification

When using evidence retrieval, results include `document_id` and `chunk_id` fields, enabling precise citation tracking.

## Testing

```bash
python example_usage.py
```

## Requirements

- Python 3.10+
- `torch` (for HuggingFaceLLM)
- `transformers` (for HuggingFaceLLM)
- `requests` (for APILLM)
- Your local LLM backend of choice

## License

MIT
