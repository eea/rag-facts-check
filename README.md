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
├── models.py         # Data classes: Claim, VerificationResult, CheckReport, Span
├── llm.py            # LLM interface + adapters (HF, API, Chat, AsyncAPI)
├── prompts.py        # Prompt templates for extraction & verification
├── retriever.py      # Evidence retrieval (chunk-based lexical matching)
├── spans.py          # Span matching (claim → answer, evidence → document)
├── checker.py        # Core pipeline: Extractor → Verifier → Aggregator
└── server.py         # FastAPI web service (POST /check, POST /halloumi/generate)
```

```
mock_datasets/       # Synthetic test datasets (JSON)
tests/               # Pytest test suite
├── conftest.py       # Shared fixtures
├── test_models.py    # Data model tests
├── test_retriever.py # Evidence retrieval tests
├── test_checker.py   # Core pipeline tests
└── test_integration.py # End-to-end tests
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

## Web Service

The package includes a FastAPI web service for async fact-checking from any client.

### Installation

```bash
make setup-server    # Installs fastapi, uvicorn, httpx, python-dotenv
```

### Running the server

```bash
# Development (auto-reload)
make serve

# Production
make serve-prod
```

The server reads LLM configuration from `.env`:

```env
LLM_API_BASE=http://localhost:4002/v1
LLM_API_KEY=not-needed
LLM_MODEL=gemma
LLM_TEMPERATURE=0.1
```

### Endpoints

#### `POST /check` — Full fact-checking report

Request:

```json
{
  "answer": "Paris is the capital of France.",
  "documents": [
    { "doc_id": "doc_1", "title": "Paris overview", "text": "Paris is the capital..." },
    { "doc_id": "doc_2", "title": "Eiffel Tower", "text": "The Eiffel Tower..." }
  ],
  "options": {
    "num_consistency_runs": 1,
    "evidence_first": true,
    "use_evidence_retrieval": true
  }
}
```

Response: Full `CheckReport` with `overall_verdict`, `dimensions`, `claims` (with `span` offsets), `results` (with `evidence_span` offsets), and `hallucination_flags`.

#### `POST /halloumi/generate` — Halloumi-compatible endpoint

Drop-in replacement for the existing halloumi middleware. Accepts the same request format and returns a response compatible with the frontend's `ClaimModal`, `ClaimSegments`, and `Citation` components.

Request:

```json
{
  "answer": "Paris is the capital of France.",
  "sources": ["Paris is the capital...", "The Eiffel Tower..."]
}
```

Response:

```json
{
  "claims": [
    {
      "startOffset": 0,
      "endOffset": 32,
      "segmentIds": ["0"],
      "score": 0.95,
      "rationale": "Document states this explicitly."
    }
  ],
  "segments": {
    "0": { "startOffset": 0, "endOffset": 22 }
  }
}
```

#### `GET /health` — Health check

Returns `{"status": "ok", "version": "0.2.0"}`.

### Span-Level Grounding

Both endpoints return character offsets for clickable highlighting:

- **`claims[].span`**: `{start, end}` offsets in the original answer text
- **`results[].evidence_span`**: `{start, end}` offsets in the source document text

The client can use these to render clickable spans in the answer that link to highlighted evidence in the source documents.

## Output Format

The `CheckReport` contains:

- **`overall_confidence`** (0-100): Weighted confidence score
- **`overall_verdict`**: `fully_supported`, `mostly_supported`, `partially_supported`, `largely_unsupported`, `no_claims`
- **`dimensions`**: Multi-dimensional scores:
  - `groundedness`: % of claims supported by documents
  - `contradiction_rate`: % of claims contradicted by documents
  - `hallucination_rate`: % of claims unsupported or contradicted
  - `completeness`: % of claims covered (same as groundedness without coverage analysis)
- **`claims`**: List of extracted claims with indices and `span` (character offsets in the answer)
- **`results`**: Per-claim verification results (verdict, confidence, evidence, explanation, document_id, chunk_id, consistency_score, evidence_span)
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

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_checker.py -v

# Run with coverage
python -m pytest tests/ --cov=rag_facts_check --cov-report=term-missing

# Run integration tests only
python -m pytest tests/test_integration.py -v
```

### MockLLM

The `MockLLM` class (in `rag_facts_check/testing/mocks.py`) provides deterministic
responses based on keyword matching, enabling testing without a real LLM.

```python
from rag_facts_check.testing import MockLLM

llm = MockLLM()
# llm.generate(prompt) returns predefined responses based on prompt content
# Tracks call_count for test assertions
```

### Test Datasets

Mock datasets in `mock_datasets/` provide realistic test cases:

- `climate_change_hallucinated.json` — 6 document chunks, 123-word answer with 5 claims
  (5.7°C vs 2-4°C, IPCC 2024 vs 2023, Arctic ice-free by 2035 vs 2040-2060)
- `renewable_energy_supported.json` — 6 document chunks, 124-word answer with 6 claims
  (30%, 42%, 89%, 340 GW — all supported by documents)
- `phosphorus_eutrophication.json` — 6 EEA documents, 2400-char answer with 22+ claims
  (real production data from the climate adapt chatbot)

### Running Examples

```bash
python example_usage.py
```

## Requirements

- Python 3.10+
- `torch` (for HuggingFaceLLM)
- `transformers` (for HuggingFaceLLM)
- `requests` (for APILLM)
- `fastapi`, `uvicorn`, `httpx`, `python-dotenv` (for the web service — `pip install rag-facts-check[server]`)
- Your local LLM backend of choice

## License

MIT
