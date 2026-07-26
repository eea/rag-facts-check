# RAG Facts Check

A modular system for verifying RAG-generated answers against their source documents.

## Overview

RAG (Retrieval-Augmented Generation) systems can hallucinate — generating answers that aren't grounded in the retrieved documents. This system checks RAG answers by:

1. **Extracting** atomic factual claims from the generated answer
2. **Verifying** each claim against the source documents
3. **Aggregating** results into a confidence score, verdict, and detailed report

## Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Single-Prompt Verification** | Feed answer + docs to LLM, ask "is this supported?" | Simple, fast | No per-claim breakdown |
| **Claim Extraction + Verification** ⭐ | Extract claims, verify each against docs | Granular, cites evidence, per-claim scores | More compute |
| **NLI-based** | Use Natural Language Inference models | Fast, model-based | Requires specialized NLI model |
| **Self-Refine / Iterative** | LLM self-critiques & refines | Can fix errors | Complex, may not converge |
| **QA-based** | Ask "does the doc say X?" per claim | Precise | Requires QA model |

This implementation uses **Claim Extraction + Verification** — it gives you all three output types (confidence score, evidence citations, per-claim breakdown) and works well with local LLMs.

## Quick Start

```python
from rag_facts_check import RAGFactsChecker, MockLLM

# Use MockLLM for testing, or implement your own LLM adapter
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
├── models.py         # Data classes (Claim, VerificationResult, CheckReport)
├── llm.py            # LLM interface + adapters (HF, API, Chat, Mock)
├── prompts.py        # Prompt templates for extraction & verification
└── checker.py        # Core pipeline (Extractor, Verifier, Aggregator)
```

### Data Flow

```
RAG Answer ──► ClaimExtractor ──► [Claim 1, Claim 2, ...]
                                       │
                                       ▼
                               ClaimVerifier ──► [Result 1, Result 2, ...]
                                       │
                                       ▼
                              RAGFactsChecker._aggregate
                                       │
                                       ▼
                              CheckReport (confidence, verdict, evidence)
```

## LLM Integration

The system uses an abstract `LLM` interface with a single `generate(prompt) -> str` method. You need to implement this for your local model:

### Option 1: Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_facts_check import HuggingFaceLLM, RAGFactsChecker

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

llm = HuggingFaceLLM(model, tokenizer, chat_format=True)
checker = RAGFactsChecker(llm)
```

### Option 2: HTTP API (vLLM, Ollama, llama.cpp server)

```python
from rag_facts_check import APILLM, RAGFactsChecker

llm = APILLM("http://localhost:8000/v1/completions", model_name="my-model")
checker = RAGFactsChecker(llm)
```

### Option 3: Custom Local Model

```python
from rag_facts_check import LLM, RAGFactsChecker

class MyLocalLLM(LLM):
    def generate(self, prompt, max_new_tokens=512, temperature=0.1, **kwargs):
        # Your model inference here
        return generated_text

llm = MyLocalLLM()
checker = RAGFactsChecker(llm)
```

## Output Format

The `CheckReport` contains:

- **`overall_confidence`** (0-100): Weighted confidence score
- **`overall_verdict`**: One of `fully_supported`, `mostly_supported`, `partially_supported`, `largely_unsupported`, `no_claims`
- **`claims`**: List of extracted claims with indices
- **`results`**: Per-claim verification results (verdict, confidence, evidence, explanation)
- **`hallucination_flags`**: Claims that are contradicted or lack evidence
- **`summary`**: Human-readable summary

### Verdicts

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Source documents contain clear evidence supporting the claim |
| `CONTRADICTED` | Source documents contain clear evidence contradicting the claim |
| `NOT ENOUGH INFO` | Source documents don't contain sufficient information |

## Configuration

```python
checker = RAGFactsChecker(
    llm=llm,
    max_claims=10,           # Limit claims for latency control
    max_new_tokens=512,      # LLM generation length
    max_docs_chars=8000,     # Truncate documents to fit context
    max_chars_per_doc=2000,  # Truncate individual documents
)
```

## Prompt Customization

You can customize the prompts by modifying `prompts.py` or passing custom prompt templates:

```python
from rag_facts_check.checker import ClaimExtractor, ClaimVerifier

extractor = ClaimExtractor(llm)
extractor.prompt_template = "Your custom extraction prompt..."

verifier = ClaimVerifier(llm)
verifier.prompt_template = "Your custom verification prompt..."
```

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
