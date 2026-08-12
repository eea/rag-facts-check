---
type: Guide
title: LLM Integration
description: Plugging in Hugging Face, HTTP API, or custom models.
tags: [llm, integration, setup]
timestamp: '2025-01-01T00:00:00Z'
---

# LLM Integration

The system uses an abstract `LLM` interface with a single `generate(prompt) -> str` method. You can plug in your model via any of the built-in adapters or by implementing the interface yourself.

## Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_facts_check import HuggingFaceLLM, RAGFactsChecker

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm = HuggingFaceLLM(model, tokenizer, chat_format=True)
checker = RAGFactsChecker(llm)
```

## HTTP API (vLLM, Ollama, llama.cpp server)

```python
from rag_facts_check import APILLM, RAGFactsChecker

llm = APILLM("http://localhost:8000/v1/completions", model_name="my-model")
checker = RAGFactsChecker(llm)
```

## Custom Local Model

Implement the `LLM` abstract class:

```python
from rag_facts_check import LLM, RAGFactsChecker

class MyLocalLLM(LLM):
    def generate(self, prompt, max_new_tokens=512, temperature=0.1, **kwargs):
        # Your model inference here
        return generated_text

llm = MyLocalLLM()
checker = RAGFactsChecker(llm)
```

## Available Adapters

| Adapter | Use case |
|---|---|
| `HuggingFaceLLM` | Local transformers models with optional chat formatting |
| `APILLM` | HTTP completion endpoints (vLLM, Ollama, llama.cpp) |
| `ChatLLM` | Chat-completion format APIs (OpenAI-compatible) |
| `AsyncAPILLM` | Async HTTP API client for the web service |
| `AsyncMock` (stdlib) | Deterministic responses for testing — see [Testing](/guides/testing.md) |
