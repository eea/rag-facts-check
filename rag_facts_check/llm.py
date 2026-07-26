"""
LLM interface abstractions for local/self-hosted models.

Provides a uniform ``generate(prompt) -> str`` interface so the checker
works with any model backend.  Implement ``LLM`` for your specific
local setup, or use one of the provided adapters.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class LLM(ABC):
    """Abstract base class for LLM backends.

    Subclass this and implement :meth:`generate` to plug in your local
    model.  The method receives a full prompt string and must return
    the generated text string.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        """Generate text from *prompt*.

        Args:
            prompt: Full prompt string (already formatted).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Backend-specific keyword arguments.

        Returns:
            Generated text string.
        """
        ...


class HuggingFaceLLM(LLM):
    """Adapter for Hugging Face Transformers models.

    Works with any decoder-only or encoder-decoder model that supports
    ``model.generate()``.

    Example::

        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        mdl = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        llm = HuggingFaceLLM(mdl, tok)
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: str = "auto",
        chat_format: bool = False,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch is required for HuggingFaceLLM")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.chat_format = chat_format

        if device == "auto":
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.model = self.model.to(device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature

        if self.chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Strip the prompt from the output (model echoes it)
        if text.startswith(prompt):
            text = text[len(prompt):]

        return text.strip()


class APILLM(LLM):
    """Adapter for HTTP-based LLM APIs (vLLM, Ollama, llama.cpp server, etc.).

    Example::

        llm = APILLM("http://localhost:8000/v1/completions",
                     model_name="my-model")
    """

    def __init__(
        self,
        api_url: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        if not _HAS_REQUESTS:
            raise ImportError("requests is required for APILLM")
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if self.model_name:
            payload["model"] = self.model_name

        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Handle different API response formats
        if "choices" in data:
            return data["choices"][0].get("text", data["choices"][0].get("message", {}).get("content", ""))
        elif "generated_text" in data:
            return data["generated_text"]
        elif "response" in data:
            return data["response"]
        else:
            return str(data)


class ChatLLM(LLM):
    """Wrapper that converts a simple prompt into a chat conversation.

    Useful when your model expects a chat format (e.g., Llama-2-Chat,
    Mistral-Instruct) but you want to pass plain prompts.

    Example::

        base_llm = HuggingFaceLLM(model, tokenizer, chat_format=False)
        chat_llm = ChatLLM(base_llm, system_prompt="You are a fact-checker.")
    """

    def __init__(self, base_llm: LLM, system_prompt: str = ""):
        self.base_llm = base_llm
        self.system_prompt = system_prompt

    def _format_chat(self, prompt: str) -> str:
        """Format a plain prompt as a chat conversation.

        Override this method if your model uses a different format.
        """
        parts = []
        if self.system_prompt:
            parts.append(f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n")
        parts.append(prompt)
        parts.append(" [/INST]")
        return "".join(parts)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        formatted = self._format_chat(prompt)
        return self.base_llm.generate(formatted, max_new_tokens, temperature, **kwargs)


class MockLLM(LLM):
    """Mock LLM for testing and development.

    Returns predefined responses based on keywords in the prompt.
    """

    def __init__(self):
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        self.call_count += 1

        lower = prompt.lower()
        if "extract all factual claims" in lower or "extract claims from the following" in lower:
            return self._mock_claims_response(prompt)
        elif "verify whether a claim is supported" in lower or "verify the following claim" in lower:
            return self._mock_verification_response(prompt)
        else:
            return "Mock response for: " + prompt[:100]

    def _mock_claims_response(self, prompt: str) -> str:
        """Mock claim extraction: split text into sentences and treat each as a claim."""
        # Extract the text portion from the prompt (between "Text:\n" and "\nList each claim")
        text_start = prompt.find("Text:")
        if text_start >= 0:
            text_start += 5  # skip "Text:"
            # Find the end: look for the instructions that follow the text
            text_end = prompt.find("List each claim", text_start)
            if text_end < 0:
                text_end = prompt.find("Claims:", text_start)
            if text_end < 0:
                text_end = len(prompt)
            text = prompt[text_start:text_end].strip()
        else:
            text = "Paris is the capital of France. The Eiffel Tower was built in 1889."

        # Simple sentence-based claim extraction for mock
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for i, s in enumerate(sentences, 1):
            s = s.strip().rstrip('.')
            if s and len(s) > 5:
                claims.append(f"CLAIM {i}: {s}.")
        if not claims:
            return "NO CLAIMS"
        return "\n".join(claims)

    def _mock_verification_response(self, prompt: str) -> str:
        """Mock verification: check if claim is supported/contradicted by docs."""
        # Extract claim from prompt
        claim_match = re.search(r'Claim:\s*\n(.+?)\n\nSource Documents:', prompt, re.DOTALL)
        claim = claim_match.group(1).strip() if claim_match else "unknown claim"

        # Extract documents from prompt
        docs_match = re.search(r'Source Documents:\s*\n(.+?)\n\nInstructions:', prompt, re.DOTALL)
        docs_text = docs_match.group(1).strip() if docs_match else ""

        # Simple keyword-based mock verification
        lower_claim = claim.lower()
        lower_docs = docs_text.lower()

        # Check for contradiction (claim says one thing, docs say another)
        if "berlin" in lower_claim and "paris" in lower_docs:
            return f"""VERDICT: CONTRADICTED
CONFIDENCE: 85
EVIDENCE: "The Louvre Museum is one of the world's largest museums, located in Paris, France."
EXPLANATION: The source documents state the Louvre is in Paris, but the claim says it is in Berlin. This is a direct contradiction."""

        # Check for support
        if "capital of france" in lower_claim and "capital of france" in lower_docs:
            return f"""VERDICT: SUPPORTED
CONFIDENCE: 95
EVIDENCE: "Paris is the capital of France."
EXPLANATION: The source document explicitly states that Paris is the capital of France, which directly supports the claim."""

        if "eiffel tower" in lower_claim and "1889" in lower_docs:
            return f"""VERDICT: SUPPORTED
CONFIDENCE: 90
EVIDENCE: "The Eiffel Tower was constructed between 1887 and 1889."
EXPLANATION: The source document confirms the Eiffel Tower was built in 1889, supporting the claim."""

        # Default: not enough info
        return f"""VERDICT: NOT ENOUGH INFO
CONFIDENCE: 60
EVIDENCE: N/A
EXPLANATION: The source documents do not contain sufficient information to verify this claim."""
