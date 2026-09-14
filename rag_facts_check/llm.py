"""
LLM interface abstractions for local/self-hosted models.

Provides a uniform ``generate(prompt) -> str`` interface so the checker
works with any model backend.  Implement ``LLM`` for your specific
local setup, or use one of the provided adapters.
"""

import asyncio
import logging
from abc import ABC, abstractmethod

log = logging.getLogger("rag_facts_check")

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

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


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
        max_new_tokens: int | None = None,
        temperature: float | None = None,
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
            text = text[len(prompt) :]

        return text.strip()


class APILLM(LLM):
    """Adapter for HTTP-based LLM APIs (vLLM, Ollama, llama.cpp server, etc.).

    Supports both completions and chat completions endpoints.

    Example::

        # Completions endpoint
        llm = APILLM("http://localhost:8000/v1/completions",
                     model_name="my-model")

        # Chat completions endpoint (for instruction-tuned models)
        llm = APILLM("http://localhost:4002/v1/chat/completions",
                     model_name="gemma", chat_mode=True)
    """

    def __init__(
        self,
        api_url: str,
        model_name: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        chat_mode: bool = False,
    ):
        if not _HAS_REQUESTS:
            raise ImportError("requests is required for APILLM")
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.chat_mode = chat_mode

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.chat_mode:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                **kwargs,
            }
        else:
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
            choice = data["choices"][0]
            content = choice.get("text", choice.get("message", {}).get("content"))
            # Reasoning models can return null content when the reasoning
            # output exhausts the token budget — surface this explicitly.
            if content is None:
                raise ValueError(
                    "LLM returned null content "
                    f"(finish_reason={choice.get('finish_reason')!r}); "
                    "reasoning tokens may have exhausted the max_tokens budget"
                )
            return content
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


class AsyncAPILLM(LLM):
    """Async adapter for HTTP-based LLM APIs using httpx.

    Drop-in replacement for APILLM in async contexts (FastAPI, etc.).
    Supports both completions and chat completions endpoints.

    Example::

        llm = AsyncAPILLM(
            "http://localhost:4002/v1/chat/completions",
            model_name="gemma",
            chat_mode=True,
        )
    """

    def __init__(
        self,
        api_url: str,
        model_name: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        chat_mode: bool = False,
        timeout: float = 120.0,
        extra_body: dict | None = None,
        retries: int = 3,
    ):
        if not _HAS_HTTPX:
            raise ImportError("httpx is required for AsyncAPILLM")
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.chat_mode = chat_mode
        self.extra_body = extra_body or {}
        self.retries = max(0, retries)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.chat_mode:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                **kwargs,
            }
        else:
            payload = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                **kwargs,
            }
        if self.model_name:
            payload["model"] = self.model_name

        payload = {**self.extra_body, **payload}

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return await self._request_once(payload, headers)
            except (ValueError, httpx.TransportError) as exc:
                # Some gateways intermittently return 200 with null content
                # (e.g. reasoning models exhausting the token budget). Retry.
                last_error = exc
                if attempt < self.retries:
                    log.warning(
                        "AsyncAPILLM: attempt %d/%d failed (%s), retrying in 2s",
                        attempt + 1, self.retries + 1, exc,
                    )
                    await asyncio.sleep(2.0)
        raise last_error  # type: ignore[misc]

    async def _request_once(self, payload: dict, headers: dict) -> str:
        """Perform a single API request and parse the response text."""
        response = await self._client.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Handle different API response formats
        if "choices" in data:
            choice = data["choices"][0]
            content = choice.get("text", choice.get("message", {}).get("content"))
            # Reasoning models can return null content when the reasoning
            # output exhausts the token budget — surface this explicitly.
            if content is None:
                raise ValueError(
                    "LLM returned null content "
                    f"(finish_reason={choice.get('finish_reason')!r}); "
                    "reasoning tokens may have exhausted the max_tokens budget"
                )
            return content
        elif "generated_text" in data:
            return data["generated_text"]
        elif "response" in data:
            return data["response"]
        else:
            return str(data)

    async def close(self) -> None:
        """Close the underlying httpx async client."""
        await self._client.aclose()
