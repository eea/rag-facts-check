"""
Testing utilities for the RAG fact-checking system.

Provides mock LLM implementations for testing without a real model.
"""

from .mocks import MockLLM

__all__ = ["MockLLM"]
