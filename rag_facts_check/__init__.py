"""
rag_facts_check — A modular RAG answer fact-checking system.

Verifies RAG-generated answers against source documents using a
claim-extraction + per-claim-verification pipeline.
"""

from .models import Claim, VerificationResult, CheckReport
from .llm import LLM, HuggingFaceLLM, APILLM, ChatLLM, MockLLM
from .checker import ClaimExtractor, ClaimVerifier, RAGFactsChecker

__all__ = [
    "Claim",
    "VerificationResult",
    "CheckReport",
    "LLM",
    "HuggingFaceLLM",
    "APILLM",
    "ChatLLM",
    "MockLLM",
    "ClaimExtractor",
    "ClaimVerifier",
    "RAGFactsChecker",
]

__version__ = "0.1.0"
