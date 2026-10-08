"""
rag_facts_check — A modular RAG answer fact-checking system.

Verifies RAG-generated answers against their source documents using a
claim-extraction + per-claim-verification pipeline.

Key features:
- Claim extraction from RAG answers
- Per-claim verification with verdict and up to 3 verbatim evidence quotes
- Evidence retrieval (relevant document chunks per claim)
- Evidence-first multi-step prompting
- Self-consistency (multiple verification runs)
- Multi-dimensional scoring (groundedness, contradiction_rate, etc.)
- Span-level verification (one evidence span per quote, with document index)
"""

from .checker import ClaimExtractor, ClaimVerifier, RAGFactsChecker
from .llm import APILLM, LLM, AsyncAPILLM, ChatLLM, HuggingFaceLLM
from .models import CheckReport, Claim, EvidenceSpan, Span, VerificationResult, score_label
from .retriever import DocumentChunk, EvidenceRetriever, LLMEvidenceRetriever

__all__ = [
    "Claim",
    "Span",
    "EvidenceSpan",
    "VerificationResult",
    "CheckReport",
    "score_label",
    "LLM",
    "HuggingFaceLLM",
    "APILLM",
    "AsyncAPILLM",
    "ChatLLM",
    "EvidenceRetriever",
    "LLMEvidenceRetriever",
    "DocumentChunk",
    "ClaimExtractor",
    "ClaimVerifier",
    "RAGFactsChecker",
]

__version__ = "0.2.0"
