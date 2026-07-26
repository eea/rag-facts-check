"""
rag_facts_check — A modular RAG answer fact-checking system.

Verifies RAG-generated answers against their source documents using a
claim-extraction + per-claim-verification pipeline.

Key features:
- Claim extraction from RAG answers
- Per-claim verification with verdict, confidence, and evidence
- Evidence retrieval (relevant document chunks per claim)
- Evidence-first multi-step prompting
- Self-consistency (multiple verification runs)
- Multi-dimensional scoring (groundedness, contradiction_rate, etc.)
- Span-level verification (document_id and chunk_id in results)
"""

from .models import Claim, VerificationResult, CheckReport
from .llm import LLM, HuggingFaceLLM, APILLM, ChatLLM
from .retriever import EvidenceRetriever, DocumentChunk
from .checker import ClaimExtractor, ClaimVerifier, RAGFactsChecker
from .testing import MockLLM

__all__ = [
    "Claim",
    "VerificationResult",
    "CheckReport",
    "LLM",
    "HuggingFaceLLM",
    "APILLM",
    "ChatLLM",
    "MockLLM",
    "EvidenceRetriever",
    "DocumentChunk",
    "ClaimExtractor",
    "ClaimVerifier",
    "RAGFactsChecker",
]

__version__ = "0.2.0"
