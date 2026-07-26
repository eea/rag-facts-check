"""Data models for the RAG fact-checking system."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Claim:
    """A single factual claim extracted from a RAG-generated answer.

    Attributes:
        text: The claim text.
        index: The 1-based index of the claim in the original answer.
    """

    text: str
    index: int


@dataclass
class VerificationResult:
    """Result of verifying a single claim against source documents.

    Attributes:
        claim: The claim text that was verified.
        claim_index: The index of the claim.
        verdict: One of "supported", "contradicted", "not_enough_info".
        confidence: Confidence score 0-100.
        evidence: Exact quote from source documents (or "N/A").
        explanation: Brief explanation of the reasoning.
    """

    claim: str
    claim_index: int
    verdict: str  # "supported" | "contradicted" | "not_enough_info"
    confidence: int  # 0-100
    evidence: str
    explanation: str


@dataclass
class CheckReport:
    """Aggregated report of fact-checking results for a RAG answer.

    Attributes:
        answer: The original RAG-generated answer.
        overall_confidence: Overall confidence score 0-100.
        overall_verdict: One of "fully_supported", "mostly_supported",
            "partially_supported", "largely_unsupported", "no_claims".
        claims: List of extracted claims.
        results: List of per-claim verification results.
        summary: Human-readable summary of the results.
        hallucination_flags: Claims that are contradicted or lack evidence.
    """

    answer: str
    overall_confidence: float
    overall_verdict: str
    claims: List[Claim] = field(default_factory=list)
    results: List[VerificationResult] = field(default_factory=list)
    summary: str = ""
    hallucination_flags: List[VerificationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the report to a dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "overall_confidence": round(self.overall_confidence, 2),
            "overall_verdict": self.overall_verdict,
            "summary": self.summary,
            "claims": [
                {"index": c.index, "text": c.text} for c in self.claims
            ],
            "results": [
                {
                    "claim_index": r.claim_index,
                    "claim": r.claim,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "explanation": r.explanation,
                }
                for r in self.results
            ],
            "hallucination_flags": [
                {
                    "claim_index": r.claim_index,
                    "claim": r.claim,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "explanation": r.explanation,
                }
                for r in self.hallucination_flags
            ],
        }
