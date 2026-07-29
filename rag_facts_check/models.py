"""Data models for the RAG fact-checking system."""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Answer quality score helpers
# ---------------------------------------------------------------------------


def score_label(score: float) -> str:
    """Map a 0-10 answer quality score to a human-readable label.

    Args:
        score: Answer quality score on a 0-10 scale.

    Returns:
        One of: "Excellent", "Good", "Acceptable", "Poor", "Failing", "No claims".
    """
    if score >= 9:
        return "Excellent"
    elif score >= 7:
        return "Good"
    elif score >= 5:
        return "Acceptable"
    elif score >= 3:
        return "Poor"
    elif score > 0:
        return "Failing"
    else:
        return "No claims"


@dataclass
class Span:
    """Character offset span within a text.

    Attributes:
        start: Start character offset (inclusive).
        end: End character offset (exclusive).
    """

    start: int
    end: int


@dataclass
class Claim:
    """A single factual claim extracted from a RAG-generated answer.

    Attributes:
        text: The claim text (may be rephrased for clarity).
        index: The 1-based index of the claim in the original answer.
        original_text: Exact verbatim fragment from the answer (for span matching).
        span: Character offsets of the claim in the original answer.
    """

    text: str
    index: int
    original_text: str = ""
    span: Span | None = None


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
        document_id: ID of the source document containing the evidence (if available).
        document_index: 0-based index of the source document containing the evidence
            (if the LLM identified it). Used for targeted evidence span matching.
        chunk_id: ID of the document chunk containing the evidence (if available).
        consistency_score: Agreement across multiple verification runs (0-1, for self-consistency).
    """

    claim: str
    claim_index: int
    verdict: str  # "supported" | "contradicted" | "not_enough_info"
    confidence: int  # 0-100
    evidence: str
    explanation: str
    document_id: str | None = None
    document_index: int | None = None
    chunk_id: str | None = None
    consistency_score: float | None = None
    evidence_span: Span | None = None


@dataclass
class CheckReport:
    """Aggregated report of fact-checking results for a RAG answer.

    Attributes:
        answer: The original RAG-generated answer.
        answer_score: Overall answer quality grade on a 0-10 scale.
            9-10=Excellent, 7-8=Good, 5-6=Acceptable, 3-4=Poor, 1-2=Failing, 0=No claims.
        overall_confidence: Overall confidence score 0-100.
        overall_verdict: One of "fully_supported", "mostly_supported",
            "partially_supported", "largely_unsupported", "no_claims".
        claims: List of extracted claims.
        results: List of per-claim verification results.
        summary: Human-readable summary of the results.
        hallucination_flags: Claims that are contradicted or lack evidence.
        dimensions: Multi-dimensional scores (groundedness, contradiction_rate, etc.).
    """

    answer: str
    answer_score: float = 0.0
    overall_confidence: float = 0.0
    overall_verdict: str = "no_claims"
    claims: list[Claim] = field(default_factory=list)
    results: list[VerificationResult] = field(default_factory=list)
    summary: str = ""
    hallucination_flags: list[VerificationResult] = field(default_factory=list)
    dimensions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the report to a dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "answer_score": self.answer_score,
            "overall_confidence": round(self.overall_confidence, 2),
            "overall_verdict": self.overall_verdict,
            "summary": self.summary,
            "dimensions": self.dimensions,
            "claims": [
                {
                    "index": c.index,
                    "text": c.text,
                    "original_text": c.original_text,
                    "span": ({"start": c.span.start, "end": c.span.end} if c.span else None),
                }
                for c in self.claims
            ],
            "results": [
                {
                    "claim_index": r.claim_index,
                    "claim": r.claim,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "explanation": r.explanation,
                    "document_id": r.document_id,
                    "document_index": r.document_index,
                    "chunk_id": r.chunk_id,
                    "consistency_score": r.consistency_score,
                    "evidence_span": (
                        {
                            "start": r.evidence_span.start,
                            "end": r.evidence_span.end,
                        }
                        if r.evidence_span
                        else None
                    ),
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
                    "document_id": r.document_id,
                    "document_index": r.document_index,
                    "chunk_id": r.chunk_id,
                    "consistency_score": r.consistency_score,
                    "evidence_span": (
                        {
                            "start": r.evidence_span.start,
                            "end": r.evidence_span.end,
                        }
                        if r.evidence_span
                        else None
                    ),
                }
                for r in self.hallucination_flags
            ],
        }
