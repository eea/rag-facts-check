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
class EvidenceSpan:
    """A verbatim evidence quote located in a source document.

    Attributes:
        quote: The verbatim quote as returned by the judge.
        start: Start character offset (inclusive) within the source document.
        end: End character offset (exclusive) within the source document.
        document_index: 0-based index of the source document the span was
            found in, or None if the quote could not be located.
    """

    quote: str
    start: int
    end: int
    document_index: int | None = None

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "document_index": self.document_index,
        }


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
        evidence: Verbatim quotes from the source documents (empty list when the
            judge found no evidence). A judge that returns a single string is
            coerced to a one-element list.
        explanation: Brief explanation of the reasoning.
        document_id: ID of the source document containing the evidence (if available).
        document_index: 0-based index of the source document containing the first
            evidence quote (as identified by the judge, or corrected by span
            matching). Used for targeted evidence span matching.
        chunk_id: ID of the document chunk containing the evidence (if available).
        consistency_score: Agreement across multiple verification runs (0-1, for self-consistency).
        evidence_spans: Located spans for the quotes that were actually found in the
            source documents. May be shorter than ``evidence`` when a quote could
            not be matched, and each span carries its own document index.
    """

    claim: str
    claim_index: int
    verdict: str  # "supported" | "contradicted" | "not_enough_info"
    confidence: int  # 0-100
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)
    document_id: str | None = None
    document_index: int | None = None
    chunk_id: str | None = None
    consistency_score: float | None = None
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)


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
            "results": [self._result_dict(r) for r in self.results],
            "hallucination_flags": [self._result_dict(r) for r in self.hallucination_flags],
        }

    @staticmethod
    def _result_dict(r: VerificationResult) -> dict:
        """Serialize a single verification result (shared by results and flags)."""
        return {
            "claim_index": r.claim_index,
            "claim": r.claim,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "evidence": list(r.evidence),
            "explanation": r.explanation,
            "document_id": r.document_id,
            "document_index": r.document_index,
            "chunk_id": r.chunk_id,
            "consistency_score": r.consistency_score,
            "evidence_spans": [s.to_dict() for s in r.evidence_spans],
        }
