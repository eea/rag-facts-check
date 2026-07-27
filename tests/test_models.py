"""
Tests for data models in rag_facts_check.models.

Covers Claim, VerificationResult, CheckReport, and the to_dict()
serialization method.
"""

from rag_facts_check.models import CheckReport, Claim, VerificationResult


class TestClaim:
    """Tests for the Claim dataclass."""

    def test_claim_creation(self):
        claim = Claim(text="Paris is the capital of France.", index=1)
        assert claim.text == "Paris is the capital of France."
        assert claim.index == 1

    def test_claim_empty_text(self):
        claim = Claim(text="", index=0)
        assert claim.text == ""
        assert claim.index == 0

    def test_claim_index_is_int(self):
        claim = Claim(text="Test claim.", index=42)
        assert isinstance(claim.index, int)

    def test_claim_repr(self):
        claim = Claim(text="Test.", index=1)
        repr_str = repr(claim)
        assert "Claim" in repr_str
        assert "Test." in repr_str


class TestVerificationResult:
    """Tests for the VerificationResult dataclass."""

    def test_result_creation(self):
        result = VerificationResult(
            claim="Paris is the capital of France.",
            claim_index=1,
            verdict="supported",
            confidence=95,
            evidence="Paris is the capital of France.",
            explanation="Document explicitly states this.",
        )
        assert result.claim == "Paris is the capital of France."
        assert result.claim_index == 1
        assert result.verdict == "supported"
        assert result.confidence == 95
        assert result.evidence == "Paris is the capital of France."
        assert result.explanation == "Document explicitly states this."

    def test_result_default_optional_fields(self):
        result = VerificationResult(
            claim="Test.",
            claim_index=1,
            verdict="not_enough_info",
            confidence=50,
            evidence="N/A",
            explanation="No info.",
        )
        assert result.document_id is None
        assert result.chunk_id is None
        assert result.consistency_score is None

    def test_result_with_span_fields(self):
        result = VerificationResult(
            claim="Test.",
            claim_index=1,
            verdict="supported",
            confidence=90,
            evidence="Evidence text.",
            explanation="Explanation.",
            document_id="doc_1",
            chunk_id="0",
            consistency_score=1.0,
        )
        assert result.document_id == "doc_1"
        assert result.chunk_id == "0"
        assert result.consistency_score == 1.0

    def test_result_confidence_bounds(self):
        """Confidence should be an int in 0-100 range."""
        result = VerificationResult(
            claim="Test.",
            claim_index=1,
            verdict="supported",
            confidence=0,
            evidence="N/A",
            explanation="None.",
        )
        assert result.confidence == 0

        result2 = VerificationResult(
            claim="Test.",
            claim_index=1,
            verdict="supported",
            confidence=100,
            evidence="N/A",
            explanation="None.",
        )
        assert result2.confidence == 100


class TestCheckReport:
    """Tests for the CheckReport dataclass and to_dict()."""

    def test_report_creation(self):
        report = CheckReport(
            answer="Paris is the capital of France.",
            overall_confidence=95.0,
            overall_verdict="fully_supported",
        )
        assert report.answer == "Paris is the capital of France."
        assert report.overall_confidence == 95.0
        assert report.overall_verdict == "fully_supported"

    def test_report_default_fields(self):
        report = CheckReport(
            answer="Test answer.",
            overall_confidence=50.0,
            overall_verdict="partially_supported",
        )
        assert report.claims == []
        assert report.results == []
        assert report.summary == ""
        assert report.hallucination_flags == []
        assert report.dimensions == {}

    def test_report_to_dict_basic(self):
        report = CheckReport(
            answer="Paris is the capital of France.",
            overall_confidence=95.0,
            overall_verdict="fully_supported",
            summary="All claims supported.",
            dimensions={"groundedness": 100.0},
        )
        d = report.to_dict()
        assert d["answer"] == "Paris is the capital of France."
        assert d["overall_confidence"] == 95.0
        assert d["overall_verdict"] == "fully_supported"
        assert d["summary"] == "All claims supported."
        assert d["dimensions"] == {"groundedness": 100.0}
        assert d["claims"] == []
        assert d["results"] == []
        assert d["hallucination_flags"] == []

    def test_report_to_dict_with_claims(self):
        claims = [
            Claim(text="Paris is the capital of France.", index=1),
            Claim(text="The Eiffel Tower was built in 1889.", index=2),
        ]
        report = CheckReport(
            answer="Paris is the capital of France. The Eiffel Tower was built in 1889.",
            overall_confidence=92.5,
            overall_verdict="mostly_supported",
            claims=claims,
        )
        d = report.to_dict()
        assert len(d["claims"]) == 2
        assert d["claims"][0]["index"] == 1
        assert d["claims"][0]["text"] == "Paris is the capital of France."
        assert d["claims"][1]["index"] == 2

    def test_report_to_dict_with_results(self):
        results = [
            VerificationResult(
                claim="Paris is the capital of France.",
                claim_index=1,
                verdict="supported",
                confidence=95,
                evidence="Paris is the capital of France.",
                explanation="Document states this.",
            ),
            VerificationResult(
                claim="The Louvre is in Berlin.",
                claim_index=2,
                verdict="contradicted",
                confidence=85,
                evidence="The Louvre is in Paris.",
                explanation="Documents say Paris.",
            ),
        ]
        report = CheckReport(
            answer="Paris is the capital of France. The Louvre is in Berlin.",
            overall_confidence=90.0,
            overall_verdict="partially_supported",
            results=results,
            hallucination_flags=[results[1]],
        )
        d = report.to_dict()
        assert len(d["results"]) == 2
        assert d["results"][0]["verdict"] == "supported"
        assert d["results"][1]["verdict"] == "contradicted"
        assert len(d["hallucination_flags"]) == 1
        assert d["hallucination_flags"][0]["claim_index"] == 2

    def test_report_to_dict_with_span_fields(self):
        results = [
            VerificationResult(
                claim="Test.",
                claim_index=1,
                verdict="supported",
                confidence=90,
                evidence="Evidence.",
                explanation="Explanation.",
                document_id="doc_1",
                chunk_id="0",
                consistency_score=1.0,
            ),
        ]
        report = CheckReport(
            answer="Test.",
            overall_confidence=90.0,
            overall_verdict="fully_supported",
            results=results,
        )
        d = report.to_dict()
        assert d["results"][0]["document_id"] == "doc_1"
        assert d["results"][0]["chunk_id"] == "0"
        assert d["results"][0]["consistency_score"] == 1.0

    def test_report_to_dict_serializable(self):
        """to_dict() output should be JSON-serializable."""
        import json

        report = CheckReport(
            answer="Test.",
            overall_confidence=50.0,
            overall_verdict="no_claims",
            dimensions={"groundedness": 0.0},
        )
        d = report.to_dict()
        json_str = json.dumps(d)
        assert json.loads(json_str) == d

    def test_report_to_dict_round_trip(self):
        """to_dict() output should preserve all fields."""
        import json

        claims = [Claim(text="Test claim.", index=1)]
        results = [
            VerificationResult(
                claim="Test claim.",
                claim_index=1,
                verdict="supported",
                confidence=95,
                evidence="Evidence.",
                explanation="Explanation.",
            )
        ]
        report = CheckReport(
            answer="Test answer.",
            overall_confidence=95.0,
            overall_verdict="fully_supported",
            claims=claims,
            results=results,
            summary="All supported.",
            dimensions={"groundedness": 100.0, "hallucination_rate": 0.0},
        )
        d = report.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        assert d2["answer"] == report.answer
        assert d2["overall_confidence"] == 95.0
        assert d2["overall_verdict"] == "fully_supported"
        assert d2["claims"][0]["text"] == "Test claim."
        assert d2["results"][0]["verdict"] == "supported"
