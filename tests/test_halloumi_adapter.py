"""Tests for the halloumi response adapter."""

import pytest

from rag_facts_check.models import CheckReport, Claim, Span, VerificationResult
from rag_facts_check.server import _to_halloumi_format


@pytest.fixture
def sample_report():
    """Create a sample CheckReport for testing."""
    return CheckReport(
        answer="Paris is the capital of France. The Eiffel Tower was built in 1889.",
        overall_confidence=90.0,
        overall_verdict="mostly_supported",
        claims=[
            Claim(
                text="Paris is the capital of France.",
                index=1,
                span=Span(start=0, end=32),
            ),
            Claim(
                text="The Eiffel Tower was built in 1889.",
                index=2,
                span=Span(start=33, end=68),
            ),
        ],
        results=[
            VerificationResult(
                claim="Paris is the capital of France.",
                claim_index=1,
                verdict="supported",
                confidence=95,
                evidence="Paris is the capital.",
                explanation="Document states this explicitly.",
                evidence_span=Span(start=0, end=22),
            ),
            VerificationResult(
                claim="The Eiffel Tower was built in 1889.",
                claim_index=2,
                verdict="supported",
                confidence=90,
                evidence="Eiffel Tower built 1889.",
                explanation="Document confirms this.",
                evidence_span=Span(start=50, end=75),
            ),
        ],
    )


class TestToHalloumiFormat:
    """Tests for _to_halloumi_format adapter."""

    def test_basic_conversion(self, sample_report):
        sources = [
            "Paris is the capital. More text here.",
            "Eiffel Tower built 1889. Additional context.",
        ]
        result = _to_halloumi_format(sample_report, sources)

        assert "claims" in result
        assert "segments" in result
        assert len(result["claims"]) == 2

    def test_claim_offsets_preserved(self, sample_report):
        sources = ["Some source text."]
        result = _to_halloumi_format(sample_report, sources)

        claim = result["claims"][0]
        assert claim["startOffset"] == 0
        assert claim["endOffset"] == 32

    def test_score_is_0_to_1(self, sample_report):
        sources = ["Some source text."]
        result = _to_halloumi_format(sample_report, sources)

        for claim in result["claims"]:
            assert 0 <= claim["score"] <= 1

    def test_score_mapped_from_confidence(self, sample_report):
        sources = ["Some source text."]
        result = _to_halloumi_format(sample_report, sources)

        # confidence 95 -> score 0.95
        assert result["claims"][0]["score"] == 0.95
        # confidence 90 -> score 0.9
        assert result["claims"][1]["score"] == 0.9

    def test_segments_have_offsets(self, sample_report):
        sources = [
            "Paris is the capital. More text here.",
            "Eiffel Tower built 1889. Additional context.",
        ]
        result = _to_halloumi_format(sample_report, sources)

        for _seg_id, seg in result["segments"].items():
            assert "startOffset" in seg
            assert "endOffset" in seg
            assert seg["startOffset"] < seg["endOffset"]

    def test_rationale_from_explanation(self, sample_report):
        sources = ["Some source text."]
        result = _to_halloumi_format(sample_report, sources)

        assert result["claims"][0]["rationale"] == "Document states this explicitly."

    def test_segment_ids_reference_segments(self, sample_report):
        sources = ["Some source text."]
        result = _to_halloumi_format(sample_report, sources)

        for claim in result["claims"]:
            for seg_id in claim["segmentIds"]:
                assert seg_id in result["segments"]

    def test_empty_report(self):
        report = CheckReport(
            answer="",
            overall_confidence=0.0,
            overall_verdict="no_claims",
        )
        result = _to_halloumi_format(report, [])
        assert result == {"claims": [], "segments": {}}

    def test_claim_without_span_skipped(self):
        """Claims without span info should be skipped."""
        report = CheckReport(
            answer="Some answer.",
            overall_confidence=50.0,
            overall_verdict="partially_supported",
            claims=[Claim(text="Some claim.", index=1, span=None)],
            results=[
                VerificationResult(
                    claim="Some claim.",
                    claim_index=1,
                    verdict="supported",
                    confidence=80,
                    evidence="Evidence.",
                    explanation="Explanation.",
                )
            ],
        )
        result = _to_halloumi_format(report, ["Some source."])
        assert result["claims"] == []
