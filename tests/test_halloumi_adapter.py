"""Tests for the halloumi response adapter."""

import pytest

from rag_facts_check.models import CheckReport, Claim, EvidenceSpan, Span, VerificationResult
from rag_facts_check.server import (
    _context_quality,
    _find_source_index,
    _to_halloumi_format,
)


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
                evidence=["Paris is the capital."],
                explanation="Document states this explicitly.",
                evidence_spans=[
                    EvidenceSpan(quote="Paris is the capital.", start=0, end=22, document_index=0)
                ],
            ),
            VerificationResult(
                claim="The Eiffel Tower was built in 1889.",
                claim_index=2,
                verdict="supported",
                confidence=90,
                evidence=["Eiffel Tower built 1889."],
                explanation="Document confirms this.",
                evidence_spans=[
                    EvidenceSpan(
                        quote="Eiffel Tower built 1889.", start=12, end=37, document_index=1
                    )
                ],
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

    def test_score_is_verdict_based(self, sample_report):
        sources = ["Some source text."]
        result = _to_halloumi_format(sample_report, sources)

        # Both claims are "supported" → score 1.0 (not raw confidence)
        assert result["claims"][0]["score"] == 1.0
        assert result["claims"][1]["score"] == 1.0

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
        assert result["answer_score"] == 0.0
        assert result["claims"] == []
        assert result["segments"] == {}
        assert result["context_quality"]["level"] == "none"

    def test_claim_without_span_uses_full_answer_range(self):
        """Claims without span info (LLM paraphrased) should use the full answer range."""
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
        result = _to_halloumi_format(report, ["Some source."], "Some answer.")
        assert len(result["claims"]) == 1
        assert result["claims"][0]["startOffset"] == 0
        assert result["claims"][0]["endOffset"] == 12  # len("Some answer.")

    def test_verdict_scores_contradicted_and_nei(self):
        """Verdict-based scores: supported=1.0, nei=0.4, contradicted=0.0."""
        report = CheckReport(
            answer="Some answer.",
            overall_confidence=50.0,
            overall_verdict="partially_supported",
            claims=[
                Claim(text="Claim 1.", index=1, span=Span(start=0, end=7)),
                Claim(text="Claim 2.", index=2, span=Span(start=8, end=15)),
                Claim(text="Claim 3.", index=3, span=Span(start=16, end=23)),
            ],
            results=[
                VerificationResult(
                    claim="Claim 1.",
                    claim_index=1,
                    verdict="supported",
                    confidence=90,
                    evidence="Evidence.",
                    explanation="Found.",
                ),
                VerificationResult(
                    claim="Claim 2.",
                    claim_index=2,
                    verdict="not_enough_info",
                    confidence=60,
                    evidence="N/A",
                    explanation="No info.",
                ),
                VerificationResult(
                    claim="Claim 3.",
                    claim_index=3,
                    verdict="contradicted",
                    confidence=85,
                    evidence="Contradiction.",
                    explanation="Wrong.",
                ),
            ],
        )
        result = _to_halloumi_format(report, ["Some source."], "Some answer.")
        assert result["claims"][0]["score"] == 1.0  # supported
        assert result["claims"][1]["score"] == 0.4  # not_enough_info
        assert result["claims"][2]["score"] == 0.0  # contradicted

    def test_answer_score_in_response(self, sample_report):
        """answer_score from CheckReport appears in halloumi response."""
        sample_report.answer_score = 7.5
        result = _to_halloumi_format(sample_report, ["Some source text."])
        assert result["answer_score"] == 7.5

    def test_claim_without_span_is_marked_skipped(self):
        """Claims without span (LLM paraphrased) are marked skipped=True."""
        report = CheckReport(
            answer="Some answer.",
            overall_confidence=50.0,
            overall_verdict="partially_supported",
            claims=[
                Claim(text="Has span.", index=1, span=Span(start=0, end=9)),
                Claim(text="No span.", index=2, span=None),
            ],
            results=[
                VerificationResult(
                    claim="Has span.",
                    claim_index=1,
                    verdict="supported",
                    confidence=90,
                    evidence="Evidence.",
                    explanation="Found.",
                ),
                VerificationResult(
                    claim="No span.",
                    claim_index=2,
                    verdict="not_enough_info",
                    confidence=60,
                    evidence="N/A",
                    explanation="No info.",
                ),
            ],
        )
        result = _to_halloumi_format(report, ["Some source."], "Some answer.")
        assert result["claims"][0]["skipped"] is False
        assert result["claims"][1]["skipped"] is True

    def test_to_halloumi_uses_document_index_mapping(self):
        """When document_index is set, segments map to the correct source offset."""
        sources = [
            "Source 1 has 20 chars.",  # len = 22
            "Source 2 has the actual evidence quote right here.",  # len = 50
        ]
        report = CheckReport(
            answer="Answer statement.",
            claims=[Claim(text="Answer statement.", index=1, span=Span(0, 17))],
            results=[
                VerificationResult(
                    claim="Answer statement.",
                    claim_index=1,
                    verdict="supported",
                    confidence=95,
                    evidence=["actual evidence quote"],
                    explanation="Found in source 2.",
                    document_index=1,
                    evidence_spans=[
                        # offset inside source 2
                        EvidenceSpan(
                            quote="actual evidence quote", start=17, end=38, document_index=1
                        )
                    ],
                )
            ],
        )
        result = _to_halloumi_format(report, sources, "Answer statement.")
        assert len(result["segments"]) == 1
        seg = result["segments"]["0"]
        # Expected: offset of source 2 (22) + span.start (17) = 39, span.end (38) + 22 = 60
        assert seg["startOffset"] == 22 + 17
        assert seg["endOffset"] == 22 + 38
        assert result["claims"][0]["segmentIds"] == ["0"]

    def test_to_halloumi_computes_missing_evidence_span_from_evidence_text(self):
        sources = [
            "First doc text without match.",
            "The European Climate Law mandates net-zero by 2050.",
        ]
        report = CheckReport(
            answer="Net-zero by 2050.",
            claims=[Claim(text="Net-zero by 2050.", index=1, span=Span(0, 17))],
            results=[
                VerificationResult(
                    claim="Net-zero by 2050.",
                    claim_index=1,
                    verdict="supported",
                    confidence=95,
                    evidence=["European Climate Law mandates net-zero by 2050"],
                    explanation="Matched via text fallback.",
                    document_index=None,
                    evidence_spans=[],  # spans never located
                )
            ],
        )
        result = _to_halloumi_format(report, sources, "Net-zero by 2050.")
        assert len(result["segments"]) == 1
        assert result["claims"][0]["segmentIds"] == ["0"]


class TestMultipleEvidenceSegments:
    """A claim may be supported by several evidence quotes → several segments."""

    ANSWER = "A claim that needs several passages."

    @classmethod
    def _report(cls, spans, claim_index=1):
        return CheckReport(
            answer=cls.ANSWER,
            claims=[Claim(text=cls.ANSWER, index=claim_index, span=Span(0, 34))],
            results=[
                VerificationResult(
                    claim="A claim that needs several passages.",
                    claim_index=claim_index,
                    verdict="supported",
                    confidence=0,
                    evidence=[s.quote for s in spans],
                    explanation="Two passages support this.",
                    evidence_spans=spans,
                )
            ],
        )

    def test_multiple_evidences_produce_multiple_segments(self):
        sources = ["Alpha beta gamma delta epsilon zeta eta theta."]
        spans = [
            EvidenceSpan(quote="Alpha beta", start=0, end=10, document_index=0),
            EvidenceSpan(quote="eta theta", start=29, end=38, document_index=0),
        ]
        result = _to_halloumi_format(self._report(spans), sources, self.ANSWER)

        assert len(result["segments"]) == 2
        assert result["claims"][0]["segmentIds"] == ["0", "1"]
        offsets = sorted((s["startOffset"], s["endOffset"]) for s in result["segments"].values())
        assert offsets == [(0, 10), (29, 38)]

    def test_evidences_from_different_sources_use_source_offsets(self):
        sources = ["First source has alpha.", "Second source has beta."]
        spans = [
            EvidenceSpan(quote="alpha", start=17, end=22, document_index=0),
            EvidenceSpan(quote="beta", start=17, end=21, document_index=1),
        ]
        result = _to_halloumi_format(self._report(spans), sources, self.ANSWER)

        offsets = sorted((s["startOffset"], s["endOffset"]) for s in result["segments"].values())
        # second source starts at offset len(sources[0]) = 23 in the joined string
        assert offsets == [(17, 22), (23 + 17, 23 + 21)]

    def test_overlapping_evidences_merge_into_one_segment(self):
        """Overlapping quotes would duplicate text in the frontend renderer."""
        sources = ["Alpha beta gamma delta epsilon zeta eta theta."]
        spans = [
            EvidenceSpan(quote="Alpha beta gamma", start=0, end=16, document_index=0),
            EvidenceSpan(quote="beta gamma delta", start=6, end=22, document_index=0),
        ]
        result = _to_halloumi_format(self._report(spans), sources, self.ANSWER)

        assert len(result["segments"]) == 1
        seg = result["segments"][result["claims"][0]["segmentIds"][0]]
        assert (seg["startOffset"], seg["endOffset"]) == (0, 22)

    def test_identical_span_shared_between_claims(self):
        """Two claims citing the same passage render one chip, not two."""
        sources = ["Alpha beta gamma delta."]
        shared = [EvidenceSpan(quote="Alpha beta", start=0, end=10, document_index=0)]
        report = CheckReport(
            answer="Claim one. Claim two.",
            claims=[
                Claim(text="Claim one.", index=1, span=Span(0, 10)),
                Claim(text="Claim two.", index=2, span=Span(11, 21)),
            ],
            results=[
                VerificationResult(
                    claim="Claim one.",
                    claim_index=1,
                    verdict="supported",
                    confidence=0,
                    evidence=[shared[0].quote],
                    evidence_spans=shared,
                    explanation="Same passage.",
                ),
                VerificationResult(
                    claim="Claim two.",
                    claim_index=2,
                    verdict="supported",
                    confidence=0,
                    evidence=[shared[0].quote],
                    evidence_spans=shared,
                    explanation="Same passage.",
                ),
            ],
        )
        result = _to_halloumi_format(report, sources, "Claim one. Claim two.")

        assert len(result["segments"]) == 1
        assert result["claims"][0]["segmentIds"] == result["claims"][1]["segmentIds"]

    def test_unlocatable_evidence_yields_no_segments(self):
        sources = ["Nothing relevant here."]
        spans = [EvidenceSpan(quote="alpha", start=100, end=120, document_index=7)]
        result = _to_halloumi_format(self._report(spans), sources, self.ANSWER)

        assert result["segments"] == {}
        assert result["claims"][0]["segmentIds"] == []


class TestFindSourceIndex:
    """Tests for _find_source_index."""

    def test_find_source_index_basic(self):
        sources = ["Source A text.", "Source B contains the target evidence.", "Source C."]
        assert _find_source_index("target evidence", sources) == 1

    def test_find_source_index_with_newlines_in_source(self):
        sources = [
            "Source A.",
            "The EU is largely on track\n to meet the agreed targets \nfor 2030.",
        ]
        evidence = "The EU is largely on track to meet the agreed targets for 2030"
        assert _find_source_index(evidence, sources) == 1

    def test_find_source_index_empty_and_na(self):
        sources = ["Source text."]
        assert _find_source_index("", sources) is None
        assert _find_source_index("N/A", sources) is None

    def test_find_source_index_not_found(self):
        sources = ["Source A.", "Source B."]
        assert _find_source_index("completely absent evidence quote", sources) is None


def _supported_and_nei_report():
    """One supported claim and one 'not enough information' claim."""
    return CheckReport(
        answer="Claim one. Claim two.",
        overall_confidence=50.0,
        overall_verdict="partially_supported",
        claims=[
            Claim(text="Claim one.", index=1, span=Span(start=0, end=10)),
            Claim(text="Claim two.", index=2, span=Span(start=11, end=21)),
        ],
        results=[
            VerificationResult(
                claim="Claim one.",
                claim_index=1,
                verdict="supported",
                confidence=90,
                evidence="Source text.",
                explanation="Found.",
            ),
            VerificationResult(
                claim="Claim two.",
                claim_index=2,
                verdict="not_enough_info",
                confidence=0,
                evidence="N/A",
                explanation="No info.",
            ),
        ],
    )


class TestContextQuality:
    """The ``kind`` a client sends must surface as an honest context label.

    A snippet is a ~600-character search blurb. A claim missing from a blurb proves
    nothing, so a low score computed over snippets has to be labelled as partial
    context instead of read as a hallucination verdict.
    """

    def test_all_chunk_sources_are_full_context(self):
        quality = _context_quality(["chunk", "chunk"])
        assert quality["level"] == "full"
        assert quality["chunk_sources"] == 2
        assert quality["note"] is None

    def test_any_snippet_makes_context_partial(self):
        quality = _context_quality(["chunk", "snippet", "snippet"])
        assert quality["level"] == "partial"
        assert quality["chunk_sources"] == 1
        assert quality["snippet_sources"] == 2
        assert "2 of 3" in quality["note"]

    def test_undeclared_kinds_are_unknown(self):
        quality = _context_quality([None, "chunk"])
        assert quality["level"] == "unknown"
        assert quality["unknown_sources"] == 1

    def test_no_sources_reports_none(self):
        quality = _context_quality([])
        assert quality["level"] == "none"
        assert quality["sources"] == 0

    def test_response_carries_context_quality(self):
        result = _to_halloumi_format(
            _supported_and_nei_report(),
            ["Source text."],
            "Claim one. Claim two.",
            ["chunk", "snippet"],
        )
        assert result["context_quality"]["level"] == "partial"

    def test_response_without_kinds_is_unknown(self):
        result = _to_halloumi_format(
            _supported_and_nei_report(), ["Source text."], "Claim one. Claim two."
        )
        assert result["context_quality"]["level"] == "unknown"

    def test_not_enough_info_is_context_limited_over_snippets(self):
        result = _to_halloumi_format(
            _supported_and_nei_report(),
            ["Source text."],
            "Claim one. Claim two.",
            ["snippet"],
        )
        assert "context_limited" not in result["claims"][0]  # supported
        assert result["claims"][1]["context_limited"] is True  # not_enough_info

    def test_full_context_does_not_flag_claims(self):
        result = _to_halloumi_format(
            _supported_and_nei_report(),
            ["Source text."],
            "Claim one. Claim two.",
            ["chunk", "chunk"],
        )
        assert all("context_limited" not in claim for claim in result["claims"])

    def test_unknown_context_does_not_flag_claims(self):
        """Unknown means an older client, not known-thin evidence."""
        result = _to_halloumi_format(
            _supported_and_nei_report(),
            ["Source text."],
            "Claim one. Claim two.",
            [None],
        )
        assert all("context_limited" not in claim for claim in result["claims"])
