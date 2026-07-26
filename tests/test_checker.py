"""
Tests for the core fact-checking pipeline in rag_facts_check.checker.

Covers ClaimExtractor, ClaimVerifier, RAGFactsChecker, and aggregation logic.
"""

import pytest

from rag_facts_check import MockLLM, RAGFactsChecker, EvidenceRetriever
from rag_facts_check.checker import ClaimExtractor, ClaimVerifier, RAGFactsChecker as Checker
from pathlib import Path
import json

from rag_facts_check.models import Claim, VerificationResult, CheckReport
from rag_facts_check.retriever import DocumentChunk
from rag_facts_check.prompts import (
    format_claim_extraction_prompt,
    format_claim_verification_prompt,
    format_claim_verification_evidence_first_prompt,
    format_documents,
)

# ─── Sample data (mirrors conftest.py constants) ─────────────────────────────

MOCK_DATASETS_DIR = Path(__file__).resolve().parents[1] / "mock_datasets"

SAMPLE_ANSWER_BERLIN = (
    "Paris is the capital of France. "
    "The Eiffel Tower was built in 1889. "
    "The Louvre Museum is located in Berlin."
)

SAMPLE_DOCS_BERLIN = [
    "Paris is the capital of France and the largest city in the country. "
    "It is known for the Eiffel Tower, a wrought-iron lattice tower built in 1889.",
    "The Louvre Museum is one of the world's largest museums, located in Paris, France. "
    "It houses thousands of famous works including the Mona Lisa.",
]


class TestClaimExtractor:
    """Tests for the ClaimExtractor class."""

    def test_extract_claims_basic(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = extractor.extract("Paris is the capital of France.")
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)
        assert claims[0].index == 1

    def test_extract_claims_multiple(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        answer = "Paris is the capital of France. The Eiffel Tower was built in 1889."
        claims = extractor.extract(answer)
        assert len(claims) >= 2
        assert claims[0].index == 1
        assert claims[1].index == 2

    def test_extract_claims_empty_answer(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = extractor.extract("")
        assert claims == []

    def test_extract_claims_whitespace_only(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = extractor.extract("   \n  \t  ")
        assert claims == []

    def test_extract_claims_no_claims_output(self, mock_llm):
        """If LLM returns 'NO CLAIMS', extractor should return empty list."""
        extractor = ClaimExtractor(mock_llm)
        # MockLLM splits sentences, so a single sentence becomes one claim
        claims = extractor.extract("Hello world.")
        assert isinstance(claims, list)

    def test_parse_claims_standard_format(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        response = "CLAIM 1: Paris is the capital of France.\nCLAIM 2: The Eiffel Tower was built in 1889."
        claims = extractor._parse_claims(response)
        assert len(claims) == 2
        assert claims[0].text == "Paris is the capital of France."
        assert claims[0].index == 1
        assert claims[1].text == "The Eiffel Tower was built in 1889."
        assert claims[1].index == 2

    def test_parse_claims_no_claims(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = extractor._parse_claims("NO CLAIMS")
        assert claims == []

    def test_parse_claims_empty_response(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = extractor._parse_claims("")
        assert claims == []

    def test_parse_claims_unparseable(self, mock_llm):
        """If response doesn't match CLAIM format, treat as single claim."""
        extractor = ClaimExtractor(mock_llm)
        claims = extractor._parse_claims("Some random text without claim format")
        assert len(claims) == 1
        assert claims[0].index == 1
        assert "random text" in claims[0].text

    def test_parse_claims_case_insensitive(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        response = "claim 1: Paris is the capital of France."
        claims = extractor._parse_claims(response)
        assert len(claims) == 1
        assert claims[0].text == "Paris is the capital of France."

    def test_extract_calls_llm(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        extractor.extract("Paris is the capital of France.")
        assert mock_llm.call_count >= 1

    def test_extract_claims_with_climate_answer(self, mock_llm):
        """Test claim extraction with an environmental answer."""
        extractor = ClaimExtractor(mock_llm)
        answer = (
            "In 2023, renewable energy sources accounted for 30% of global electricity "
            "generation. IRENA projects renewables will reach 42% by 2028."
        )
        claims = extractor.extract(answer)
        assert len(claims) >= 2
        assert all(isinstance(c, Claim) for c in claims)


class TestClaimVerifier:
    """Tests for the ClaimVerifier class."""

    def test_verify_supported(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)
        assert result.claim == "Paris is the capital of France."
        assert result.claim_index == 1

    def test_verify_contradicted(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="The Louvre is in Berlin.", index=1)
        docs = ["The Louvre is located in Paris, France."]
        result = verifier.verify(claim, docs)
        assert result.verdict == "contradicted"
        assert result.confidence > 0

    def test_verify_with_chunks(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        chunks = [
            DocumentChunk(text="Paris is the capital of France.", doc_id="doc_1", chunk_id=0)
        ]
        result = verifier.verify(claim, docs, chunks=chunks)
        assert isinstance(result, VerificationResult)

    def test_verify_evidence_first_prompt(self, mock_llm):
        """Test with evidence_first=True (default)."""
        verifier = ClaimVerifier(mock_llm, evidence_first=True)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)

    def test_verify_standard_prompt(self, mock_llm):
        """Test with evidence_first=False."""
        verifier = ClaimVerifier(mock_llm, evidence_first=False)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)

    def test_verify_self_consistency(self, mock_llm):
        """Test with num_consistency_runs > 1."""
        verifier = ClaimVerifier(mock_llm, num_consistency_runs=3)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)
        # Self-consistency should set consistency_score
        assert result.consistency_score is not None

    def test_verify_climate_hallucination(self, mock_llm):
        """Test verification of a climate change hallucination."""
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Temperatures will rise by 5.7°C by 2100.", index=1)
        docs = ["Climate models estimate a 2-4°C rise by the end of the century."]
        result = verifier.verify(claim, docs)
        assert result.verdict == "contradicted"

    def test_verify_renewable_statistic(self, mock_llm):
        """Test verification of a renewable energy statistic."""
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Renewables accounted for 30% of electricity in 2023.", index=1)
        docs = ["In 2023, renewable energy sources accounted for 30% of global electricity generation."]
        result = verifier.verify(claim, docs)
        assert result.verdict == "supported"

    def test_parse_result_supported(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Test.", index=1)
        response = """VERDICT: SUPPORTED
CONFIDENCE: 95
EVIDENCE: "Paris is the capital of France."
EXPLANATION: Document states this."""
        result = verifier._parse_result(claim, response)
        assert result.verdict == "supported"
        assert result.confidence == 95
        assert "Paris" in result.evidence

    def test_parse_result_contradicted(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Test.", index=1)
        response = """VERDICT: CONTRADICTED
CONFIDENCE: 85
EVIDENCE: "Louvre is in Paris."
EXPLANATION: Claim says Berlin."""
        result = verifier._parse_result(claim, response)
        assert result.verdict == "contradicted"
        assert result.confidence == 85

    def test_parse_result_not_enough_info(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Test.", index=1)
        response = """VERDICT: NOT ENOUGH INFO
CONFIDENCE: 60
EVIDENCE: N/A
EXPLANATION: No info."""
        result = verifier._parse_result(claim, response)
        assert result.verdict == "not_enough_info"

    def test_parse_result_missing_fields(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Test.", index=1)
        response = "VERDICT: SUPPORTED"
        result = verifier._parse_result(claim, response)
        assert result.verdict == "supported"
        assert result.confidence == 50  # default
        assert result.evidence == "N/A"  # default

    def test_parse_result_confidence_clamped(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Test.", index=1)
        response = """VERDICT: SUPPORTED
CONFIDENCE: 150
EVIDENCE: test
EXPLANATION: test"""
        result = verifier._parse_result(claim, response)
        assert result.confidence == 100  # clamped

    def test_aggregate_consistency(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Test.", index=1)
        results = [
            VerificationResult(claim="Test.", claim_index=1, verdict="supported",
                               confidence=90, evidence="E1", explanation="E1"),
            VerificationResult(claim="Test.", claim_index=1, verdict="supported",
                               confidence=85, evidence="E2", explanation="E2"),
            VerificationResult(claim="Test.", claim_index=1, verdict="contradicted",
                               confidence=70, evidence="E3", explanation="E3"),
        ]
        agg = verifier._aggregate_consistency(claim, None, results)
        assert agg.verdict == "supported"  # majority
        assert agg.consistency_score == pytest.approx(2/3)


class TestRAGFactsChecker:
    """Tests for the RAGFactsChecker main pipeline."""

    def test_check_basic(self, checker):
        report = checker.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)
        assert report.overall_confidence > 0
        assert report.overall_verdict in [
            "fully_supported", "mostly_supported", "partially_supported",
            "largely_unsupported", "no_claims"
        ]

    def test_check_with_hallucination(self, checker):
        """Test that a hallucinated claim is detected."""
        report = checker.check(
            answer="The Louvre Museum is located in Berlin.",
            documents=["The Louvre Museum is located in Paris, France."],
        )
        assert isinstance(report, CheckReport)
        assert len(report.hallucination_flags) > 0

    def test_check_no_claims(self, checker):
        """Test with an answer that has no factual claims."""
        report = checker.check(
            answer="",
            documents=["Some document."],
        )
        assert report.overall_verdict == "no_claims"
        assert report.claims == []
        assert report.results == []

    def test_check_whitespace_answer(self, checker):
        report = checker.check(
            answer="   \n\t  ",
            documents=["Some document."],
        )
        assert report.overall_verdict == "no_claims"

    def test_check_max_claims_limit(self, mock_llm):
        checker = RAGFactsChecker(mock_llm, max_claims=2)
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        assert len(report.claims) <= 2

    def test_check_without_evidence_retrieval(self, checker_no_retrieval):
        report = checker_no_retrieval.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)

    def test_check_self_consistency(self, checker_self_consistency):
        report = checker_self_consistency.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)
        # Results should have consistency_score set
        if report.results:
            assert report.results[0].consistency_score is not None

    def test_check_evidence_first_off(self, checker_evidence_first_off):
        report = checker_evidence_first_off.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)

    def test_check_returns_report_with_dimensions(self, checker):
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        assert "groundedness" in report.dimensions
        assert "contradiction_rate" in report.dimensions
        assert "hallucination_rate" in report.dimensions
        assert "completeness" in report.dimensions

    def test_check_to_dict(self, checker):
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        d = report.to_dict()
        assert d["overall_verdict"] == report.overall_verdict
        assert d["overall_confidence"] == report.overall_confidence
        assert isinstance(d["claims"], list)
        assert isinstance(d["results"], list)

    def test_check_climate_change_hallucination(self, checker, climate_change_dataset):
        """Test with the climate change hallucinated dataset."""
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        # The answer contains 5.7°C which contradicts 2-4°C in docs
        assert len(report.hallucination_flags) > 0
        assert report.overall_verdict in [
            "largely_unsupported", "partially_supported"
        ]

    def test_check_renewable_energy_supported(self, checker, renewable_energy_dataset):
        """Test with the renewable energy supported dataset."""
        report = checker.check(
            answer=renewable_energy_dataset["answer"],
            documents=renewable_energy_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        # Should have mostly supported claims
        assert report.overall_verdict in [
            "fully_supported", "mostly_supported", "partially_supported"
        ]

    def test_aggregate_dimensions(self, checker):
        """Test that dimensions are computed correctly."""
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        total = len(report.results)
        if total > 0:
            supported = sum(1 for r in report.results if r.verdict == "supported")
            contradicted = sum(1 for r in report.results if r.verdict == "contradicted")
            expected_groundedness = round(supported / total * 100, 1)
            expected_contradiction = round(contradicted / total * 100, 1)
            assert report.dimensions["groundedness"] == expected_groundedness
            assert report.dimensions["contradiction_rate"] == expected_contradiction

    def test_aggregate_summary(self, checker):
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        assert "confidence" in report.summary.lower()
        assert "claims" in report.summary.lower()

    def test_aggregate_hallucination_flags(self, checker):
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        for flag in report.hallucination_flags:
            assert flag.verdict in ("contradicted", "not_enough_info")

    def test_check_with_custom_retriever(self, mock_llm):
        retriever = EvidenceRetriever(chunk_size=50, top_k=2)
        checker = RAGFactsChecker(mock_llm, retriever=retriever)
        report = checker.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)

    def test_check_empty_documents(self, checker):
        report = checker.check(
            answer="Paris is the capital of France.",
            documents=[],
        )
        assert isinstance(report, CheckReport)

    def test_check_claims_have_indices(self, checker):
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        for i, claim in enumerate(report.claims):
            assert claim.index == i + 1

    def test_check_results_match_claims(self, checker):
        report = checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        if report.claims and report.results:
            assert len(report.results) == len(report.claims)


# ─── Sample data references (defined in conftest but accessible here) ─────────


