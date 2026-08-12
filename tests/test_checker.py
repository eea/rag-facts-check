"""
Tests for the core fact-checking pipeline in rag_facts_check.checker.

Covers ClaimExtractor, ClaimVerifier, RAGFactsChecker, and aggregation logic.
"""

import pytest

from rag_facts_check import EvidenceRetriever, RAGFactsChecker
from rag_facts_check.checker import ClaimExtractor, ClaimVerifier
from rag_facts_check.models import CheckReport, Claim, Span, VerificationResult
from rag_facts_check.retriever import DocumentChunk

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

    async def test_extract_claims_basic(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = await extractor.extract("Paris is the capital of France.")
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)
        assert claims[0].index == 1

    async def test_extract_claims_multiple(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        answer = "Paris is the capital of France. The Eiffel Tower was built in 1889."
        claims = await extractor.extract(answer)
        assert len(claims) >= 2
        assert claims[0].index == 1
        assert claims[1].index == 2

    async def test_extract_claims_empty_answer(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = await extractor.extract("")
        assert claims == []

    async def test_extract_claims_whitespace_only(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        claims = await extractor.extract("   \n  \t  ")
        assert claims == []

    async def test_extract_claims_no_claims_output(self, mock_llm):
        """If LLM returns 'NO CLAIMS', extractor should return empty list."""
        extractor = ClaimExtractor(mock_llm)
        claims = await extractor.extract("Hello world.")
        assert isinstance(claims, list)

    # ── Parser tests (no LLM needed) ──

    def test_parse_claims_standard_format(self):
        extractor = ClaimExtractor.__new__(ClaimExtractor)
        response = (
            "CLAIM 1: Paris is the capital of France.\n"
            "CLAIM 2: The Eiffel Tower was built in 1889."
        )
        claims = extractor._parse_claims(response)
        assert len(claims) == 2
        assert claims[0].text == "Paris is the capital of France."
        assert claims[0].index == 1
        assert claims[1].text == "The Eiffel Tower was built in 1889."
        assert claims[1].index == 2

    def test_parse_claims_no_claims(self):
        extractor = ClaimExtractor.__new__(ClaimExtractor)
        claims = extractor._parse_claims("NO CLAIMS")
        assert claims == []

    def test_parse_claims_empty_response(self):
        extractor = ClaimExtractor.__new__(ClaimExtractor)
        claims = extractor._parse_claims("")
        assert claims == []

    def test_parse_claims_unparseable(self):
        """If response doesn't match CLAIM format, treat as single claim."""
        extractor = ClaimExtractor.__new__(ClaimExtractor)
        claims = extractor._parse_claims("Some random text without claim format")
        assert len(claims) == 1
        assert claims[0].index == 1
        assert "random text" in claims[0].text

    def test_parse_claims_case_insensitive(self):
        extractor = ClaimExtractor.__new__(ClaimExtractor)
        response = "claim 1: Paris is the capital of France."
        claims = extractor._parse_claims(response)
        assert len(claims) == 1
        assert claims[0].text == "Paris is the capital of France."

    async def test_extract_calls_llm(self, mock_llm):
        extractor = ClaimExtractor(mock_llm)
        await extractor.extract("Paris is the capital of France.")
        assert mock_llm.generate.call_count >= 1

    async def test_extract_claims_with_climate_answer(self, mock_llm):
        """Test claim extraction with an environmental answer."""
        extractor = ClaimExtractor(mock_llm)
        answer = (
            "In 2023, renewable energy sources accounted for 30% of global electricity "
            "generation. IRENA projects renewables will reach 42% by 2028."
        )
        claims = await extractor.extract(answer)
        assert len(claims) >= 2
        assert all(isinstance(c, Claim) for c in claims)


class TestClaimVerifier:
    """Tests for the ClaimVerifier class."""

    async def test_verify_supported(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = await verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)
        assert result.claim == "Paris is the capital of France."
        assert result.claim_index == 1

    async def test_verify_contradicted(self, mock_llm_contradicted):
        verifier = ClaimVerifier(mock_llm_contradicted)
        claim = Claim(text="The Louvre is in Berlin.", index=1)
        docs = ["The Louvre is located in Paris, France."]
        result = await verifier.verify(claim, docs)
        assert result.verdict == "contradicted"

    async def test_verify_with_chunks(self, mock_llm):
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        chunks = [DocumentChunk(text="Paris is the capital of France.", doc_id="doc_1", chunk_id=0)]
        result = await verifier.verify(claim, docs, chunks=chunks)
        assert isinstance(result, VerificationResult)

    async def test_verify_evidence_first_prompt(self, mock_llm):
        """Test with evidence_first=True (default)."""
        verifier = ClaimVerifier(mock_llm, evidence_first=True)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = await verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)

    async def test_verify_standard_prompt(self, mock_llm):
        """Test with evidence_first=False."""
        verifier = ClaimVerifier(mock_llm, evidence_first=False)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = await verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)

    async def test_verify_self_consistency(self, mock_llm):
        """Test with num_consistency_runs > 1."""
        verifier = ClaimVerifier(mock_llm, num_consistency_runs=3)
        claim = Claim(text="Paris is the capital of France.", index=1)
        docs = ["Paris is the capital of France."]
        result = await verifier.verify(claim, docs)
        assert isinstance(result, VerificationResult)
        assert result.consistency_score is not None

    async def test_verify_climate_hallucination(self, mock_llm_contradicted):
        """Test verification of a climate change hallucination."""
        verifier = ClaimVerifier(mock_llm_contradicted)
        claim = Claim(text="Temperatures will rise by 5.7°C by 2100.", index=1)
        docs = ["Climate models estimate a 2-4°C rise by the end of the century."]
        result = await verifier.verify(claim, docs)
        assert result.verdict == "contradicted"

    async def test_verify_renewable_statistic(self, mock_llm):
        """Test verification of a renewable energy statistic."""
        verifier = ClaimVerifier(mock_llm)
        claim = Claim(text="Renewables accounted for 30% of electricity in 2023.", index=1)
        docs = [
            "In 2023, renewable energy sources accounted for 30% of global electricity generation."
        ]
        result = await verifier.verify(claim, docs)
        assert result.verdict == "supported"

    # ── Parser tests (no LLM needed) ──

    def test_parse_result_supported(self):
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        claim = Claim(text="Test.", index=1)
        response = """VERDICT: SUPPORTED
CONFIDENCE: 95
EVIDENCE: "Paris is the capital of France."
EXPLANATION: Document states this."""
        result = verifier._parse_result(claim, response)
        assert result.verdict == "supported"
        assert result.confidence == 0  # confidence no longer parsed from LLM
        assert "Paris" in result.evidence

    def test_parse_result_contradicted(self):
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        claim = Claim(text="Test.", index=1)
        response = """VERDICT: CONTRADICTED
CONFIDENCE: 85
EVIDENCE: "Louvre is in Paris."
EXPLANATION: Claim says Berlin."""
        result = verifier._parse_result(claim, response)
        assert result.verdict == "contradicted"
        assert result.confidence == 0

    def test_parse_result_not_enough_info(self):
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        claim = Claim(text="Test.", index=1)
        response = """VERDICT: NOT ENOUGH INFO
CONFIDENCE: 60
EVIDENCE: N/A
EXPLANATION: No info."""
        result = verifier._parse_result(claim, response)
        assert result.verdict == "not_enough_info"

    def test_parse_json_result_with_document_index(self):
        """JSON parsing should extract document_index."""
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        claim = Claim(text="Test.", index=1)
        response = (
            '{"verdict": "SUPPORTED", "confidence": 90, '
            '"evidence": "Paris is the capital.", '
            '"document_index": 1, '
            '"explanation": "Found in doc 2."}'
        )
        result = verifier._parse_result(claim, response)
        assert result.verdict == "supported"
        assert result.document_index == 1

    def test_parse_json_result_without_document_index(self):
        """JSON without document_index should default to None."""
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        claim = Claim(text="Test.", index=1)
        response = (
            '{"verdict": "SUPPORTED", "confidence": 90, '
            '"evidence": "Evidence.", "explanation": "Found it."}'
        )
        result = verifier._parse_result(claim, response)
        assert result.document_index is None

    def test_parse_result_missing_fields(self):
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        claim = Claim(text="Test.", index=1)
        response = "VERDICT: SUPPORTED"
        result = verifier._parse_result(claim, response)
        assert result.verdict == "supported"
        assert result.confidence == 0
        assert result.evidence == "N/A"

    def test_aggregate_consistency(self):
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        claim = Claim(text="Test.", index=1)
        results = [
            VerificationResult(
                claim="Test.", claim_index=1, verdict="supported",
                confidence=90, evidence="E1", explanation="E1",
            ),
            VerificationResult(
                claim="Test.", claim_index=1, verdict="supported",
                confidence=85, evidence="E2", explanation="E2",
            ),
            VerificationResult(
                claim="Test.", claim_index=1, verdict="contradicted",
                confidence=70, evidence="E3", explanation="E3",
            ),
        ]
        agg = verifier._aggregate_consistency(claim, None, results)
        assert agg.verdict == "supported"  # majority
        assert agg.consistency_score == pytest.approx(2 / 3)


class TestRAGFactsChecker:
    """Tests for the RAGFactsChecker main pipeline."""

    # ── Evidence span tests (no LLM needed) ──

    def test_find_evidence_span_targeted_match(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        docs = ["First document.", "Second document with evidence."]
        result = VerificationResult(
            claim="Test.", claim_index=1, verdict="supported",
            confidence=90, evidence="with evidence", explanation="Found it.",
            document_index=1,
        )
        span = checker._find_evidence_span(result, docs, None)
        assert span is not None
        assert docs[1][span.start : span.end] == "with evidence"

    def test_find_evidence_span_fallback_all_docs(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        docs = ["First document with evidence.", "Second document."]
        result = VerificationResult(
            claim="Test.", claim_index=1, verdict="supported",
            confidence=90, evidence="with evidence", explanation="Found it.",
            document_index=1,  # Wrong! Evidence is in doc 0
        )
        span = checker._find_evidence_span(result, docs, None)
        assert span is not None
        assert docs[0][span.start : span.end] == "with evidence"

    def test_find_evidence_span_not_found_returns_none(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        docs = ["Some document text that doesn't match the evidence."]
        result = VerificationResult(
            claim="Test.", claim_index=1, verdict="supported",
            confidence=90, evidence="Paraphrased evidence that won't match",
            explanation="Found it.", document_index=0,
        )
        chunks = [
            DocumentChunk(
                text="Some document text", doc_id="doc_1",
                doc_index=0, chunk_id=0, start=0, end=18,
            )
        ]
        span = checker._find_evidence_span(result, docs, chunks)
        assert span is None

    def test_find_evidence_span_na_evidence(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        docs = ["Some document."]
        result = VerificationResult(
            claim="Test.", claim_index=1, verdict="not_enough_info",
            confidence=60, evidence="N/A", explanation="No info.",
        )
        span = checker._find_evidence_span(result, docs, None)
        assert span is None

    # ── Full-pipeline tests (use mock_llm fixture) ──

    async def test_check_basic(self, checker):
        report = await checker.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)
        assert report.overall_confidence > 0
        assert report.overall_verdict in [
            "fully_supported", "mostly_supported",
            "partially_supported", "largely_unsupported", "no_claims",
        ]

    async def test_check_with_hallucination(self, checker):
        report = await checker.check(
            answer="The Louvre Museum is located in Berlin.",
            documents=["The Louvre Museum is located in Paris, France."],
        )
        assert isinstance(report, CheckReport)
        # With default mock (all supported), no hallucination flags
        # This tests the pipeline runs end-to-end, not specific verdicts

    async def test_check_no_claims(self, checker):
        report = await checker.check(answer="", documents=["Some document."])
        assert report.overall_verdict == "no_claims"
        assert report.claims == []
        assert report.results == []

    async def test_check_whitespace_answer(self, checker):
        report = await checker.check(answer="   \n\t  ", documents=["Some document."])
        assert report.overall_verdict == "no_claims"

    async def test_check_max_claims_limit(self, mock_llm):
        checker = RAGFactsChecker(mock_llm, max_claims=2)
        report = await checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        assert len(report.claims) <= 2

    async def test_check_without_evidence_retrieval(self, checker_no_retrieval):
        report = await checker_no_retrieval.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)

    async def test_check_self_consistency(self, checker_self_consistency):
        report = await checker_self_consistency.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)
        if report.results:
            assert report.results[0].consistency_score is not None

    async def test_check_evidence_first_off(self, checker_evidence_first_off):
        report = await checker_evidence_first_off.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)

    async def test_check_returns_report_with_dimensions(self, checker):
        report = await checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        assert "groundedness" in report.dimensions
        assert "contradiction_rate" in report.dimensions
        assert "hallucination_rate" in report.dimensions
        assert "completeness" in report.dimensions

    async def test_check_to_dict(self, checker):
        report = await checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        d = report.to_dict()
        assert d["overall_verdict"] == report.overall_verdict
        assert d["overall_confidence"] == pytest.approx(report.overall_confidence, abs=0.01)
        assert isinstance(d["claims"], list)
        assert isinstance(d["results"], list)

    async def test_check_climate_change_hallucination(self, checker, climate_change_dataset):
        """Test with the climate change hallucinated dataset."""
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)

    async def test_check_renewable_energy_supported(self, checker, renewable_energy_dataset):
        """Test with the renewable energy supported dataset."""
        report = await checker.check(
            answer=renewable_energy_dataset["answer"],
            documents=renewable_energy_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        assert report.overall_verdict in [
            "fully_supported", "mostly_supported", "partially_supported",
        ]

    async def test_aggregate_dimensions(self, checker):
        report = await checker.check(
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

    async def test_aggregate_summary(self, checker):
        report = await checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        assert "confidence" in report.summary.lower()
        assert "claims" in report.summary.lower()

    async def test_aggregate_hallucination_flags(self, checker):
        report = await checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        for flag in report.hallucination_flags:
            assert flag.verdict in ("contradicted", "not_enough_info")

    async def test_check_with_custom_retriever(self, mock_llm):
        retriever = EvidenceRetriever(chunk_size=50, top_k=2)
        checker = RAGFactsChecker(mock_llm, retriever=retriever)
        report = await checker.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert isinstance(report, CheckReport)

    async def test_check_empty_documents(self, checker):
        report = await checker.check(
            answer="Paris is the capital of France.",
            documents=[],
        )
        assert isinstance(report, CheckReport)

    async def test_check_claims_have_indices(self, checker):
        report = await checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        for i, claim in enumerate(report.claims):
            assert claim.index == i + 1

    async def test_check_results_match_claims(self, checker):
        report = await checker.check(
            answer=SAMPLE_ANSWER_BERLIN,
            documents=SAMPLE_DOCS_BERLIN,
        )
        if report.claims and report.results:
            assert len(report.results) == len(report.claims)

    # ── Answer quality score tests (no LLM needed) ──

    def test_answer_score_all_supported_all_cited(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        results = [
            VerificationResult(
                claim="Claim 1.", claim_index=1, verdict="supported",
                confidence=95, evidence="Evidence 1.", explanation="Found it.",
                evidence_span=Span(start=0, end=10),
            ),
            VerificationResult(
                claim="Claim 2.", claim_index=2, verdict="supported",
                confidence=90, evidence="Evidence 2.", explanation="Found it.",
                evidence_span=Span(start=20, end=30),
            ),
        ]
        score = checker._compute_answer_score(results)
        assert 8.0 <= score <= 10.0

    def test_answer_score_with_contradictions(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        results = [
            VerificationResult(
                claim="Claim 1.", claim_index=1, verdict="supported",
                confidence=90, evidence="Evidence.", explanation="Found.",
                evidence_span=Span(start=0, end=10),
            ),
            VerificationResult(
                claim="Claim 2.", claim_index=2, verdict="contradicted",
                confidence=80, evidence="Contradiction.", explanation="Wrong.",
                evidence_span=Span(start=20, end=30),
            ),
        ]
        score = checker._compute_answer_score(results)
        assert score < 3.0

    def test_answer_score_uncited_penalty(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        results = [
            VerificationResult(
                claim="Claim 1.", claim_index=1, verdict="supported",
                confidence=95, evidence="Paraphrased evidence", explanation="Found it.",
            ),
            VerificationResult(
                claim="Claim 2.", claim_index=2, verdict="supported",
                confidence=90, evidence="Paraphrased too", explanation="Found it.",
            ),
        ]
        score = checker._compute_answer_score(results)
        assert score == 7.0

    def test_answer_score_not_enough_info(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        results = [
            VerificationResult(
                claim="Claim 1.", claim_index=1, verdict="supported",
                confidence=90, evidence="Evidence.", explanation="Found.",
                evidence_span=Span(start=0, end=10),
            ),
            VerificationResult(
                claim="Claim 2.", claim_index=2, verdict="not_enough_info",
                confidence=60, evidence="N/A", explanation="No info.",
            ),
        ]
        score = checker._compute_answer_score(results)
        assert 4.0 < score < 8.0

    def test_answer_score_empty_results(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        score = checker._compute_answer_score([])
        assert score == 0.0

    async def test_answer_score_in_report(self, checker):
        report = await checker.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        assert 0 <= report.answer_score <= 10
        assert report.to_dict()["answer_score"] == report.answer_score

    def test_answer_score_severe_contradictions_floor_to_zero(self):
        checker = RAGFactsChecker.__new__(RAGFactsChecker)
        results = [
            VerificationResult(
                claim="Claim 1.", claim_index=1, verdict="supported",
                confidence=90, evidence="Evidence.", explanation="Found.",
                evidence_span=Span(start=0, end=10),
            ),
            VerificationResult(
                claim="Claim 2.", claim_index=2, verdict="contradicted",
                confidence=80, evidence="Contradiction.", explanation="Wrong.",
                evidence_span=Span(start=20, end=30),
            ),
            VerificationResult(
                claim="Claim 3.", claim_index=3, verdict="contradicted",
                confidence=85, evidence="Another contradiction.", explanation="Wrong again.",
                evidence_span=Span(start=40, end=50),
            ),
        ]
        score = checker._compute_answer_score(results)
        assert score == 0.0
