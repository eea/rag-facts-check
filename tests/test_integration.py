"""
End-to-end integration tests for the RAG fact-checking pipeline.

Tests the full pipeline (claim extraction → evidence retrieval →
verification → aggregation) using the mock datasets and MockLLM.
"""

import json

import pytest

from rag_facts_check import MockLLM, RAGFactsChecker, EvidenceRetriever
from rag_facts_check.models import CheckReport


class TestEndToEnd:
    """End-to-end tests with the full pipeline."""

    def test_full_pipeline_supported(self, mock_llm):
        """Test the full pipeline with a supported answer."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer="Paris is the capital of France. The Eiffel Tower was built in 1889.",
            documents=["Paris is the capital of France. The Eiffel Tower was built in 1889."],
        )
        assert isinstance(report, CheckReport)
        assert report.overall_confidence > 0
        assert len(report.claims) >= 1
        assert len(report.results) == len(report.claims)

    def test_full_pipeline_contradicted(self, mock_llm):
        """Test the full pipeline with a contradicted answer."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer="The Louvre Museum is located in Berlin.",
            documents=["The Louvre Museum is located in Paris, France."],
        )
        assert isinstance(report, CheckReport)
        assert len(report.hallucination_flags) > 0

    def test_full_pipeline_with_climate_dataset(self, mock_llm, climate_change_dataset):
        """Test with the climate change hallucinated dataset."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        assert len(report.claims) >= 3
        assert len(report.results) == len(report.claims)
        assert report.dimensions["hallucination_rate"] > 0

    def test_full_pipeline_with_renewable_dataset(self, mock_llm, renewable_energy_dataset):
        """Test with the renewable energy supported dataset."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=renewable_energy_dataset["answer"],
            documents=renewable_energy_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        assert len(report.claims) >= 3
        assert len(report.results) == len(report.claims)
        assert report.dimensions["groundedness"] > 0

    def test_climate_change_flags_hallucination(self, mock_llm, climate_change_dataset):
        """The climate change answer should be flagged for hallucination."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        # 5.7°C contradicts 2-4°C
        assert len(report.hallucination_flags) > 0
        assert report.overall_verdict in [
            "largely_unsupported", "partially_supported"
        ]

    def test_renewable_energy_no_hallucination(self, mock_llm, renewable_energy_dataset):
        """The renewable energy answer should have minimal hallucinations."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=renewable_energy_dataset["answer"],
            documents=renewable_energy_dataset["documents"],
        )
        # Most claims should be supported
        assert report.dimensions["groundedness"] >= 50.0

    def test_pipeline_with_self_consistency(self, mock_llm, climate_change_dataset):
        """Test the full pipeline with self-consistency enabled."""
        checker = RAGFactsChecker(mock_llm, num_consistency_runs=3)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        if report.results:
            assert report.results[0].consistency_score is not None

    def test_pipeline_no_evidence_retrieval(self, mock_llm, climate_change_dataset):
        """Test the full pipeline without evidence retrieval."""
        checker = RAGFactsChecker(mock_llm, use_evidence_retrieval=False)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)

    def test_pipeline_evidence_first_off(self, mock_llm, climate_change_dataset):
        """Test the full pipeline with evidence-first prompting disabled."""
        checker = RAGFactsChecker(mock_llm, evidence_first=False)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)

    def test_pipeline_custom_retriever(self, mock_llm, climate_change_dataset):
        """Test the full pipeline with a custom retriever."""
        retriever = EvidenceRetriever(chunk_size=50, top_k=2)
        checker = RAGFactsChecker(mock_llm, retriever=retriever)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)

    def test_pipeline_max_claims(self, mock_llm, climate_change_dataset):
        """Test the full pipeline with max_claims limit."""
        checker = RAGFactsChecker(mock_llm, max_claims=3)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert len(report.claims) <= 3

    def test_report_to_dict_after_full_pipeline(self, mock_llm, renewable_energy_dataset):
        """Test that to_dict() works after the full pipeline."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=renewable_energy_dataset["answer"],
            documents=renewable_energy_dataset["documents"],
        )
        d = report.to_dict()
        assert d["overall_verdict"] == report.overall_verdict
        assert d["overall_confidence"] == pytest.approx(report.overall_confidence, abs=0.01)
        assert isinstance(d["claims"], list)
        assert isinstance(d["results"], list)
        assert isinstance(d["dimensions"], dict)
        assert isinstance(d["hallucination_flags"], list)

    def test_pipeline_with_empty_answer(self, mock_llm):
        """Test the full pipeline with an empty answer."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer="",
            documents=["Some document."],
        )
        assert report.overall_verdict == "no_claims"

    def test_pipeline_with_empty_documents(self, mock_llm):
        """Test the full pipeline with empty documents."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer="Paris is the capital of France.",
            documents=[],
        )
        assert isinstance(report, CheckReport)

    def test_pipeline_claim_verdicts(self, mock_llm, climate_change_dataset):
        """Test that claim verdicts are valid."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        for result in report.results:
            assert result.verdict in ("supported", "contradicted", "not_enough_info")

    def test_pipeline_dimensions_sum(self, mock_llm, climate_change_dataset):
        """Test that dimensions are computed correctly."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        total = len(report.results)
        if total > 0:
            supported = sum(1 for r in report.results if r.verdict == "supported")
            contradicted = sum(1 for r in report.results if r.verdict == "contradicted")
            not_enough = sum(1 for r in report.results if r.verdict == "not_enough_info")

            assert report.dimensions["groundedness"] == round(supported / total * 100, 1)
            assert report.dimensions["contradiction_rate"] == round(contradicted / total * 100, 1)
            assert report.dimensions["hallucination_rate"] == round((contradicted + not_enough) / total * 100, 1)

    def test_pipeline_summary_not_empty(self, mock_llm, climate_change_dataset):
        """Test that the summary is not empty."""
        checker = RAGFactsChecker(mock_llm)
        report = checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert len(report.summary) > 0


class TestMockLLM:
    """Tests specifically for the MockLLM behavior."""

    def test_mock_llm_claim_extraction(self, mock_llm):
        """MockLLM should return CLAIM format for extraction prompts."""
        from rag_facts_check.prompts import format_claim_extraction_prompt
        prompt = format_claim_extraction_prompt("Paris is the capital of France.")
        response = mock_llm.generate(prompt)
        assert "CLAIM" in response

    def test_mock_llm_verification_supported(self, mock_llm):
        """MockLLM should return SUPPORTED for matching claims."""
        from rag_facts_check.prompts import format_claim_verification_evidence_first_prompt
        docs = ["Paris is the capital of France."]
        prompt = format_claim_verification_evidence_first_prompt(
            "Paris is the capital of France.", docs
        )
        response = mock_llm.generate(prompt)
        assert "SUPPORTED" in response

    def test_mock_llm_verification_contradicted(self, mock_llm):
        """MockLLM should return CONTRADICTED for contradicting claims."""
        from rag_facts_check.prompts import format_claim_verification_evidence_first_prompt
        docs = ["The Louvre is located in Paris, France."]
        prompt = format_claim_verification_evidence_first_prompt(
            "The Louvre is located in Berlin.", docs
        )
        response = mock_llm.generate(prompt)
        assert "CONTRADICTED" in response

    def test_mock_llm_call_count(self, mock_llm):
        """MockLLM should track call count."""
        assert mock_llm.call_count == 0
        mock_llm.generate("test prompt")
        assert mock_llm.call_count == 1
        mock_llm.generate("test prompt 2")
        assert mock_llm.call_count == 2

    def test_mock_llm_climate_hallucination(self, mock_llm):
        """MockLLM should detect climate change hallucinations."""
        from rag_facts_check.prompts import format_claim_verification_evidence_first_prompt
        docs = ["Climate models estimate a 2-4°C rise by the end of the century."]
        prompt = format_claim_verification_evidence_first_prompt(
            "Temperatures will rise by 5.7°C by 2100.", docs
        )
        response = mock_llm.generate(prompt)
        assert "CONTRADICTED" in response

    def test_mock_llm_renewable_statistic(self, mock_llm):
        """MockLLM should verify renewable energy statistics."""
        from rag_facts_check.prompts import format_claim_verification_evidence_first_prompt
        docs = ["In 2023, renewable energy sources accounted for 30% of global electricity generation."]
        prompt = format_claim_verification_evidence_first_prompt(
            "Renewables accounted for 30% of electricity in 2023.", docs
        )
        response = mock_llm.generate(prompt)
        assert "SUPPORTED" in response
