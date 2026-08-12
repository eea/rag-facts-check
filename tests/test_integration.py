"""
End-to-end integration tests for the RAG fact-checking pipeline.

Structural tests (pipeline wiring, edge cases) use mock LLMs.
Semantic tests (hallucination detection, groundedness) require a real
LLM and are marked with ``@pytest.mark.llm`` (skipped by default).
"""

import pytest

from rag_facts_check import EvidenceRetriever, RAGFactsChecker
from rag_facts_check.models import CheckReport


class TestEndToEnd:
    """End-to-end structural tests — verify pipeline wiring, not LLM quality."""

    async def test_full_pipeline_supported(self, mock_llm):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
            answer="Paris is the capital of France. The Eiffel Tower was built in 1889.",
            documents=["Paris is the capital of France. The Eiffel Tower was built in 1889."],
        )
        assert isinstance(report, CheckReport)
        assert len(report.claims) >= 1
        assert len(report.results) == len(report.claims)

    async def test_full_pipeline_contradicted(self, mock_llm_contradicted):
        checker = RAGFactsChecker(mock_llm_contradicted)
        report = await checker.check(
            answer="The Louvre Museum is located in Berlin.",
            documents=["The Louvre Museum is located in Paris, France."],
        )
        assert isinstance(report, CheckReport)
        assert len(report.hallucination_flags) > 0

    async def test_full_pipeline_with_climate_dataset(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        assert len(report.claims) >= 3
        assert len(report.results) == len(report.claims)

    async def test_full_pipeline_with_renewable_dataset(self, mock_llm, renewable_energy_dataset):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
            answer=renewable_energy_dataset["answer"],
            documents=renewable_energy_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        assert len(report.claims) >= 3
        assert len(report.results) == len(report.claims)

    async def test_pipeline_with_self_consistency(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm, num_consistency_runs=3)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)
        if report.results:
            assert report.results[0].consistency_score is not None

    async def test_pipeline_no_evidence_retrieval(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm, use_evidence_retrieval=False)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)

    async def test_pipeline_evidence_first_off(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm, evidence_first=False)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)

    async def test_pipeline_custom_retriever(self, mock_llm, climate_change_dataset):
        retriever = EvidenceRetriever(chunk_size=50, top_k=2)
        checker = RAGFactsChecker(mock_llm, retriever=retriever)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert isinstance(report, CheckReport)

    async def test_pipeline_max_claims(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm, max_claims=3)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert len(report.claims) <= 3

    async def test_report_to_dict_after_full_pipeline(self, mock_llm, renewable_energy_dataset):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
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

    async def test_pipeline_with_empty_answer(self, mock_llm):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(answer="", documents=["Some document."])
        assert report.overall_verdict == "no_claims"

    async def test_pipeline_with_empty_documents(self, mock_llm):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
            answer="Paris is the capital of France.",
            documents=[],
        )
        assert isinstance(report, CheckReport)

    async def test_pipeline_claim_verdicts(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        for result in report.results:
            assert result.verdict in ("supported", "contradicted", "not_enough_info")

    async def test_pipeline_dimensions_sum(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
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
            expected_hallucination = round((contradicted + not_enough) / total * 100, 1)
            assert report.dimensions["hallucination_rate"] == expected_hallucination

    async def test_pipeline_summary_not_empty(self, mock_llm, climate_change_dataset):
        checker = RAGFactsChecker(mock_llm)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        assert len(report.summary) > 0


class TestEndToEndWithRealLLM:
    """Semantic E2E tests that require a real LLM.

    These verify that the pipeline actually detects hallucinations and
    supports correct claims — behaviour that keyword mocks cannot
    reliably reproduce.
    """

    @pytest.mark.llm
    async def test_climate_change_flags_hallucination(self, live_llm, climate_change_dataset):
        checker = RAGFactsChecker(live_llm)
        report = await checker.check(
            answer=climate_change_dataset["answer"],
            documents=climate_change_dataset["documents"],
        )
        # 5.7°C contradicts 2-4°C
        assert len(report.hallucination_flags) > 0
        assert report.overall_verdict in ["largely_unsupported", "partially_supported"]

    @pytest.mark.llm
    async def test_renewable_energy_no_hallucination(self, live_llm, renewable_energy_dataset):
        checker = RAGFactsChecker(live_llm)
        report = await checker.check(
            answer=renewable_energy_dataset["answer"],
            documents=renewable_energy_dataset["documents"],
        )
        assert report.dimensions["groundedness"] >= 50.0
