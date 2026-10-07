"""Tests for the FastAPI web service."""

import pytest
from starlette.testclient import TestClient

from rag_facts_check.server import create_app


@pytest.fixture
def app():
    """Create a fresh FastAPI app for each test."""
    return create_app()


@pytest.fixture
def client(app):
    """Sync test client (uses threadpool for async endpoints)."""
    return TestClient(app)


class TestHealth:
    """Tests for the /health endpoint."""

    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestCheckRequestValidation:
    """Tests for request validation on /check."""

    def test_missing_answer(self, client):
        response = client.post(
            "/check",
            json={"documents": [{"doc_id": "d1", "text": "Some doc"}]},
        )
        assert response.status_code == 422

    def test_missing_documents(self, client):
        response = client.post(
            "/check",
            json={"answer": "Paris is the capital of France."},
        )
        assert response.status_code == 422

    def test_document_missing_doc_id(self, client):
        response = client.post(
            "/check",
            json={
                "answer": "Paris is the capital of France.",
                "documents": [{"text": "Some doc"}],
            },
        )
        assert response.status_code == 422

    def test_document_missing_text(self, client):
        response = client.post(
            "/check",
            json={
                "answer": "Paris is the capital of France.",
                "documents": [{"doc_id": "d1"}],
            },
        )
        assert response.status_code == 422


class TestHalloumiSourceOffsets:
    """`/halloumi/generate` must keep source text verbatim.

    The frontend computes highlight offsets by concatenating the same source
    texts, so stripping a source on the server shifts every segment that comes
    after it — visible as wrong highlights once a claim cites several passages.
    """

    ANSWER = "Alpha claim."
    SOURCES = [
        {"text": "   alpha beta gamma", "title": "S1", "source_type": "file"},
        {"text": "\n\ngamma delta epsilon", "title": "S2", "source_type": "web"},
    ]
    QUOTES = ["alpha beta", "gamma delta"]

    @pytest.fixture
    def stub_client(self, monkeypatch):
        """App whose checker is replaced by a deterministic stub (no LLM)."""
        import rag_facts_check.checker as checker_module
        from rag_facts_check.models import (
            CheckReport,
            Claim,
            EvidenceSpan,
            Span,
            VerificationResult,
        )

        class StubChecker:
            seen_documents = None

            def __init__(self, *args, **kwargs):
                pass

            async def check(self, answer, documents, batch_size=None):
                StubChecker.seen_documents = documents
                spans = []
                for quote in self.quotes:
                    for i, doc in enumerate(documents):
                        pos = doc["text"].find(quote)
                        if pos >= 0:
                            spans.append(
                                EvidenceSpan(
                                    quote=quote,
                                    start=pos,
                                    end=pos + len(quote),
                                    document_index=i,
                                )
                            )
                            break
                return CheckReport(
                    answer=answer,
                    answer_score=10.0,
                    claims=[Claim(text="Alpha claim.", index=1, span=Span(0, 12))],
                    results=[
                        VerificationResult(
                            claim="Alpha claim.",
                            claim_index=1,
                            verdict="supported",
                            confidence=0,
                            evidence=list(self.quotes),
                            explanation="One passage per source.",
                            evidence_spans=spans,
                        )
                    ],
                )

        StubChecker.quotes = self.QUOTES
        monkeypatch.setattr(checker_module, "RAGFactsChecker", StubChecker)
        self.stub = StubChecker
        return TestClient(create_app())

    def test_sources_reach_the_checker_verbatim(self, stub_client):
        response = stub_client.post(
            "/halloumi/generate",
            json={"answer": self.ANSWER, "sources": self.SOURCES},
        )
        assert response.status_code == 200
        assert [doc["text"] for doc in self.stub.seen_documents] == [
            src["text"] for src in self.SOURCES
        ]

    def test_segments_line_up_with_unstripped_sources(self, stub_client):
        response = stub_client.post(
            "/halloumi/generate",
            json={"answer": self.ANSWER, "sources": self.SOURCES},
        )
        assert response.status_code == 200
        data = response.json()

        joined = "".join(src["text"] for src in self.SOURCES)
        segment_ids = data["claims"][0]["segmentIds"]
        assert segment_ids == ["0", "1"]
        quotes = [
            joined[data["segments"][sid]["startOffset"] : data["segments"][sid]["endOffset"]]
            for sid in segment_ids
        ]
        assert quotes == self.QUOTES

    def test_blank_sources_are_skipped_without_breaking_offsets(self, stub_client):
        sources = [{"text": "   \n "}, {"text": "   alpha beta gamma"}]
        response = stub_client.post(
            "/halloumi/generate",
            json={"answer": self.ANSWER, "sources": sources},
        )
        data = response.json()
        segment_ids = data["claims"][0]["segmentIds"]
        assert segment_ids
        seg = data["segments"][segment_ids[0]]
        assert sources[1]["text"][seg["startOffset"] : seg["endOffset"]] == "alpha beta"


class TestCheckEndpoint:
    """Tests for the /check endpoint with live LLM.

    These tests require a running LLM server.  Skip with: pytest -m "not llm"
    """

    @pytest.mark.llm
    def test_check_returns_report(self, client):
        response = client.post(
            "/check",
            json={
                "answer": "Paris is the capital of France.",
                "documents": [
                    {
                        "doc_id": "doc_1",
                        "text": "Paris is the capital of France.",
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "overall_verdict" in data
        assert "overall_confidence" in data
        assert "claims" in data
        assert "results" in data
        assert "dimensions" in data

    @pytest.mark.llm
    def test_check_with_multiple_documents(self, client):
        response = client.post(
            "/check",
            json={
                "answer": ("Paris is the capital of France. The Eiffel Tower was built in 1889."),
                "documents": [
                    {
                        "doc_id": "doc_1",
                        "text": "Paris is the capital of France.",
                    },
                    {
                        "doc_id": "doc_2",
                        "text": "The Eiffel Tower was built in 1889.",
                    },
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["claims"]) >= 1

    @pytest.mark.llm
    def test_check_doc_id_flows_through(self, client):
        """Verify that user-provided doc_id appears in results."""
        response = client.post(
            "/check",
            json={
                "answer": "Paris is the capital of France.",
                "documents": [
                    {
                        "doc_id": "my-custom-doc-id",
                        "text": "Paris is the capital of France.",
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        # When evidence retrieval is used, doc_id should appear in results
        if data.get("results"):
            doc_ids = [r.get("document_id") for r in data["results"] if r.get("document_id")]
            assert "my-custom-doc-id" in doc_ids

    @pytest.mark.llm
    def test_check_with_options(self, client):
        response = client.post(
            "/check",
            json={
                "answer": "Paris is the capital of France.",
                "documents": [
                    {
                        "doc_id": "doc_1",
                        "text": "Paris is the capital of France.",
                    }
                ],
                "options": {
                    "num_consistency_runs": 1,
                    "evidence_first": True,
                    "use_evidence_retrieval": True,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "overall_verdict" in data
