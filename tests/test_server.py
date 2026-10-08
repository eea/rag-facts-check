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


class TestHalloumiContextQuality:
    """`kind` on incoming sources must come back as an honest context label.

    Without it, a check run over search blurbs is indistinguishable from a check
    run over the full text the answer was written from — and the low score reads
    as "hallucination" in the UI.
    """

    ANSWER = "Alpha claim. Beta claim."
    SOURCES = [
        {"text": "alpha beta gamma", "kind": "chunk"},
        {"text": "delta epsilon zeta", "kind": "snippet"},
    ]

    @pytest.fixture
    def stub_client(self, monkeypatch):
        """App with a stub checker returning one supported and one unverifiable claim."""
        import rag_facts_check.checker as checker_module
        from rag_facts_check.models import CheckReport, Claim, Span, VerificationResult

        class StubChecker:
            def __init__(self, *args, **kwargs):
                pass

            async def check(self, answer, documents, batch_size=None):
                return CheckReport(
                    answer=answer,
                    answer_score=4.0,
                    claims=[
                        Claim(text="Alpha claim.", index=1, span=Span(0, 12)),
                        Claim(text="Beta claim.", index=2, span=Span(14, 24)),
                    ],
                    results=[
                        VerificationResult(
                            claim="Alpha claim.",
                            claim_index=1,
                            verdict="supported",
                            confidence=0,
                            evidence="alpha beta",
                            explanation="Found in the chunk.",
                        ),
                        VerificationResult(
                            claim="Beta claim.",
                            claim_index=2,
                            verdict="not_enough_info",
                            confidence=0,
                            evidence="N/A",
                            explanation="Not in the blurb.",
                        ),
                    ],
                )

        monkeypatch.setattr(checker_module, "RAGFactsChecker", StubChecker)
        return TestClient(create_app())

    def _post(self, client, sources):
        response = client.post(
            "/halloumi/generate", json={"answer": self.ANSWER, "sources": sources}
        )
        assert response.status_code == 200
        return response.json()

    def test_mixed_kinds_report_partial_context(self, stub_client):
        data = self._post(stub_client, self.SOURCES)
        assert data["context_quality"]["level"] == "partial"
        assert data["context_quality"]["chunk_sources"] == 1
        assert data["context_quality"]["snippet_sources"] == 1
        assert "1 of 2" in data["context_quality"]["note"]

    def test_unverifiable_claim_is_flagged_over_snippets(self, stub_client):
        data = self._post(stub_client, self.SOURCES)
        by_score = {claim["score"]: claim for claim in data["claims"]}
        assert by_score[0.4]["context_limited"] is True
        assert "context_limited" not in by_score[1.0]

    def test_all_chunk_kinds_report_full_context(self, stub_client):
        sources = [{**src, "kind": "chunk"} for src in self.SOURCES]
        data = self._post(stub_client, sources)
        assert data["context_quality"]["level"] == "full"
        assert all("context_limited" not in claim for claim in data["claims"])

    def test_missing_kind_reports_unknown(self, stub_client):
        sources = [{k: v for k, v in src.items() if k != "kind"} for src in self.SOURCES]
        assert self._post(stub_client, sources)["context_quality"]["level"] == "unknown"

    def test_invalid_kind_is_treated_as_unknown(self, stub_client):
        sources = [{**src, "kind": "full_text"} for src in self.SOURCES]
        assert self._post(stub_client, sources)["context_quality"]["level"] == "unknown"

    def test_plain_string_sources_report_unknown(self, stub_client):
        sources = [src["text"] for src in self.SOURCES]
        assert self._post(stub_client, sources)["context_quality"]["level"] == "unknown"

    def test_blank_sources_are_excluded_from_the_counts(self, stub_client):
        sources = [{"text": "   ", "kind": "snippet"}] + self.SOURCES
        quality = self._post(stub_client, sources)["context_quality"]
        assert quality["sources"] == 2
        assert quality["snippet_sources"] == 1


class TestCheckEndpointContextQuality:
    """`POST /check` reports the same `context_quality` as `/halloumi/generate`."""

    ANSWER = "Alpha claim. Beta claim."

    @pytest.fixture
    def stub_client(self, monkeypatch):
        import rag_facts_check.checker as checker_module
        from rag_facts_check.models import CheckReport, Claim, VerificationResult

        class StubChecker:
            def __init__(self, *args, **kwargs):
                pass

            async def check(self, answer, documents, batch_size=None):
                return CheckReport(
                    answer=answer,
                    answer_score=4.0,
                    claims=[Claim(text="Alpha claim.", index=1)],
                    results=[
                        VerificationResult(
                            claim="Alpha claim.",
                            claim_index=1,
                            verdict="supported",
                            confidence=0,
                            evidence="alpha beta",
                            explanation="Found.",
                        ),
                        VerificationResult(
                            claim="Beta claim.",
                            claim_index=2,
                            verdict="not_enough_info",
                            confidence=0,
                            evidence="N/A",
                            explanation="Not in the blurb.",
                        ),
                    ],
                )

        monkeypatch.setattr(checker_module, "RAGFactsChecker", StubChecker)
        return TestClient(create_app())

    def _post(self, client, documents):
        response = client.post("/check", json={"answer": self.ANSWER, "documents": documents})
        assert response.status_code == 200
        return response.json()

    def test_mixed_kinds_report_partial_context(self, stub_client):
        data = self._post(
            stub_client,
            [
                {"doc_id": "doc_1", "text": "alpha beta gamma", "kind": "chunk"},
                {"doc_id": "doc_2", "text": "delta epsilon", "kind": "snippet"},
            ],
        )
        assert data["context_quality"]["level"] == "partial"
        assert data["context_quality"]["chunk_sources"] == 1
        assert data["context_quality"]["snippet_sources"] == 1

    def test_all_chunk_kinds_report_full_context(self, stub_client):
        data = self._post(
            stub_client,
            [{"doc_id": "doc_1", "text": "alpha beta", "kind": "chunk"}],
        )
        assert data["context_quality"]["level"] == "full"

    def test_missing_kind_reports_unknown(self, stub_client):
        data = self._post(stub_client, [{"doc_id": "doc_1", "text": "alpha beta"}])
        assert data["context_quality"]["level"] == "unknown"

    def test_invalid_kind_is_accepted_and_unknown(self, stub_client):
        """A bad kind is a labelling mistake, not a request error."""
        data = self._post(
            stub_client,
            [{"doc_id": "doc_1", "text": "alpha beta", "kind": "full_text"}],
        )
        assert data["context_quality"]["level"] == "unknown"

    def test_report_shape_is_unchanged_apart_from_the_new_key(self, stub_client):
        data = self._post(stub_client, [{"doc_id": "doc_1", "text": "alpha beta"}])
        for key in ("overall_verdict", "claims", "results", "dimensions"):
            assert key in data


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
