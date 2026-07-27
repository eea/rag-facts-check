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
