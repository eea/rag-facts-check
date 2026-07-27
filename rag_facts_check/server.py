"""
FastAPI web service for the RAG fact-checking pipeline.

Exposes ``POST /check`` for async fact-checking of RAG answers.
Reads LLM configuration from environment variables at startup.

Usage::

    uvicorn rag_facts_check.server:app --host 0.0.0.0 --port 8000
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def _load_env() -> dict[str, str]:
    """Load environment variables from .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_vars: dict[str, str] = {}
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed, rely on system env

    for key in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL", "LLM_TEMPERATURE"):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DocumentInput(BaseModel):
    """A source document with an optional identifier."""

    doc_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document text")


class CheckOptions(BaseModel):
    """Optional overrides for the fact-checking pipeline."""

    max_claims: int | None = Field(None, description="Maximum number of claims to verify")
    num_consistency_runs: int = Field(1, description="Self-consistency runs (1 = single pass)")
    evidence_first: bool = Field(True, description="Use evidence-first multi-step prompting")
    use_evidence_retrieval: bool = Field(True, description="Retrieve relevant chunks per claim")


class CheckRequest(BaseModel):
    """Request body for POST /check."""

    answer: str = Field(..., description="RAG-generated answer to verify")
    documents: list[DocumentInput] = Field(
        ..., description="Source documents retrieved by the RAG system"
    )
    options: CheckOptions | None = Field(None, description="Optional pipeline overrides")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG Facts Check",
        description=(
            "Fact-checking service for RAG-generated answers. "
            "Extracts claims, verifies each against source documents, "
            "and returns a detailed report."
        ),
        version="0.2.0",
    )

    # CORS
    cors_origins_str = os.environ.get("CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy LLM initialization
    _llm = None
    _checker = None

    def _get_checker():
        nonlocal _llm, _checker
        if _checker is None:
            from .checker import RAGFactsChecker
            from .llm import AsyncAPILLM

            env = _load_env()
            api_base = env.get("LLM_API_BASE", "http://localhost:4002/v1")
            api_key = env.get("LLM_API_KEY")
            model = env.get("LLM_MODEL", "gemma")
            temperature = float(env.get("LLM_TEMPERATURE", "0.1"))

            api_url = api_base.rstrip("/") + "/chat/completions"
            _llm = AsyncAPILLM(
                api_url=api_url,
                model_name=model,
                api_key=api_key,
                temperature=temperature,
                chat_mode=True,
            )
            _checker = RAGFactsChecker(_llm)
        return _checker

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "version": "0.2.0"}

    @app.post("/check")
    async def check(request: CheckRequest) -> dict:
        """Run the fact-checking pipeline on a RAG answer.

        Accepts an answer and source documents, extracts claims,
        verifies each claim against the documents, and returns a
        detailed report with per-claim verdicts and evidence.
        """
        checker = _get_checker()

        # Build documents list for the checker
        documents = [{"doc_id": d.doc_id, "text": d.text} for d in request.documents]

        try:
            report = checker.check(
                answer=request.answer,
                documents=documents,
            )
            return report.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


# Default app instance for uvicorn
app = create_app()
