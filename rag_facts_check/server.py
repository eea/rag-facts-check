"""
FastAPI web service for the RAG fact-checking pipeline.

Exposes ``POST /check`` for async fact-checking of RAG answers.
Reads LLM configuration from environment variables at startup.

Usage::

    uvicorn rag_facts_check.server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("rag_facts_check")


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


class HalloumiRequest(BaseModel):
    """Request body for POST /halloumi/generate (halloumi-compatible)."""

    answer: str = Field(..., description="RAG-generated answer to verify")
    sources: list[str] = Field(..., description="Source document texts (plain strings)")
    max_context_segments: int = Field(0, description="Max context segments (unused, for compat)")


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

            # Build instructor-wrapped client for structured output
            try:
                import instructor
                from openai import AsyncOpenAI

                base_url = api_base.rstrip("/")
                openai_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key or "not-needed",
                )
                instructor_client = instructor.from_openai(
                    openai_client, mode=instructor.Mode.MD_JSON
                )
            except ImportError:
                instructor_client = None
                log.warning(
                    "instructor/openai not available, falling back to raw LLM calls. "
                    "Install with: pip install instructor openai"
                )

            _checker = RAGFactsChecker(
                _llm,
                instructor_client=instructor_client,
                model=model,
                temperature=temperature,
            )
        return _checker

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "version": "0.2.0"}

    @app.post("/halloumi/generate")
    async def halloumi_generate(request: HalloumiRequest) -> dict:
        """Halloumi-compatible endpoint for claim verification.

        Accepts the same request format as the halloumi middleware and
        returns a response in halloumi's format so the existing frontend
        components work without changes.

        Request: {"answer": "...", "sources": ["doc1...", "doc2..."]}
        Response: {"answer_score": 0-10, "claims": [...], "segments": {...}}
        """
        checker = _get_checker()

        # Convert plain string sources to document dicts
        sources = [s for s in request.sources if s.strip()]
        documents = [{"doc_id": f"doc_{i + 1}", "text": src} for i, src in enumerate(sources)]

        log.info(
            "halloumi/generate: answer=%d chars, sources=%d docs (%d non-empty)",
            len(request.answer),
            len(request.sources),
            len(sources),
        )

        try:
            report = await checker.check(
                answer=request.answer,
                documents=documents,
            )
            log.info(
                "halloumi/generate: claims=%d, results=%d, verdict=%s",
                len(report.claims),
                len(report.results),
                report.overall_verdict,
            )
            for i, c in enumerate(report.claims):
                log.info("  claim[%d]: span=%s text=%s", i, c.span, c.text[:80])
            for i, r in enumerate(report.results):
                log.info(
                    "  result[%d]: verdict=%s confidence=%d span=%s",
                    i,
                    r.verdict,
                    r.confidence,
                    r.evidence_span,
                )
            return _to_halloumi_format(report, sources, request.answer)
        except Exception as e:
            log.exception("halloumi/generate error")
            raise HTTPException(status_code=500, detail=str(e)) from e

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
            report = await checker.check(
                answer=request.answer,
                documents=documents,
            )
            return report.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


# Default app instance for uvicorn
app = create_app()


# ---------------------------------------------------------------------------
# Halloumi response adapter
# ---------------------------------------------------------------------------


def _to_halloumi_format(report, sources: list[str], answer_text: str = "") -> dict:
    """Convert a CheckReport to halloumi-compatible response format.

    Halloumi format:
    {
      "answer_score": 7.5,
      "claims": [
        {
          "startOffset": 12,
          "endOffset": 58,
          "segmentIds": ["0", "2"],
          "score": 1.0,
          "rationale": "..."
        }
      ],
      "segments": {
        "0": {"startOffset": 45, "endOffset": 92},
        "2": {"startOffset": 200, "endOffset": 250}
      }
    }

    The segments are computed from evidence_spans, mapped into the
    joined sources string so the frontend can highlight them.

    Per-claim score is verdict-based (not raw LLM confidence):
    - supported: 1.0
    - not_enough_info: 0.4
    - contradicted: 0.0
    """
    # Verdict-to-score mapping for per-claim scores
    verdict_scores = {
        "supported": 1.0,
        "not_enough_info": 0.4,
        "contradicted": 0.0,
    }

    # Build a mapping from source index to start offset in joined string
    source_offsets: list[int] = []
    offset = 0
    for src in sources:
        source_offsets.append(offset)
        offset += len(src) + 1  # +1 for the newline

    segments: dict[str, dict[str, int]] = {}
    claims: list[dict] = []

    for result in report.results:
        claim = report.claims[result.claim_index - 1]

        # Build claim span — if the LLM paraphrased the claim and no span
        # was found, fall back to the full answer range so the claim is
        # still included in the output.
        if claim.span:
            start_offset = claim.span.start
            end_offset = claim.span.end
        else:
            log.debug(
                "_to_halloumi: claim[%d] has no span (LLM paraphrased), using full answer range",
                result.claim_index,
            )
            start_offset = 0
            end_offset = len(answer_text)

        # Build segment IDs from evidence spans
        segment_ids: list[str] = []
        if result.evidence_span:
            seg_id = str(len(segments))
            segments[seg_id] = {
                "id": int(seg_id),
                "startOffset": result.evidence_span.start,
                "endOffset": result.evidence_span.end,
            }
            segment_ids.append(seg_id)

        # Verdict-based score (not raw LLM confidence)
        score = verdict_scores.get(result.verdict, 0.4)

        # Extract the claim text from the answer using the span
        claim_string = answer_text[start_offset:end_offset] if claim.span else claim.text

        claims.append(
            {
                "claimString": claim_string,
                "startOffset": start_offset,
                "endOffset": end_offset,
                "segmentIds": segment_ids,
                "score": score,
                "rationale": result.explanation,
            }
        )

    with_spans = sum(1 for r in report.results if report.claims[r.claim_index - 1].span)
    log.info(
        "_to_halloumi: %d claims in output (of %d results, %d with spans)",
        len(claims),
        len(report.results),
        with_spans,
    )
    return {
        "answer_score": report.answer_score,
        "claims": claims,
        "segments": segments,
    }
