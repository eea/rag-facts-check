"""
Shared test fixtures for the rag_facts_check test suite.

Fixtures provide sample data, mock LLMs, and pre-configured checkers
so individual test modules can focus on specific behaviors.

LLM mocking strategy
--------------------
We use ``unittest.mock.AsyncMock`` so each test declares exactly what
the LLM should return.  No keyword-matching, no prompt parsing, no
fragile regex — just explicit response sequences.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rag_facts_check import EvidenceRetriever, RAGFactsChecker
from rag_facts_check.models import Claim, VerificationResult

# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_DATASETS_DIR = PROJECT_ROOT / "mock_datasets"

# ─── Sample Data ─────────────────────────────────────────────────────────────

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

SAMPLE_ANSWER_SIMPLE = "Paris is the capital of France."
SAMPLE_DOCS_SIMPLE = ["Paris is the capital of France."]

SAMPLE_ANSWER_RENEWABLE = (
    "In 2023, renewable energy sources accounted for 30% of global electricity "
    "generation, with solar photovoltaic and onshore wind leading the growth. "
    "The International Renewable Energy Agency (IRENA) projects that renewables "
    "will reach 42% of global electricity generation by 2028. "
    "Battery storage costs have fallen by 89% since 2010."
)

SAMPLE_DOCS_RENEWABLE = [
    "In 2023, renewable energy sources accounted for 30% of global electricity "
    "generation, with solar and wind leading the growth.",
    "IRENA projects renewables will reach 42% of global electricity by 2028.",
    "Battery storage costs have fallen by 89% since 2010, enabling greater "
    "grid integration of variable renewables.",
]

# ─── LLM Response Helpers ────────────────────────────────────────────────────


def _extraction_response(text: str) -> str:
    """Build a CLAIM-format extraction response from sentences in *text*."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for i, s in enumerate(sentences, 1):
        s = s.strip().rstrip(".")
        if s and len(s) > 5:
            claims.append(f"CLAIM {i}: {s}.")
    return "\n".join(claims) if claims else "NO CLAIMS"


def _verification_json(verdict: str, evidence: str, explanation: str, document_index: int | None = None) -> str:
    """Build a JSON verification response."""
    obj = {
        "verdict": verdict.upper(),
        "evidence": evidence,
        "explanation": explanation,
    }
    if document_index is not None:
        obj["document_index"] = document_index
    return json.dumps(obj)


# ─── LLM Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """AsyncMock LLM that returns parseable responses for any prompt.

    The mock inspects the prompt to decide which response to return:
    - Extraction prompts → CLAIM-format sentences
    - Verification prompts → JSON with ``supported`` verdict

    ``llm.generate`` is an ``AsyncMock`` so ``.called`` / ``.call_count`` work.
    """
    llm = AsyncMock()

    async def _respond(prompt: str, **kwargs) -> str:
        lower = prompt.lower()
        # Extraction prompts mention "extract" and "answer" or "text"
        if "extract" in lower and ("answer" in lower or "text:" in lower):
            return _extraction_response(prompt)
        # Verification prompts
        return _verification_json("supported", "Evidence from documents.", "Match found.")

    llm.generate = AsyncMock(side_effect=_respond)
    return llm


@pytest.fixture
def mock_llm_contradicted():
    """AsyncMock LLM whose verification responses return ``contradicted``."""
    llm = AsyncMock()

    async def _respond(prompt: str, **kwargs) -> str:
        lower = prompt.lower()
        if "extract" in lower and ("answer" in lower or "text:" in lower):
            return _extraction_response(prompt)
        return _verification_json("contradicted", "Contradictory evidence.", "Does not match.")

    llm.generate = AsyncMock(side_effect=_respond)
    return llm


@pytest.fixture
def live_llm():
    """Real LLM for ``@pytest.mark.llm`` tests. Reads config from .env."""
    import os
    from rag_facts_check import AsyncAPILLM

    base = os.getenv("LLM_API_BASE", "http://localhost:4002/v1")
    url = base.rstrip("/") + "/chat/completions"
    model = os.getenv("LLM_MODEL", "gemma")
    api_key = os.getenv("LLM_API_KEY", "not-needed")
    return AsyncAPILLM(url, model_name=model, api_key=api_key, chat_mode=True)


# ─── Checker Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def checker(mock_llm):
    """Default RAGFactsChecker with mock LLM."""
    return RAGFactsChecker(mock_llm)


@pytest.fixture
def checker_no_retrieval(mock_llm):
    """Checker without evidence retrieval (passes all docs to verifier)."""
    return RAGFactsChecker(mock_llm, use_evidence_retrieval=False)


@pytest.fixture
def checker_self_consistency(mock_llm):
    """Checker with self-consistency (3 runs)."""
    return RAGFactsChecker(mock_llm, num_consistency_runs=3)


@pytest.fixture
def checker_evidence_first_off(mock_llm):
    """Checker with evidence-first prompting disabled."""
    return RAGFactsChecker(mock_llm, evidence_first=False)


@pytest.fixture
def retriever():
    """Default EvidenceRetriever."""
    return EvidenceRetriever(chunk_size=200, top_k=3)


# ─── Dataset Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def climate_change_dataset():
    """Load the climate change hallucinated dataset."""
    path = MOCK_DATASETS_DIR / "climate_change_hallucinated.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def renewable_energy_dataset():
    """Load the renewable energy supported dataset."""
    path = MOCK_DATASETS_DIR / "renewable_energy_supported.json"
    with open(path) as f:
        return json.load(f)


# ─── Model Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_claim():
    return Claim(text="Paris is the capital of France.", index=1)


@pytest.fixture
def sample_claims():
    return [
        Claim(text="Paris is the capital of France.", index=1),
        Claim(text="The Eiffel Tower was built in 1889.", index=2),
        Claim(text="The Louvre Museum is located in Berlin.", index=3),
    ]


@pytest.fixture
def supported_result():
    return VerificationResult(
        claim="Paris is the capital of France.",
        claim_index=1,
        verdict="supported",
        confidence=95,
        evidence="Paris is the capital of France.",
        explanation="The source document explicitly states this.",
    )


@pytest.fixture
def contradicted_result():
    return VerificationResult(
        claim="The Louvre is in Berlin.",
        claim_index=2,
        verdict="contradicted",
        confidence=85,
        evidence="The Louvre is located in Paris, France.",
        explanation="Documents state Paris, claim says Berlin.",
    )


@pytest.fixture
def not_enough_info_result():
    return VerificationResult(
        claim="The moon is made of cheese.",
        claim_index=3,
        verdict="not_enough_info",
        confidence=60,
        evidence="N/A",
        explanation="Documents do not contain sufficient information.",
    )
