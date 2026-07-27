"""
Shared test fixtures for the rag_facts_check test suite.

Fixtures provide sample data, mock LLMs, and pre-configured checkers
so individual test modules can focus on specific behaviors.
"""

import json
from pathlib import Path

import pytest

from rag_facts_check import EvidenceRetriever, MockLLM, RAGFactsChecker
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


# ─── LLM Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """Fresh MockLLM instance for each test."""
    return MockLLM()


@pytest.fixture
def mock_llm_with_calls():
    """MockLLM that tracks call count across tests."""
    llm = MockLLM()
    yield llm
    assert llm.call_count > 0


# ─── Checker Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def checker(mock_llm):
    """Default RAGFactsChecker with MockLLM."""
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
