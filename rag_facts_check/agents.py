"""Atomic agents for the RAG fact-checking pipeline.

Uses atomic-agents + instructor for structured LLM I/O with Pydantic
schemas. Each agent phase gets its own input/output schema pair.
"""

from __future__ import annotations

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from instructor import Mode
from pydantic import Field

# ---------------------------------------------------------------------------
# Claim Extraction schemas
# ---------------------------------------------------------------------------


class ExtractedClaim(BaseIOSchema):
    """A single extracted factual claim with its original text fragment."""

    claim: str = Field(
        ...,
        description=(
            "The atomic factual claim, rephrased for clarity if needed. "
            "Must be a single verifiable statement."
        ),
    )
    original_text: str = Field(
        ...,
        description=(
            "The EXACT verbatim text fragment from the source answer that "
            "this claim is based on. Must be a substring of the original "
            "answer text — do NOT paraphrase this field."
        ),
    )


class ClaimExtractionInput(BaseIOSchema):
    """Input for the claim extraction agent."""

    answer: str = Field(..., description="The RAG-generated answer text to extract claims from.")


class ClaimExtractionOutput(BaseIOSchema):
    """Output from the claim extraction agent."""

    claims: list[ExtractedClaim] = Field(
        ...,
        description=(
            "List of extracted factual claims. Each claim has a rephrased "
            "version and the exact original text fragment it was derived from. "
            "Empty list if no factual claims were found."
        ),
    )
    has_claims: bool = Field(
        ...,
        description="True if any factual claims were extracted, false otherwise.",
    )


# ---------------------------------------------------------------------------
# Claim Verification schemas
# ---------------------------------------------------------------------------


class VerificationInput(BaseIOSchema):
    """Input for the claim verification agent."""

    claim: str = Field(..., description="The factual claim to verify.")
    documents: str = Field(
        ...,
        description="Source documents formatted as text, for evidence lookup.",
    )


class VerificationOutput(BaseIOSchema):
    """Output from the claim verification agent."""

    verdict: str = Field(
        ...,
        description="One of: SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO.",
        pattern=r"^(SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO)$",
    )
    confidence: int = Field(
        ...,
        description="Confidence score 0-100.",
        ge=0,
        le=100,
    )
    evidence: str = Field(
        ...,
        description=(
            "Exact quote from source documents supporting or contradicting "
            "the claim, or 'N/A' if not enough info."
        ),
    )
    explanation: str = Field(
        ...,
        description="Brief explanation of the verdict (max 2 sentences).",
    )


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def make_claim_extraction_agent(
    client,
    model: str,
    system_prompt: str,
    *,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> AtomicAgent[ClaimExtractionInput, ClaimExtractionOutput]:
    """Build a claim extraction agent.

    Args:
        client: Instructor-compatible async client (e.g. from instructor.patch).
        model: Model name.
        system_prompt: System prompt for claim extraction.
        temperature: Sampling temperature.
        max_retries: Maximum retry attempts for structured output.
    """
    return AtomicAgent[ClaimExtractionInput, ClaimExtractionOutput](
        AgentConfig(
            client=client,
            model=model,
            mode=Mode.JSON,
            model_api_parameters={
                "temperature": temperature,
                "max_retries": max_retries,
            },
            system_prompt_generator=lambda **_: system_prompt,
            max_context_tokens=None,
        ),
    )


def make_verification_agent(
    client,
    model: str,
    system_prompt: str,
    *,
    temperature: float = 0.1,
    max_retries: int = 3,
    evidence_first: bool = True,
) -> AtomicAgent[VerificationInput, VerificationOutput]:
    """Build a claim verification agent.

    Args:
        client: Instructor-compatible async client.
        model: Model name.
        system_prompt: System prompt for verification.
        temperature: Sampling temperature.
        max_retries: Maximum retry attempts.
        evidence_first: If True, use evidence-first multi-step prompt.
    """
    return AtomicAgent[VerificationInput, VerificationOutput](
        AgentConfig(
            client=client,
            model=model,
            mode=Mode.JSON,
            model_api_parameters={
                "temperature": temperature,
                "max_retries": max_retries,
            },
            system_prompt_generator=lambda **_: system_prompt,
            max_context_tokens=None,
        ),
    )
