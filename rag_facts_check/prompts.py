"""
Prompt templates for claim extraction and verification.

Prompts are stored as plain text files in the ``prompts/`` package directory
so they can be reviewed and edited independently of the Python code.

Template variables (substituted at runtime):
- ``{system_prompt}`` — the system instruction for the phase
- ``{text}`` — the answer text (extraction)
- ``{claim}`` — the claim text (verification)
- ``{documents}`` — formatted source documents (verification)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load(name: str) -> str:
    """Load a prompt file from the ``prompts/`` directory."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip("\n")


# ---------------------------------------------------------------------------
# Claim Extraction
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_SYSTEM = _load("claim-extraction-system.txt")
CLAIM_EXTRACTION_PROMPT = _load("claim-extraction-prompt.txt")


def format_claim_extraction_prompt(text: str) -> str:
    """Build the full claim extraction prompt."""
    return CLAIM_EXTRACTION_PROMPT.format(
        system_prompt=CLAIM_EXTRACTION_SYSTEM, text=text
    )


# ---------------------------------------------------------------------------
# Claim Verification — Standard
# ---------------------------------------------------------------------------

CLAIM_VERIFICATION_SYSTEM = _load("claim-verification-system.txt")
CLAIM_VERIFICATION_PROMPT = _load("claim-verification-prompt.txt")


def format_claim_verification_prompt(claim: str, documents: list[str]) -> str:
    """Build the full claim verification prompt.

    Args:
        claim: The claim text to verify.
        documents: List of document strings.

    Returns:
        Formatted prompt string.
    """
    formatted_docs = format_documents(documents)
    return CLAIM_VERIFICATION_PROMPT.format(
        system_prompt=CLAIM_VERIFICATION_SYSTEM,
        claim=claim,
        documents=formatted_docs,
    )


# ---------------------------------------------------------------------------
# Claim Verification — Evidence-First (Multi-Step)
# ---------------------------------------------------------------------------

CLAIM_VERIFICATION_EVIDENCE_FIRST_SYSTEM = _load(
    "claim-verification-evidence-first-system.txt"
)
CLAIM_VERIFICATION_EVIDENCE_FIRST_PROMPT = _load(
    "claim-verification-evidence-first-prompt.txt"
)


def format_claim_verification_evidence_first_prompt(
    claim: str, documents: list[str]
) -> str:
    """Build the evidence-first multi-step verification prompt.

    This prompt explicitly asks the model to extract evidence first,
    then compare it to the claim, then provide a verdict. This reduces
    hallucinated evaluations where the verifier makes up evidence.

    Args:
        claim: The claim text to verify.
        documents: List of document strings.

    Returns:
        Formatted prompt string.
    """
    formatted_docs = format_documents(documents)
    return CLAIM_VERIFICATION_EVIDENCE_FIRST_PROMPT.format(
        system_prompt=CLAIM_VERIFICATION_EVIDENCE_FIRST_SYSTEM,
        claim=claim,
        documents=formatted_docs,
    )


# ---------------------------------------------------------------------------
# Document Formatting
# ---------------------------------------------------------------------------


def format_documents(
    documents: list[str] | list[dict[str, str]],
    max_chars_per_doc: int = 2000,
    max_total_chars: int = 8000,
) -> str:
    """Format a list of documents into a single string for the LLM.

    Args:
        documents: List of document strings, or list of dicts with
            ``{"text": ..., "title": ...}`` entries. When dicts are
            provided, the title is included as a header.
        max_chars_per_doc: Maximum characters per document (truncated).
        max_total_chars: Maximum total characters across all documents.

    Returns:
        Formatted documents string.
    """
    if not documents:
        return "(No source documents provided)"

    parts = []
    total_chars = 0

    for i, doc in enumerate(documents):
        if total_chars >= max_total_chars:
            parts.append("\n[Remaining documents truncated to fit context window]")
            break

        if isinstance(doc, dict):
            text = doc["text"]
            title = doc.get("title")
            header = f"Document {i + 1}: {title}" if title else f"Document {i + 1}:"
        else:
            text = doc
            header = f"Document {i + 1}:"

        # Truncate individual document
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "... [truncated]"

        remaining = max_total_chars - total_chars
        if len(text) > remaining:
            text = text[:remaining] + "... [truncated]"

        parts.append(f"{header}\n{text}\n")
        total_chars += len(text)

    return "\n".join(parts)
