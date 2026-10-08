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
# Evidence Retrieval (LLM-based)
# ---------------------------------------------------------------------------

EVIDENCE_RETRIEVAL_SYSTEM = _load("evidence-retrieval-system.txt")
EVIDENCE_RETRIEVAL_PROMPT = _load("evidence-retrieval-prompt.txt")


def format_evidence_retrieval_prompt(
    claim: str,
    chunks: list[dict],
) -> str:
    """Build the LLM-based evidence retrieval prompt.

    Args:
        claim: The claim text to find evidence for.
        chunks: List of chunk dicts with ``{"id": int, "title": str, "text": str}``.

    Returns:
        Formatted prompt string.
    """
    chunk_lines = []
    for c in chunks:
        title_part = f" — {c['title']}" if c.get("title") else ""
        chunk_lines.append(f"Chunk {c['id']}{title_part}:\n{c['text']}")
    chunks_text = "\n\n".join(chunk_lines)

    return EVIDENCE_RETRIEVAL_PROMPT.format(
        system_prompt=EVIDENCE_RETRIEVAL_SYSTEM,
        claim=claim,
        chunks=chunks_text,
    )


# ---------------------------------------------------------------------------
# Claim Verification — Standard
# ---------------------------------------------------------------------------

CLAIM_VERIFICATION_SYSTEM = _load("claim-verification-system.txt")
CLAIM_VERIFICATION_PROMPT = _load("claim-verification-prompt.txt")


def format_claim_verification_prompt(
    claim: str,
    documents: list[str],
    max_docs_chars: int = 100000,
    max_chars_per_doc: int = 10000,
) -> str:
    """Build the full claim verification prompt.

    Args:
        claim: The claim text to verify.
        documents: List of document strings.
        max_docs_chars: Total character budget for all documents.
        max_chars_per_doc: Fairness cap per document, applied only when the
            corpus does not fit in ``max_docs_chars``.

    Returns:
        Formatted prompt string.
    """
    formatted_docs = format_documents(
        documents,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_docs_chars,
    )
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
    claim: str,
    documents: list[str],
    max_docs_chars: int = 100000,
    max_chars_per_doc: int = 10000,
) -> str:
    """Build the evidence-first multi-step verification prompt.

    This prompt explicitly asks the model to extract evidence first,
    then compare it to the claim, then provide a verdict. This reduces
    hallucinated evaluations where the verifier makes up evidence.

    Args:
        claim: The claim text to verify.
        documents: List of document strings.
        max_docs_chars: Total character budget for all documents.
        max_chars_per_doc: Fairness cap per document, applied only when the
            corpus does not fit in ``max_docs_chars``.

    Returns:
        Formatted prompt string.
    """
    formatted_docs = format_documents(
        documents,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_docs_chars,
    )
    return CLAIM_VERIFICATION_EVIDENCE_FIRST_PROMPT.format(
        system_prompt=CLAIM_VERIFICATION_EVIDENCE_FIRST_SYSTEM,
        claim=claim,
        documents=formatted_docs,
    )


# ---------------------------------------------------------------------------
# Claim Verification — Batch
# ---------------------------------------------------------------------------

CLAIM_VERIFICATION_BATCH_SYSTEM = _load("claim-verification-batch-system.txt")
CLAIM_VERIFICATION_BATCH_PROMPT = _load("claim-verification-batch-prompt.txt")


def format_claim_verification_batch_prompt(
    claims: list[tuple[int, str]],
    documents: list[str] | list[dict[str, str]],
    max_docs_chars: int = 100000,
    max_chars_per_doc: int = 10000,
) -> str:
    """Build a batch verification prompt for multiple claims.

    Args:
        claims: List of (claim_index, claim_text) tuples.
        documents: List of source document strings or dicts.
        max_docs_chars: Total character budget for all documents.
        max_chars_per_doc: Fairness cap per document, applied only when the
            corpus does not fit in ``max_docs_chars``.

    Returns:
        Formatted prompt string.
    """
    formatted_docs = format_documents(
        documents,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_docs_chars,
    )
    claims_text = "\n".join(
        f"  Claim {idx}: {text}"
        for idx, text in claims
    )
    return CLAIM_VERIFICATION_BATCH_PROMPT.format(
        system_prompt=CLAIM_VERIFICATION_BATCH_SYSTEM,
        documents=formatted_docs,
        claims=claims_text,
    )


# ---------------------------------------------------------------------------
# Document Formatting
# ---------------------------------------------------------------------------


def format_documents(
    documents: list[str] | list[dict[str, str]],
    max_chars_per_doc: int = 10000,
    max_total_chars: int = 100000,
) -> str:
    """Format a list of documents into a single string for the LLM.

    Args:
        documents: List of document strings, or list of dicts with
            ``{"text": ..., "title": ...}`` entries. When dicts are
            provided, the title is included as a header.
        max_chars_per_doc: Fairness cap per document. Only applied when the
            corpus as a whole does not fit in ``max_total_chars`` — a source
            that fits inside the total budget is passed whole, because cutting
            it hides evidence the answer was written from.
        max_total_chars: Maximum total characters across all documents.

    Returns:
        Formatted documents string.
    """
    if not documents:
        return "(No source documents provided)"

    texts = [doc["text"] if isinstance(doc, dict) else doc for doc in documents]
    # The per-document cap exists so that one huge source cannot starve the
    # others. When the whole corpus already fits the total budget, that guard
    # is unnecessary and would only hide evidence.
    corpus_chars = sum(len(t) for t in texts)
    per_doc_limit = max_chars_per_doc if corpus_chars > max_total_chars else None

    parts = []
    total_chars = 0

    for i, (doc, text) in enumerate(zip(documents, texts, strict=True)):
        if total_chars >= max_total_chars:
            parts.append("\n[Remaining documents truncated to fit context window]")
            break

        if isinstance(doc, dict):
            title = doc.get("title")
            header = f"Document {i + 1}: {title}" if title else f"Document {i + 1}:"
        else:
            header = f"Document {i + 1}:"

        # Truncate individual document, but only when the corpus overflows
        if per_doc_limit is not None and len(text) > per_doc_limit:
            text = text[:per_doc_limit] + "... [truncated]"

        remaining = max_total_chars - total_chars
        if len(text) > remaining:
            text = text[:remaining] + "... [truncated]"

        parts.append(f"{header}\n{text}\n")
        total_chars += len(text)

    return "\n".join(parts)
