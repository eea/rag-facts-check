"""
Span matching utilities for grounding claims and evidence in source texts.

Provides fuzzy matching of claim text against the original answer (to find
where in the answer each claim was extracted from) and exact matching of
evidence quotes against source documents (to find character offsets for
highlighting).
"""

from difflib import SequenceMatcher


def find_span_in_text(needle: str, haystack: str) -> tuple[int, int] | None:
    """Find the best matching span of *needle* in *haystack*.

    Uses exact matching first, then falls back to sequence matching for
    minor paraphrasing and punctuation differences.

    Args:
        needle: The text to find (e.g., an extracted claim).
        haystack: The text to search in (e.g., the original answer).

    Returns:
        ``(start, end)`` character offsets of the best match, or ``None``
        if no match above the similarity threshold.
    """
    if not needle or not haystack:
        return None

    # Try exact match first
    exact = haystack.find(needle)
    if exact >= 0:
        return (exact, exact + len(needle))

    # Fuzzy match: find the best local alignment
    # Use SequenceMatcher to find the best matching block
    sm = SequenceMatcher(None, needle, haystack, autojunk=False)
    best = sm.find_longest_match(0, len(needle), 0, len(haystack))

    if best.size == 0:
        return None

    # Check if the match covers enough of the needle
    match_ratio = best.size / len(needle)
    if match_ratio < 0.85:
        return None

    # Expand the match to include surrounding context from the needle
    start = best.b
    end = best.b + best.size

    # Try to expand left and right to cover more of the needle
    needle_start = best.a
    needle_end = best.a + best.size

    # Include the full matched region plus any gaps
    if needle_start > 0 and start > 0:
        # Expand left to include preceding text from haystack
        while start > 0 and needle_start > 0:
            start -= 1
            needle_start -= 1
    if needle_end < len(needle) and end < len(haystack):
        while end < len(haystack) and needle_end < len(needle):
            end += 1
            needle_end += 1

    return (start, end)


def find_evidence_span_in_doc(
    evidence: str,
    text: str,
) -> tuple[int, int] | None:
    """Find the character span of *evidence* in a single document text.

    Args:
        evidence: The evidence quote to find.
        text: The document text to search in.

    Returns:
        ``(start, end)`` character offsets, or ``None`` if not found.
    """
    if not evidence or evidence == "N/A":
        return None

    # Try exact match first
    exact = text.find(evidence)
    if exact >= 0:
        return (exact, exact + len(evidence))

    # Try fuzzy match
    span = find_span_in_text(evidence, text)
    return span


def find_evidence_span(
    evidence: str,
    documents: list[str] | list[dict[str, str]],
) -> tuple[str | None, int, int] | None:
    """Find the character span of *evidence* in the source documents.

    Searches all documents for the evidence quote. Returns the document
    identifier and character offsets.

    Args:
        evidence: The evidence quote to find.
        documents: List of document strings or dicts with ``doc_id`` and
            ``text`` keys.

    Returns:
        ``(doc_id, start, end)`` or ``None`` if not found.
    """
    if not evidence or evidence == "N/A":
        return None

    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            doc_id = doc.get("doc_id", f"doc_{i + 1}")
            text = doc["text"]
        else:
            doc_id = f"doc_{i + 1}"
            text = doc

        # Try exact match first
        exact = text.find(evidence)
        if exact >= 0:
            return (doc_id, exact, exact + len(evidence))

        # Try fuzzy match
        span = find_span_in_text(evidence, text)
        if span is not None:
            return (doc_id, span[0], span[1])

    return None
