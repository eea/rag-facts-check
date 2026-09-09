"""
Span matching utilities for grounding claims and evidence in source texts.

Provides fuzzy matching of claim text against the original answer (to find
where in the answer each claim was extracted from) and exact matching of
evidence quotes against source documents (to find character offsets for
highlighting).
"""

import re
from difflib import SequenceMatcher


def find_span_in_text(needle: str, haystack: str) -> tuple[int, int] | None:
    """Find the best matching span of *needle* in *haystack*.

    Uses exact matching first, then whitespace-flexible matching,
    and falls back to sequence matching for minor paraphrasing and
    punctuation differences.

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

    # Whitespace-flexible regex match
    needle_words = re.findall(r"\S+", needle)
    if len(needle_words) >= 2:
        pattern = r"\s+".join(re.escape(w) for w in needle_words)
        try:
            m = re.search(pattern, haystack)
            if m:
                return (m.start(), m.end())
        except re.error:
            pass

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

    Tolerates quotation marks, ellipses, whitespace/newline differences,
    and minor wording variations often introduced by LLMs when quoting.

    Args:
        evidence: The evidence quote to find.
        text: The document text to search in.

    Returns:
        ``(start, end)`` character offsets, or ``None`` if not found.
    """
    if not evidence or evidence == "N/A":
        return None

    # Strip surrounding quotes and ellipses
    cleaned_ev = evidence.strip().strip('"\'“”«»')
    cleaned_ev = re.sub(r"^\.\.\.|\.\.\.$|^…|…$", "", cleaned_ev).strip()
    if not cleaned_ev:
        return None

    # 1. Exact match on raw and cleaned evidence
    exact = text.find(evidence)
    if exact >= 0:
        return (exact, exact + len(evidence))
    exact = text.find(cleaned_ev)
    if exact >= 0:
        return (exact, exact + len(cleaned_ev))

    # 2. Whitespace-flexible regex match (handles newlines, double spaces, tabs)
    words = re.findall(r"\S+", cleaned_ev)
    if len(words) >= 2:
        pattern = r"\s+".join(re.escape(w) for w in words)
        try:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return (m.start(), m.end())
        except re.error:
            pass

    # 3. Punctuation & whitespace flexible match (alphanumeric words only)
    alnum_words = [re.sub(r"[^\w]", "", w) for w in words]
    alnum_words = [w for w in alnum_words if w]
    if len(alnum_words) >= 2:
        pattern = r"[\W_]+".join(re.escape(w) for w in alnum_words)
        try:
            m = re.search(r"\b" + pattern + r"\b", text, flags=re.IGNORECASE)
            if m:
                return (m.start(), m.end())
        except re.error:
            pass

    # 4. Fall back to fuzzy character matching
    return find_span_in_text(cleaned_ev, text)


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

        span = find_evidence_span_in_doc(evidence, text)
        if span is not None:
            return (doc_id, span[0], span[1])

    return None
