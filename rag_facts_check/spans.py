"""
Span matching utilities for grounding claims and evidence in source texts.

Provides fuzzy matching of claim text against the original answer (to find
where in the answer each claim was extracted from) and exact matching of
evidence quotes against source documents (to find character offsets for
highlighting).
"""

import re
from difflib import SequenceMatcher

# Maximum number of evidence quotes accepted per claim. Judges that return more
# are truncated (first quotes win), which bounds prompt/output size and the
# number of citation chips rendered in the UI.
MAX_EVIDENCE_QUOTES = 3

# Placeholder values judges use when they found no evidence.
_NO_EVIDENCE_VALUES = {"", "n/a", "na", "none", "null", "-"}


def normalize_evidence_quotes(raw: object, max_quotes: int = MAX_EVIDENCE_QUOTES) -> list[str]:
    """Coerce whatever the judge returned for ``evidence`` into a clean quote list.

    Judges are asked for a JSON array of verbatim quotes, but models also return
    a bare string, ``null``, ``"N/A"``, or a list of objects. Normalizing here
    keeps a chatty judge from crashing the pipeline (which previously took down
    the whole verification response).

    Args:
        raw: The raw ``evidence`` value from the judge.
        max_quotes: Maximum number of quotes to keep.

    Returns:
        List of non-empty, de-duplicated quote strings (order preserved),
        truncated to *max_quotes*. Empty list means "no evidence".
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        candidates: list[object] = [raw]
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        candidates = [raw]

    quotes: list[str] = []
    seen: set[str] = set()

    for item in candidates:
        if isinstance(item, dict):
            # Some judges return {"quote": ..., "document_index": ...} objects.
            item = item.get("quote") or item.get("evidence") or item.get("text")
        if not isinstance(item, str):
            continue

        quote = item.strip().strip('"\u201c\u201d')
        if quote.lower() in _NO_EVIDENCE_VALUES:
            continue

        key = " ".join(quote.split()).lower()
        if not key or key in seen:
            continue

        seen.add(key)
        quotes.append(quote)
        if len(quotes) >= max_quotes:
            break

    return quotes


def merge_overlapping_spans(
    spans: list[tuple[int, int]],
    max_gap: int = 0,
) -> list[tuple[int, int]]:
    """Merge spans that overlap or touch each other.

    Multiple evidence quotes for one claim can point at the same passage (or at
    adjacent passages). Downstream renderers split text on segment boundaries,
    so overlapping spans duplicate text. Merging keeps one highlight per
    passage.

    Args:
        spans: List of ``(start, end)`` offsets (end exclusive), unordered.
        max_gap: Also merge spans separated by at most this many characters.

    Returns:
        Sorted list of disjoint, non-touching ``(start, end)`` spans.
    """
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


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
        evidence: The evidence quote to find. A list of quotes is tolerated and
            the first quote that matches is returned.
        text: The document text to search in.

    Returns:
        ``(start, end)`` character offsets, or ``None`` if not found.
    """
    if isinstance(evidence, (list, tuple, set)):
        for quote in normalize_evidence_quotes(evidence):
            span = find_evidence_span_in_doc(quote, text)
            if span is not None:
                return span
        return None

    if not evidence or evidence == "N/A":
        return None

    # Strip surrounding quotes and ellipses
    cleaned_ev = evidence.strip().strip("\"'“”«»")
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


def find_evidence_spans(
    quotes: list[str],
    documents: list[str] | list[dict[str, str]],
    preferred_document_index: int | None = None,
) -> list[tuple[str, int, int, int]]:
    """Locate every evidence quote in the source documents.

    For each quote the preferred document (as claimed by the judge) is searched
    first, then all documents. Quotes that cannot be found are skipped.

    Args:
        quotes: Normalized evidence quotes.
        documents: List of document strings or dicts with ``text`` keys.
        preferred_document_index: Document index claimed by the judge for the
            first quote; used as a fast path for every quote.

    Returns:
        List of ``(quote, document_index, start, end)`` tuples, one per located
        quote, in quote order. A quote found in no document produces no entry.
    """
    located: list[tuple[str, int, int, int]] = []

    for quote in quotes:
        # Fast path: the document the judge pointed at.
        if preferred_document_index is not None and 0 <= preferred_document_index < len(documents):
            doc = documents[preferred_document_index]
            doc_text = doc["text"] if isinstance(doc, dict) else doc
            span = find_evidence_span_in_doc(quote, doc_text)
            if span is not None:
                located.append((quote, preferred_document_index, span[0], span[1]))
                continue

        for i, doc in enumerate(documents):
            doc_text = doc["text"] if isinstance(doc, dict) else doc
            span = find_evidence_span_in_doc(quote, doc_text)
            if span is not None:
                located.append((quote, i, span[0], span[1]))
                break

    return located


def find_evidence_span(
    evidence: str,
    documents: list[str] | list[dict[str, str]],
) -> tuple[str | None, int, int] | None:
    """Find the character span of *evidence* in the source documents.

    Searches all documents for the evidence quote. Returns the document
    identifier and character offsets.

    Args:
        evidence: The evidence quote to find (a list of quotes is tolerated).
        documents: List of document strings or dicts with ``doc_id`` and
            ``text`` keys.

    Returns:
        ``(doc_id, start, end)`` or ``None`` if not found.
    """
    if isinstance(evidence, (list, tuple, set)):
        for quote in normalize_evidence_quotes(evidence):
            found = find_evidence_span(quote, documents)
            if found is not None:
                return found
        return None

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
