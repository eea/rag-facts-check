"""Tests for span matching utilities."""

from rag_facts_check.spans import find_evidence_span, find_span_in_text


class TestFindSpanInText:
    """Tests for find_span_in_text."""

    def test_exact_match(self):
        text = "Paris is the capital of France."
        needle = "Paris is the capital"
        result = find_span_in_text(needle, text)
        assert result == (0, len(needle))

    def test_exact_match_middle(self):
        text = "Hello, Paris is the capital of France. Goodbye."
        needle = "Paris is the capital"
        result = find_span_in_text(needle, text)
        assert result is not None
        start, end = result
        assert text[start:end] == needle

    def test_no_match(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = find_span_in_text("Paris is the capital of France", text)
        assert result is None

    def test_empty_needle(self):
        result = find_span_in_text("", "Some text")
        assert result is None

    def test_empty_haystack(self):
        result = find_span_in_text("Some text", "")
        assert result is None

    def test_fuzzy_match_minor_diff(self):
        """Fuzzy match handles minor differences like articles."""
        text = "The total phosphorus in lakes has fallen steadily."
        needle = "Total phosphorus in lakes has fallen"
        result = find_span_in_text(needle, text)
        assert result is not None
        start, end = result
        matched = text[start:end]
        # Should capture the core phrase
        assert "phosphorus" in matched.lower()
        assert "fallen" in matched.lower()

    def test_returns_correct_offsets(self):
        """Verify returned offsets actually point to the matched text."""
        text = "ABC DEF GHI"
        needle = "DEF"
        result = find_span_in_text(needle, text)
        assert result == (4, 7)
        assert text[4:7] == "DEF"


class TestFindEvidenceSpan:
    """Tests for find_evidence_span."""

    def test_exact_match_string_docs(self):
        docs = ["Paris is the capital of France."]
        needle = "Paris is the capital"
        result = find_evidence_span(needle, docs)
        assert result is not None
        doc_id, start, end = result
        assert doc_id == "doc_1"
        assert docs[0][start:end] == needle

    def test_exact_match_dict_docs(self):
        docs = [{"doc_id": "my-doc", "text": "Paris is the capital of France."}]
        needle = "Paris is the capital"
        result = find_evidence_span(needle, docs)
        assert result is not None
        doc_id, start, end = result
        assert doc_id == "my-doc"
        assert docs[0]["text"][start:end] == needle

    def test_na_evidence(self):
        docs = ["Some document text."]
        result = find_evidence_span("N/A", docs)
        assert result is None

    def test_empty_evidence(self):
        docs = ["Some document text."]
        result = find_evidence_span("", docs)
        assert result is None

    def test_multi_doc_search(self):
        docs = [
            {"doc_id": "doc_1", "text": "First document text."},
            {"doc_id": "doc_2", "text": "Second document with evidence."},
        ]
        result = find_evidence_span("with evidence", docs)
        assert result is not None
        doc_id, start, end = result
        assert doc_id == "doc_2"

    def test_not_found(self):
        docs = ["Paris is the capital of France."]
        result = find_evidence_span("Berlin is the capital of Germany", docs)
        assert result is None

    def test_returns_valid_offsets(self):
        """Verify returned offsets point to actual text in the document."""
        docs = [{"doc_id": "d1", "text": "The Eiffel Tower was built in 1889."}]
        needle = "Eiffel Tower was built"
        result = find_evidence_span(needle, docs)
        assert result is not None
        doc_id, start, end = result
        assert docs[0]["text"][start:end] == needle
