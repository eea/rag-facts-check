"""Tests for span matching utilities."""

from rag_facts_check.spans import (
    find_evidence_span,
    find_evidence_span_in_doc,
    find_span_in_text,
)


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

    def test_whitespace_flexible_newlines(self):
        """Needle with spaces matches text containing line breaks."""
        text = "The EU is largely on track\nfor 2030 climate targets."
        needle = "The EU is largely on track for 2030"
        result = find_span_in_text(needle, text)
        assert result is not None
        assert result == (0, 35)
        assert text[result[0]:result[1]] == "The EU is largely on track\nfor 2030"

    def test_whitespace_flexible_tabs_and_extra_spaces(self):
        """Needle matches text containing multiple spaces and tabs."""
        text = "Greenhouse gas emissions   \t  reduced significantly."
        needle = "Greenhouse gas emissions reduced significantly."
        result = find_span_in_text(needle, text)
        assert result is not None
        assert text[result[0]:result[1]] == text


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


class TestFindEvidenceSpanInDoc:
    """Tests for find_evidence_span_in_doc (targeted single-doc search)."""

    def test_exact_match(self):
        text = "Paris is the capital of France."
        result = find_evidence_span_in_doc("Paris is the capital", text)
        assert result == (0, 20)  # len("Paris is the capital") == 20

    def test_no_match(self):
        text = "Paris is the capital of France."
        result = find_evidence_span_in_doc("Berlin is the capital", text)
        assert result is None

    def test_na_evidence(self):
        result = find_evidence_span_in_doc("N/A", "Some text")
        assert result is None

    def test_empty_evidence(self):
        result = find_evidence_span_in_doc("", "Some text")
        assert result is None

    def test_fuzzy_match(self):
        text = "The total phosphorus in lakes has fallen steadily."
        result = find_evidence_span_in_doc("Total phosphorus in lakes has fallen", text)
        assert result is not None

    def test_quoted_evidence_stripping(self):
        """Evidence quotes with straight and curly quotes are stripped and matched."""
        text = "The European Climate Law sets a binding net-zero target by 2050."
        assert find_evidence_span_in_doc('"binding net-zero target"', text) is not None
        assert find_evidence_span_in_doc('“binding net-zero target”', text) is not None
        assert find_evidence_span_in_doc("'binding net-zero target'", text) is not None

    def test_evidence_with_ellipses(self):
        """Evidence with leading/trailing ellipses matches core text."""
        text = "The European Climate Law sets a binding net-zero target by 2050 at latest."
        span = find_evidence_span_in_doc('... binding net-zero target by 2050 ...', text)
        assert span is not None
        assert text[span[0]:span[1]] == "binding net-zero target by 2050"

    def test_evidence_across_newlines(self):
        """Handles documents with hard linebreaks from PDF or HTML formatting."""
        text = (
            "The EU is largely on track\n"
            " to meet the agreed targets \n"
            "for 2030 if full implementation \n"
            "of policies is ensured."
        )
        quote = (
            "The EU is largely on track to meet the agreed targets "
            "for 2030 if full implementation of policies is ensured"
        )
        span = find_evidence_span_in_doc(quote, text)
        assert span is not None
        assert text[span[0]:span[1]] == (
            "The EU is largely on track\n"
            " to meet the agreed targets \n"
            "for 2030 if full implementation \n"
            "of policies is ensured"
        )

    def test_evidence_punctuation_variations(self):
        """Handles word-boundary matching across minor punctuation variations."""
        text = "A cap-and-trade system for power & industry operates across Europe."
        quote = "cap and trade system for power industry"
        span = find_evidence_span_in_doc(quote, text)
        assert span is not None
        assert "cap-and-trade system for power & industry" in text[span[0]:span[1]]

    def test_evidence_only_quotes_or_ellipses(self):
        """Evidence containing only quote marks or ellipses returns None safely."""
        text = "Some document text."
        assert find_evidence_span_in_doc('""', text) is None
        assert find_evidence_span_in_doc('...', text) is None
        assert find_evidence_span_in_doc('“…”', text) is None
