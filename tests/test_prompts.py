"""
Tests for prompt formatting and the source character budgets.

The budgets matter because the chatbot now sends real Onyx chunk text — a few
thousand characters per document, sometimes over 20k. Silently cutting a source
at a fixed per-document limit hides the very passage a claim would have been
matched against, which shows up as a false `not_enough_info` verdict.
"""

from rag_facts_check.prompts import (
    format_claim_verification_batch_prompt,
    format_claim_verification_evidence_first_prompt,
    format_claim_verification_prompt,
    format_documents,
)


class TestFormatDocuments:
    """Document formatting and truncation rules."""

    def test_empty_documents(self):
        assert format_documents([]) == "(No source documents provided)"

    def test_plain_string_documents_get_headers(self):
        out = format_documents(["first source", "second source"])
        assert "Document 1:\nfirst source" in out
        assert "Document 2:\nsecond source" in out

    def test_dict_documents_keep_title_headers_and_raw_text(self):
        text = "Paris is the capital of France."
        out = format_documents([{"text": text, "title": "France"}])
        assert "Document 1: France" in out
        assert text in out

    def test_long_sources_pass_whole_when_corpus_fits_budget(self):
        """The per-document cap must not bite when everything fits the total.

        This is the case the chatbot now produces: four Onyx documents of
        8k-12k characters each, 40k total, well inside the 100k budget.
        """
        docs = ["a" * 12074, "b" * 11473, "c" * 8622, "d" * 7904]
        out = format_documents(docs, max_chars_per_doc=10000, max_total_chars=100000)
        assert "[truncated]" not in out
        for doc in docs:
            assert doc in out

    def test_one_large_source_passes_when_total_fits(self):
        out = format_documents(["x" * 90000], max_chars_per_doc=10000, max_total_chars=100000)
        assert "[truncated]" not in out
        assert "x" * 90000 in out

    def test_per_doc_cap_applies_when_corpus_overflows_budget(self):
        """Fairness guard: when the corpus cannot fit, no single source may
        starve the others."""
        docs = ["a" * 40000, "b" * 40000, "c" * 40000]
        out = format_documents(docs, max_chars_per_doc=10000, max_total_chars=100000)
        assert "[truncated]" in out
        assert "a" * 10000 in out
        assert "a" * 10001 not in out

    def test_total_budget_drops_remaining_documents(self):
        docs = ["d" * 10000 for _ in range(20)]
        out = format_documents(docs, max_chars_per_doc=10000, max_total_chars=100000)
        assert "[Remaining documents truncated to fit context window]" in out
        assert out.count("Document ") == 10

    def test_dict_sources_are_not_reordered(self):
        docs = [
            {"text": "one", "title": "First"},
            {"text": "two", "title": "Second"},
            {"text": "three", "title": "Third"},
        ]
        out = format_documents(docs)
        assert (
            out.index("Document 1: First")
            < out.index("Document 2: Second")
            < out.index("Document 3: Third")
        )


class TestVerificationPromptBudgets:
    """The checker's budget knobs must reach the prompt, not be silently
    replaced by format_documents defaults."""

    docs = ["z" * 12000]

    def test_standard_prompt_respects_budget(self):
        prompt = format_claim_verification_prompt(
            "claim", self.docs, max_docs_chars=100000, max_chars_per_doc=10000
        )
        assert "z" * 12000 in prompt

        tight = format_claim_verification_prompt(
            "claim", self.docs, max_docs_chars=1000, max_chars_per_doc=500
        )
        assert "z" * 12000 not in tight
        assert "z" * 500 in tight

    def test_evidence_first_prompt_respects_budget(self):
        prompt = format_claim_verification_evidence_first_prompt(
            "claim", self.docs, max_docs_chars=100000, max_chars_per_doc=10000
        )
        assert "z" * 12000 in prompt

        tight = format_claim_verification_evidence_first_prompt(
            "claim", self.docs, max_docs_chars=1000, max_chars_per_doc=500
        )
        assert "z" * 12000 not in tight

    def test_batch_prompt_respects_budget(self):
        claims = [(1, "claim one"), (2, "claim two")]
        prompt = format_claim_verification_batch_prompt(
            claims, self.docs, max_docs_chars=100000, max_chars_per_doc=10000
        )
        assert "z" * 12000 in prompt
        assert "Claim 1: claim one" in prompt

        tight = format_claim_verification_batch_prompt(
            claims, self.docs, max_docs_chars=1000, max_chars_per_doc=500
        )
        assert "z" * 12000 not in tight
