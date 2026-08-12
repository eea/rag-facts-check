"""
Tests for the EvidenceRetriever, LLMEvidenceRetriever, and DocumentChunk
in rag_facts_check.retriever.

Covers document chunking, lexical retrieval, LLM-based retrieval,
tokenization, and edge cases.
"""

import pytest

from rag_facts_check.retriever import DocumentChunk, EvidenceRetriever, LLMEvidenceRetriever


class TestDocumentChunk:
    """Tests for the DocumentChunk dataclass."""

    def test_chunk_creation(self):
        chunk = DocumentChunk(text="Paris is the capital.", doc_id="doc_1", chunk_id=0)
        assert chunk.text == "Paris is the capital."
        assert chunk.doc_id == "doc_1"
        assert chunk.chunk_id == 0

    def test_chunk_default_fields(self):
        chunk = DocumentChunk(text="Test.", doc_id="doc_1", chunk_id=1)
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.doc_id, str)
        assert isinstance(chunk.chunk_id, int)


class TestEvidenceRetriever:
    """Tests for the EvidenceRetriever class."""

    def test_default_init(self):
        retriever = EvidenceRetriever()
        assert retriever.chunk_size == 200
        assert retriever.top_k == 3
        assert retriever.min_overlap == 1
        assert retriever.stop_words is not None

    def test_custom_init(self):
        retriever = EvidenceRetriever(chunk_size=100, top_k=5, min_overlap=2)
        assert retriever.chunk_size == 100
        assert retriever.top_k == 5
        assert retriever.min_overlap == 2

    def test_default_stop_words(self):
        sw = EvidenceRetriever._default_stop_words()
        assert "the" in sw
        assert "a" in sw
        assert "and" in sw
        assert "paris" not in sw

    def test_chunk_documents_single(self):
        retriever = EvidenceRetriever(chunk_size=100)
        docs = ["Paris is the capital of France. The Eiffel Tower was built in 1889."]
        chunks = retriever.chunk_documents(docs)
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert chunks[0].doc_id == "doc_1"
        assert chunks[0].chunk_id == 0

    def test_chunk_documents_tracks_offsets(self):
        """Chunks should have valid start/end offsets into the original document."""
        retriever = EvidenceRetriever(chunk_size=100)
        doc_text = "Paris is the capital of France. The Eiffel Tower was built in 1889."
        docs = [doc_text]
        chunks = retriever.chunk_documents(docs)
        for chunk in chunks:
            # The chunk's text (without title prefix) should be findable at the offsets
            text_without_prefix = chunk.text
            if text_without_prefix.startswith("Title: "):
                text_without_prefix = text_without_prefix.split(". ", 1)[1]
            assert chunk.start >= 0
            assert chunk.end <= len(doc_text)
            # The chunk text should be a substring of the original
            assert text_without_prefix in doc_text

    def test_chunk_documents_tracks_doc_index(self):
        """Chunks should have the correct 0-based document index."""
        retriever = EvidenceRetriever(chunk_size=100)
        docs = ["First document text.", "Second document text.", "Third document text."]
        chunks = retriever.chunk_documents(docs)
        for chunk in chunks:
            assert chunk.doc_index in (0, 1, 2)
        # Verify each doc_index appears
        indices = sorted(set(c.doc_index for c in chunks))
        assert indices == [0, 1, 2]

    def test_chunk_documents_multiple(self):
        retriever = EvidenceRetriever(chunk_size=100)
        docs = [
            "Paris is the capital of France. The Eiffel Tower was built in 1889.",
            "Berlin is the capital of Germany. Munich is a city in Bavaria.",
        ]
        chunks = retriever.chunk_documents(docs)
        assert len(chunks) >= 2
        doc_ids = [c.doc_id for c in chunks]
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids

    def test_chunk_documents_empty(self):
        retriever = EvidenceRetriever()
        chunks = retriever.chunk_documents([])
        assert chunks == []

    def test_chunk_documents_empty_string(self):
        retriever = EvidenceRetriever()
        chunks = retriever.chunk_documents([""])
        assert chunks == []

    def test_chunk_documents_respects_chunk_size(self):
        """Chunks should not exceed chunk_size words significantly."""
        retriever = EvidenceRetriever(chunk_size=10)
        # Use multiple short sentences so chunking by sentence works
        long_doc = ". ".join(f"word{i}" for i in range(50))
        chunks = retriever.chunk_documents([long_doc])
        for chunk in chunks:
            word_count = len(chunk.text.split())
            assert word_count <= 15  # Allow some slack for sentence boundaries

    def test_retrieve_relevant_chunks(self):
        retriever = EvidenceRetriever(chunk_size=200, top_k=3)
        documents = [
            "Paris is the capital of France. The Eiffel Tower was built in 1889.",
            "Berlin is the capital of Germany. Munich is a city in Bavaria.",
            "London is the capital of England. The Thames River flows through London.",
        ]
        chunks = retriever.chunk_documents(documents)
        relevant = retriever.retrieve("What is the capital of France?", chunks)
        assert len(relevant) <= 3
        # The Paris chunk should be in the results
        assert any("Paris" in c.text for c in relevant)

    def test_retrieve_top_k_limit(self):
        retriever = EvidenceRetriever(chunk_size=200, top_k=1)
        documents = [
            "Paris is the capital of France. The Eiffel Tower was built in 1889.",
            "Berlin is the capital of Germany.",
            "London is the capital of England.",
        ]
        chunks = retriever.chunk_documents(documents)
        relevant = retriever.retrieve("capital of France", chunks)
        assert len(relevant) <= 1

    def test_retrieve_no_matching_claim(self):
        """When no chunks match, should return empty or fallback."""
        retriever = EvidenceRetriever(chunk_size=200, top_k=3)
        documents = [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
        ]
        chunks = retriever.chunk_documents(documents)
        # A claim with only stop words
        relevant = retriever.retrieve("the and of", chunks)
        # Should return empty or fallback (claim_words is empty)
        assert isinstance(relevant, list)

    def test_retrieve_empty_claim(self):
        retriever = EvidenceRetriever()
        chunks = [DocumentChunk(text="Paris is the capital.", doc_id="doc_1", chunk_id=0)]
        relevant = retriever.retrieve("", chunks)
        # Empty claim → claim_words is empty → returns top_k chunks
        assert len(relevant) <= retriever.top_k

    def test_tokenize_basic(self):
        retriever = EvidenceRetriever()
        tokens = retriever._tokenize("Paris is the capital of France")
        assert "paris" in tokens
        assert "capital" in tokens
        assert "france" in tokens
        assert "the" not in tokens  # stop word
        assert "of" not in tokens  # stop word

    def test_tokenize_removes_short_words(self):
        retriever = EvidenceRetriever()
        tokens = retriever._tokenize("A I am in Paris")
        assert "paris" in tokens
        assert "am" not in tokens  # 2 chars, filtered
        assert "in" not in tokens  # stop word

    def test_tokenize_case_insensitive(self):
        retriever = EvidenceRetriever()
        tokens = retriever._tokenize("Paris PARIS paris")
        assert "paris" in tokens
        assert len(tokens) == 1  # All same word

    def test_tokenize_empty_string(self):
        retriever = EvidenceRetriever()
        tokens = retriever._tokenize("")
        assert tokens == set()

    def test_retrieve_with_climate_claim(self):
        """Test retrieval with environmental claims."""
        retriever = EvidenceRetriever(chunk_size=200, top_k=2)
        documents = [
            "In 2023, renewable energy sources accounted for 30% of global electricity generation.",
            "The IPCC projects a 2-4°C temperature rise by 2100.",
            "Battery storage costs have fallen by 89% since 2010.",
        ]
        chunks = retriever.chunk_documents(documents)
        relevant = retriever.retrieve("renewable energy accounted for 30%", chunks)
        assert len(relevant) <= 2
        assert any("30%" in c.text for c in relevant)

    def test_retrieve_sorts_by_relevance(self):
        """Retrieved chunks should be sorted by relevance score."""
        retriever = EvidenceRetriever(chunk_size=200, top_k=3)
        documents = [
            "Paris is the capital of France. The Eiffel Tower was built in 1889.",
            "Berlin is the capital of Germany.",
            "London is the capital of England.",
            "Paris is known for the Louvre Museum.",
        ]
        chunks = retriever.chunk_documents(documents)
        relevant = retriever.retrieve("Paris capital France Eiffel Tower", chunks)
        assert len(relevant) <= 3
        # The most relevant chunk should mention Paris and Eiffel Tower
        if relevant:
            assert "Paris" in relevant[0].text or "Eiffel" in relevant[0].text


class TestLLMEvidenceRetriever:
    """Tests for the LLM-based evidence retriever."""

    def test_init_defaults(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm)
        assert retriever.chunk_size == 1000
        assert retriever.top_k == 5
        assert retriever.llm is mock_llm

    def test_init_custom_params(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm, chunk_size=500, top_k=3)
        assert retriever.chunk_size == 500
        assert retriever.top_k == 3

    @pytest.mark.asyncio
    async def test_retrieve_returns_relevant_chunks(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm, chunk_size=1000)
        documents = [
            "Paris is the capital of France. The Eiffel Tower stands in Paris.",
            "Tokyo is the capital of Japan. Mount Fuji is near Tokyo.",
            "Berlin is the capital of Germany. The Brandenburg Gate is iconic.",
        ]
        chunks = retriever.chunk_documents(documents)

        result = await retriever.retrieve("Paris is the capital", chunks)

        assert isinstance(result, list)
        assert all(isinstance(c, DocumentChunk) for c in result)
        # The LLM should have been called
        assert mock_llm.generate.called

    @pytest.mark.asyncio
    async def test_retrieve_empty_chunks(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm)
        result = await retriever.retrieve("some claim", [])
        assert result == []
        # LLM should NOT be called for empty chunks
        assert not mock_llm.generate.called

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm, top_k=2)
        documents = [f"Document {i} has some content about topic {i}." for i in range(10)]
        chunks = retriever.chunk_documents(documents)

        result = await retriever.retrieve("topic 1", chunks)
        assert len(result) <= 2

    def test_parse_chunk_ids_json_array(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm)
        ids = retriever._parse_chunk_ids("[0, 3, 7]")
        assert ids == [0, 3, 7]

    def test_parse_chunk_ids_spaced(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm)
        ids = retriever._parse_chunk_ids("[ 0 , 3 , 7 ]")
        assert ids == [0, 3, 7]

    def test_parse_chunk_ids_in_prose(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm)
        ids = retriever._parse_chunk_ids("The relevant chunks are: [1, 4]")
        assert ids == [1, 4]

    def test_parse_chunk_ids_empty(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm)
        ids = retriever._parse_chunk_ids("[]")
        assert ids == []

    def test_parse_chunk_ids_malformed_fallback(self, mock_llm):
        retriever = LLMEvidenceRetriever(llm=mock_llm)
        # No JSON array, falls back to extracting all numbers
        ids = retriever._parse_chunk_ids("chunks 3 and 7 are relevant")
        assert 3 in ids
        assert 7 in ids

    @pytest.mark.asyncio
    async def test_fallback_when_llm_returns_nothing(self, mock_llm):
        """When LLM returns empty/no valid IDs, fall back to first top_k chunks."""
        retriever = LLMEvidenceRetriever(llm=mock_llm, top_k=3)
        documents = ["Some document text here."]
        chunks = retriever.chunk_documents(documents)

        # Monkey-patch _parse_chunk_ids to return empty
        original = retriever._parse_chunk_ids
        retriever._parse_chunk_ids = lambda text: []

        result = await retriever.retrieve("unrelated claim", chunks)

        retriever._parse_chunk_ids = original  # restore
        # Should fall back to first top_k chunks
        assert len(result) == 1  # only 1 chunk exists
        assert result[0] is chunks[0]
