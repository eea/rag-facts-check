"""
Tests for the EvidenceRetriever and DocumentChunk in rag_facts_check.retriever.

Covers document chunking, lexical retrieval, tokenization, and edge cases.
"""

from rag_facts_check.retriever import DocumentChunk, EvidenceRetriever


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
