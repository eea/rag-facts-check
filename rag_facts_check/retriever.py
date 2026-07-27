"""
Evidence retrieval for per-claim verification.

Instead of passing all documents to every claim verification (which wastes
context and compute), this module splits documents into chunks and retrieves
only the most relevant chunks for each claim.

The default implementation uses simple lexical matching (keyword overlap).
For better accuracy, replace with an embedding-based retriever.
"""

import re
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A chunk of a source document.

    Attributes:
        text: The chunk text.
        doc_id: Identifier of the source document.
        chunk_id: Identifier of the chunk within the document.
    """

    text: str
    doc_id: str
    chunk_id: int


class EvidenceRetriever:
    """Retrieves relevant document chunks for a given claim.

    Uses lexical matching (keyword overlap) by default.  For better
    accuracy, subclass and override :meth:`retrieve` with an
    embedding-based approach.

    Example::

        retriever = EvidenceRetriever(chunk_size=200, top_k=3)
        chunks = retriever.chunk_documents(documents)
        relevant = retriever.retrieve("Paris is the capital of France", chunks)
    """

    def __init__(
        self,
        chunk_size: int = 200,
        top_k: int = 3,
        min_overlap: int = 1,
        stop_words: set | None = None,
    ):
        """Initialize the retriever.

        Args:
            chunk_size: Target number of words per chunk.
            top_k: Number of relevant chunks to return per claim.
            min_overlap: Minimum keyword overlap to consider a chunk relevant.
            stop_words: Set of stop words to ignore in matching.
        """
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.min_overlap = min_overlap
        self.stop_words = stop_words or self._default_stop_words()

    @staticmethod
    def _default_stop_words() -> set:
        """Return a basic set of English stop words."""
        return {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "their",
            "his",
            "her",
            "its",
            "our",
            "your",
            "my",
            "as",
            "than",
            "then",
            "so",
            "if",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "very",
            "just",
            "also",
            "now",
        }

    def chunk_documents(
        self,
        documents: list[str] | list[dict[str, str]],
    ) -> list[DocumentChunk]:
        """Split documents into chunks of approximately *chunk_size* words.

        Args:
            documents: List of document strings, or list of dicts with
                ``{"doc_id": ..., "text": ...}`` entries. When dicts are
                provided, the user-supplied ``doc_id`` is preserved so it
                flows through to verification results.

        Returns:
            List of :class:`DocumentChunk` objects.
        """
        chunks = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                doc_id = doc.get("doc_id", f"doc_{i + 1}")
                text = doc["text"]
            else:
                doc_id = f"doc_{i + 1}"
                text = doc
            doc_chunks = self._chunk_text(text, doc_id)
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_text(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """Split a single document into chunks."""
        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        current_chunk = ""
        chunk_id = 0
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunks.append(
                    DocumentChunk(
                        text=current_chunk.strip(),
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1
                current_chunk = ""
                current_word_count = 0

            current_chunk += sentence + " "
            current_word_count += sentence_words

        if current_chunk.strip():
            chunks.append(
                DocumentChunk(
                    text=current_chunk.strip(),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
            )

        return chunks

    def retrieve(self, claim: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Retrieve the most relevant chunks for a claim.

        Uses keyword overlap (Jaccard-like similarity on non-stop-words).

        Args:
            claim: The claim text.
            chunks: List of document chunks to search.

        Returns:
            Top-*k* most relevant chunks, sorted by relevance score.
        """
        claim_words = self._tokenize(claim)
        if not claim_words:
            return chunks[: self.top_k]

        scored = []
        for chunk in chunks:
            chunk_words = self._tokenize(chunk.text)
            if not chunk_words:
                continue
            overlap = len(claim_words & chunk_words)
            if overlap >= self.min_overlap:
                # Jaccard-like score
                union = len(claim_words | chunk_words)
                score = overlap / union if union > 0 else 0
                scored.append((score, overlap, chunk))

        # Sort by score (then by raw overlap as tiebreaker)
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [chunk for _, _, chunk in scored[: self.top_k]]

    def _tokenize(self, text: str) -> set:
        """Tokenize text into a set of non-stop-word tokens."""
        tokens = set()
        for word in re.findall(r"\b[a-zA-Z]+\b", text.lower()):
            if word not in self.stop_words and len(word) > 2:
                tokens.add(word)
        return tokens
