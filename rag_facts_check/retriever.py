"""
Evidence retrieval for per-claim verification.

Instead of passing all documents to every claim verification (which wastes
context and compute), this module splits documents into chunks and retrieves
only the most relevant chunks for each claim.

Two retrieval strategies are available:
- ``EvidenceRetriever`` — simple lexical matching (keyword overlap)
- ``LLMEvidenceRetriever`` — uses the LLM itself to select relevant chunks,
  providing semantic understanding rather than surface-level word matching
"""

import json
import re
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A chunk of a source document.

    Attributes:
        text: The chunk text (raw, no title prefix — title is carried
            separately so span offsets remain accurate).
        doc_id: Identifier of the source document.
        doc_index: 0-based index of the source document in the original list.
        chunk_id: Identifier of the chunk within the document.
        start: Start character offset within the original document text.
        end: End character offset within the original document text.
        title: Document title (for LLM context, not mixed into text).
    """

    text: str
    doc_id: str
    doc_index: int = 0
    chunk_id: int = 0
    start: int = 0
    end: int = 0
    title: str | None = None


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
                ``{"doc_id": ..., "text": ...}`` entries. Optional keys:
                ``title`` (prepended to each chunk for context) and
                ``doc_id`` (preserved through to verification results).

        Returns:
            List of :class:`DocumentChunk` objects.
        """
        chunks = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                doc_id = doc.get("doc_id", f"doc_{i + 1}")
                text = doc["text"]
                title = doc.get("title")
            else:
                doc_id = f"doc_{i + 1}"
                text = doc
                title = None
            doc_chunks = self._chunk_text(text, doc_id, doc_index=i, title=title)
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_text(
        self, text: str, doc_id: str, doc_index: int = 0, title: str | None = None
    ) -> list[DocumentChunk]:
        """Split a single document into chunks.

        Args:
            text: Document text.
            doc_id: Document identifier.
            doc_index: 0-based index of this document in the original list.
            title: Document title (stored separately, not mixed into text).
        """
        # Guard against empty text
        if not text.strip():
            return []

        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        current_sentences: list[str] = []
        current_word_count = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > self.chunk_size and current_sentences:
                # Flush current chunk
                chunk_text = " ".join(current_sentences)
                # Find this chunk's text in the original document for offsets
                start = text.find(chunk_text)
                end = start + len(chunk_text) if start >= 0 else 0
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_index=doc_index,
                        chunk_id=chunk_id,
                        start=start if start >= 0 else 0,
                        end=end,
                        title=title,
                    )
                )
                chunk_id += 1
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sentence_words

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start = text.find(chunk_text)
            end = start + len(chunk_text) if start >= 0 else 0
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    doc_index=doc_index,
                    chunk_id=chunk_id,
                    start=start if start >= 0 else 0,
                    end=end,
                    title=title,
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


class LLMEvidenceRetriever(EvidenceRetriever):
    """Retrieves relevant document chunks using the LLM itself.

    Unlike the keyword-based :class:`EvidenceRetriever`, this class uses
    the LLM to judge semantic relevance. For each claim, it sends all
    chunks to the LLM and asks it to return the IDs of relevant chunks.

    This is more accurate than keyword matching because the LLM understands
    paraphrases, synonyms, and domain-specific terminology (e.g., it knows
    that "cap-and-trade" and "emissions allowance trading" refer to the
    same concept).

    Uses larger chunks (default 1000 words) so the LLM has enough context
    to judge relevance without missing cross-sentence connections.

    Example::

        retriever = LLMEvidenceRetriever(llm=llm, chunk_size=1000)
        chunks = retriever.chunk_documents(documents)
        relevant = await retriever.retrieve("EU has a carbon trading system", chunks)
    """

    def __init__(
        self,
        llm,
        chunk_size: int = 1000,
        top_k: int = 5,
    ):
        """Initialize the LLM-based retriever.

        Args:
            llm: LLM backend implementing the :class:`LLM` interface
                (must support ``async generate()``).
            chunk_size: Target number of words per chunk. Larger chunks
                give the LLM more context for relevance judgment.
            top_k: Maximum number of chunks to return. The LLM may return
                fewer if fewer chunks are relevant.
        """
        super().__init__(chunk_size=chunk_size, top_k=top_k)
        self.llm = llm

    async def retrieve(
        self, claim: str, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Use the LLM to select relevant chunks for a claim.

        Args:
            claim: The claim text.
            chunks: List of document chunks to search.

        Returns:
            Chunks the LLM deemed relevant, up to ``top_k``.
        """
        if not chunks:
            return []

        # Build chunk descriptors for the prompt
        chunk_dicts = [
            {
                "id": i,
                "title": chunk.title,
                "text": chunk.text,
            }
            for i, chunk in enumerate(chunks)
        ]

        # Build and send the retrieval prompt
        from .prompts import format_evidence_retrieval_prompt

        prompt = format_evidence_retrieval_prompt(claim, chunk_dicts)
        response = await self.llm.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
        )

        # Parse the LLM's response as a JSON array of chunk IDs
        selected_ids = self._parse_chunk_ids(response)

        # Map IDs back to chunks, respecting top_k
        id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
        result = [
            id_to_chunk[iid]
            for iid in selected_ids[: self.top_k]
            if iid in id_to_chunk
        ]

        # Fallback: if the LLM returned nothing, return the first top_k chunks
        if not result:
            result = chunks[: self.top_k]

        return result

    def _parse_chunk_ids(self, text: str) -> list[int]:
        """Parse chunk IDs from the LLM's JSON array response.

        Handles various formats: pure JSON, JSON in code fences,
        JSON embedded in prose, etc.
        """
        # Try direct JSON parse first
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [int(x) for x in parsed if isinstance(x, (int, float))]
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find a JSON array in the text
        match = re.search(r"\[\s*(?:\d+\s*,?\s*)*\]", text)
        if match:
            try:
                parsed = json.loads(match.group())
                return [int(x) for x in parsed if isinstance(x, (int, float))]
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: extract all integers from the text
        numbers = re.findall(r"\b(\d+)\b", text)
        return [int(n) for n in numbers]
