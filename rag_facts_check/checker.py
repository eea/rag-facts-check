"""
Core fact-checking pipeline: claim extraction, per-claim verification,
and result aggregation.

Enhancements over the base implementation:
- Evidence retrieval per claim (reduces context, improves accuracy)
- Evidence-first multi-step prompting (reduces hallucinated evaluations)
- Self-consistency (multiple verification runs with different temperatures)
- Multi-dimensional scoring (groundedness, contradiction_rate, etc.)
- Span-level verification (document_id and chunk_id in results)
"""

import json
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomic_agents import AtomicAgent

from .llm import LLM
from .models import CheckReport, Claim, Span, VerificationResult
from .prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_VERIFICATION_BATCH_SYSTEM,
    CLAIM_VERIFICATION_EVIDENCE_FIRST_SYSTEM,
    CLAIM_VERIFICATION_SYSTEM,
    format_claim_extraction_prompt,
    format_claim_verification_batch_prompt,
    format_claim_verification_evidence_first_prompt,
    format_claim_verification_prompt,
    format_documents,
)
from .retriever import DocumentChunk, EvidenceRetriever
from .spans import find_evidence_span, find_evidence_span_in_doc, find_span_in_text

log = logging.getLogger("rag_facts_check")

# Max retry rounds for claim extraction refinement
_MAX_EXTRACTION_RETRIES = 3


class ClaimExtractor:
    """Extracts atomic factual claims from a RAG-generated answer.

    Uses an LLM to parse the answer and identify individual verifiable
    statements. Supports multi-turn refinement: if a claim's original_text
    cannot be located in the answer, the extractor asks the LLM to fix it.

    When an atomic-agents extraction_agent is provided, uses structured
    output with automatic retries for reliable JSON parsing.
    """

    def __init__(
        self,
        llm: LLM,
        max_new_tokens: int = 2048,
        extraction_agent: "AtomicAgent | None" = None,
    ):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.extraction_agent = extraction_agent

    async def extract(self, answer: str) -> list[Claim]:
        """Extract factual claims from *answer*.

        Uses multi-turn refinement: claims whose original_text cannot be
        found in the answer are sent back to the LLM for correction.

        When an extraction_agent is available, uses structured output
        with automatic retries for reliable JSON parsing.

        Args:
            answer: The RAG-generated answer text.

        Returns:
            List of :class:`Claim` objects.
        """
        if not answer or not answer.strip():
            return []

        # Round 1: initial extraction
        if self.extraction_agent is not None:
            raw_claims = await self._extract_with_agent(answer)
        else:
            prompt = format_claim_extraction_prompt(answer)
            response = await self.llm.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
            )
            log.debug("extract: LLM response (%d chars): %s", len(response), response[:500])
            raw_claims = self._parse_extraction_response(response)

        # Multi-turn refinement: fix claims whose original_text isn't in answer
        for attempt in range(1, _MAX_EXTRACTION_RETRIES + 1):
            matched, unmatched = self._split_by_span(raw_claims, answer)
            if not unmatched:
                break  # all claims have matchable original_text

            log.info(
                "extract: %d/%d claims need refinement (attempt %d)",
                len(unmatched),
                len(raw_claims),
                attempt,
            )
            refined = await self._refine_claims(answer, unmatched)
            # Merge: keep matched, replace unmatched with refined
            raw_claims = matched + refined

        claims = self._to_claim_objects(raw_claims)

        # Deduplicate claims that map to the same span in the answer.
        # The LLM sometimes extracts the same fact twice with different
        # phrasing but identical original_text → same span.
        # Only dedup claims with valid spans (span=None claims are kept).
        seen_spans: set[tuple[int, int]] = set()
        deduped: list[Claim] = []
        for claim in claims:
            if claim.span:
                span_key = (claim.span.start, claim.span.end)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    deduped.append(claim)
            else:
                deduped.append(claim)
        if len(deduped) < len(claims):
            log.info(
                "extract: deduplicated %d → %d claims (removed %d overlapping spans)",
                len(claims),
                len(deduped),
                len(claims) - len(deduped),
            )
        claims = deduped

        log.info("extract: %d claims extracted", len(claims))
        return claims

    def _parse_extraction_response(self, response: str) -> list[dict[str, str]]:
        """Parse LLM response into list of {claim, original_text} dicts.

        Handles both JSON output (structured) and legacy CLAIM N: format.
        """
        # Try JSON first
        claims = self._try_parse_json(response)
        if claims:
            return claims

        # Fall back to legacy CLAIM N: format
        return self._parse_legacy_format(response)

    def _try_parse_json(self, response: str) -> list[dict[str, str]]:
        """Try to parse response as JSON with claims array."""
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

        claims = []
        items = data.get("claims", []) if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict):
                claim_text = item.get("claim", "")
                original = item.get("original_text", item.get("original", ""))
                if claim_text:
                    claims.append({"claim": claim_text, "original_text": original or claim_text})
        return claims

    def _parse_legacy_format(self, response: str) -> list[dict[str, str]]:
        """Parse legacy CLAIM N: format (no original_text available)."""
        claims = []
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            match = re.match(r"^CLAIM\s+(\d+):\s*(.+)$", line, re.IGNORECASE)
            if match:
                text = match.group(2).strip()
                if text:
                    claims.append({"claim": text, "original_text": text})
        # If no claims parsed, try whole response as single claim
        if not claims and response.strip() and "NO CLAIMS" not in response.upper():
            claims.append({"claim": response.strip(), "original_text": response.strip()})
        return claims

    def _split_by_span(
        self, raw: list[dict[str, str]], answer: str
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Split claims into those whose original_text is found in answer and those that aren't."""
        matched, unmatched = [], []
        for item in raw:
            orig = item.get("original_text", "")
            if orig and self._find_in_answer(orig, answer) is not None:
                matched.append(item)
            else:
                unmatched.append(item)
        return matched, unmatched

    def _find_in_answer(self, text: str, answer: str) -> tuple[int, int] | None:
        """Find text in answer, returning (start, end) or None."""
        return find_span_in_text(text, answer)

    async def _extract_with_agent(self, answer: str) -> list[dict[str, str]]:
        """Extract claims using the atomic agent with structured output."""
        from .agents import ClaimExtractionInput

        self.extraction_agent.reset_history()
        input_schema = ClaimExtractionInput(answer=answer)
        result = await self.extraction_agent.run_async(input_schema)

        log.debug("extract: agent returned %d claims", len(result.claims) if result.claims else 0)
        return [
            {
                "claim": c.claim,
                "original_text": c.original_text or c.claim,
            }
            for c in (result.claims or [])
        ]

    async def _refine_claims(
        self, answer: str, unmatched: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Ask the LLM to fix claims whose original_text wasn't found in the answer.

        Sends the original answer + the problematic claims + feedback about
        which original_text values failed, and asks the LLM to redo them
        with exact verbatim text from the answer.
        """
        # Build list of problematic claims with feedback
        problems = []
        for i, item in enumerate(unmatched, 1):
            problems.append(
                f'  {i}. claim="{item["claim"]}" '
                f'original_text="{item.get("original_text", "")}" '
                f"(NOT FOUND in answer)"
            )
        problems_text = "\n".join(problems)

        refinement_prompt = (
            f"{CLAIM_EXTRACTION_SYSTEM}\n\n"
            f"You previously extracted these claims, but their original_text "
            f"fragments were NOT found in the answer. The LLM likely paraphrased "
            f"the original_text instead of quoting it verbatim.\n\n"
            f"Problematic claims:\n{problems_text}\n\n"
            f"Please re-extract ONLY these claims. For each, provide:\n"
            f"- claim: the factual statement (rephrasing OK)\n"
            f"- original_text: the EXACT verbatim text from the answer below. "
            f"It MUST appear word-for-word in the answer.\n\n"
            f"Answer:\n{answer}\n\n"
            f"Return JSON with claims array and has_claims boolean."
        )

        response = await self.llm.generate(
            refinement_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
        )

        log.debug("extract refine: LLM response (%d chars): %s", len(response), response[:500])
        return self._parse_extraction_response(response)

    def _to_claim_objects(self, raw: list[dict[str, str]]) -> list[Claim]:
        """Convert raw dicts to Claim objects."""
        return [
            Claim(
                text=item["claim"],
                index=i + 1,
                original_text=item.get("original_text", item["claim"]),
            )
            for i, item in enumerate(raw)
        ]

    def _parse_claims(self, response: str) -> list[Claim]:
        """Legacy parser kept for backward compatibility."""
        raw = self._parse_legacy_format(response)
        return self._to_claim_objects(raw)


class ClaimVerifier:
    """Verifies individual claims against a set of source documents.

    Supports:
    - Evidence-first multi-step prompting
    - Self-consistency (multiple runs with different temperatures)
    - Evidence retrieval (only relevant document chunks are passed)
    - Structured output via atomic-agents (when verification_agent is provided)
    """

    def __init__(
        self,
        llm: LLM,
        max_new_tokens: int = 512,
        max_docs_chars: int = 100000,
        max_chars_per_doc: int = 10000,
        num_consistency_runs: int = 1,
        evidence_first: bool = True,
        verification_agent: "AtomicAgent | None" = None,
        batch_size: int = 1,
    ):
        """Initialize the claim verifier.

        Args:
            llm: LLM backend implementing the :class:`LLM` interface.
            max_new_tokens: Max tokens for LLM generation.
            max_docs_chars: Maximum total characters of documents to include.
            max_chars_per_doc: Maximum characters per individual document.
            num_consistency_runs: Number of verification runs for self-consistency.
                If >1, runs verification multiple times with increasing temperatures
                and aggregates via majority vote.
            evidence_first: If True, use the evidence-first multi-step prompt.
            verification_agent: Optional atomic-agents agent for structured output.
            batch_size: Number of claims to verify in a single LLM call.
                When >1, claims are grouped into batches and verified together,
                reducing the number of LLM calls. Default is 1 (sequential).
        """
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.max_docs_chars = max_docs_chars
        self.max_chars_per_doc = max_chars_per_doc
        self.num_consistency_runs = num_consistency_runs
        self.evidence_first = evidence_first
        self.verification_agent = verification_agent
        self.batch_size = batch_size

    async def verify(
        self,
        claim: Claim,
        documents: list[str] | list[dict[str, str]],
        chunks: list[DocumentChunk] | None = None,
    ) -> VerificationResult:
        """Verify a single claim against the source documents.

        Args:
            claim: The claim to verify.
            documents: List of source document strings, or list of dicts
                with ``{"text": ..., "title": ...}`` entries.
            chunks: Pre-retrieved relevant document chunks. If provided,
                only these chunks are used for verification.

        Returns:
            :class:`VerificationResult` with verdict, confidence, evidence,
            and explanation.
        """
        # Determine which documents to use
        if chunks is not None:
            # Pass chunks as dicts so format_documents can include titles
            # as headers without polluting the raw text (span offsets).
            docs_to_verify = [
                {"text": c.text, "title": c.title}
                for c in chunks
                if c.title is not None
            ] or [c.text for c in chunks]  # fallback to plain text if no titles
        else:
            docs_to_verify = documents

        # Self-consistency: run multiple times with different temperatures
        if self.num_consistency_runs > 1:
            results = []
            for i in range(self.num_consistency_runs):
                # Vary temperature: 0.1, 0.2, 0.3, etc.
                temp = 0.1 + (i * 0.1)
                result = await self._single_verify(claim, docs_to_verify, temperature=temp)
                results.append(result)
            return self._aggregate_consistency(claim, chunks, results)
        else:
            result = await self._single_verify(claim, docs_to_verify, temperature=0.1)
            # Extract document_id from chunks when available
            if chunks and result.evidence and result.evidence != "N/A":
                for chunk in chunks:
                    if (
                        result.evidence.strip('"').strip() in chunk.text
                        or chunk.text in result.evidence
                    ):
                        result.document_id = chunk.doc_id
                        result.chunk_id = str(chunk.chunk_id)
                        break
            return result

    async def _single_verify(
        self,
        claim: Claim,
        documents: list[str],
        temperature: float = 0.1,
    ) -> VerificationResult:
        """Single verification pass."""
        if self.verification_agent is not None:
            return await self._verify_with_agent(claim, documents)

        if self.evidence_first:
            prompt = format_claim_verification_evidence_first_prompt(claim.text, documents)
        else:
            prompt = format_claim_verification_prompt(claim.text, documents)

        response = await self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
        )

        return self._parse_result(claim, response)

    async def _verify_with_agent(self, claim: Claim, documents: list[str]) -> VerificationResult:
        """Verify a claim using the atomic agent with structured output."""
        from .agents import VerificationInput

        formatted_docs = format_documents(documents)
        self.verification_agent.reset_history()
        input_schema = VerificationInput(claim=claim.text, documents=formatted_docs)
        result = await self.verification_agent.run_async(input_schema)

        # Map verdict variants
        raw_verdict = str(result.verdict).upper().strip()
        if "SUPPORTED" in raw_verdict and "NOT" not in raw_verdict:
            verdict = "supported"
        elif "CONTRADICTED" in raw_verdict:
            verdict = "contradicted"
        else:
            verdict = "not_enough_info"

        # Clamp document_index to valid range
        doc_index = result.document_index
        if doc_index is not None and (doc_index < 0 or doc_index >= len(documents)):
            log.debug(
                "verify: document_index %d out of range (%d docs), ignoring",
                doc_index,
                len(documents),
            )
            doc_index = None

        return VerificationResult(
            claim=claim.text,
            claim_index=claim.index,
            verdict=verdict,
            confidence=0,
            evidence=result.evidence or "N/A",
            explanation=result.explanation or "",
            document_index=doc_index,
        )

    def _aggregate_consistency(
        self,
        claim: Claim,
        chunks: list[DocumentChunk] | None,
        results: list[VerificationResult],
    ) -> VerificationResult:
        """Aggregate multiple verification results using majority vote.

        - Verdict: majority vote
        - Confidence: consistency score as percentage (how many runs agreed)
        - Evidence: from the result with the majority verdict
        - Consistency score: fraction of runs that agree with the majority
        """
        verdicts = [r.verdict for r in results]
        verdict_counts = Counter(verdicts)
        majority_verdict, majority_count = verdict_counts.most_common(1)[0]

        # Use evidence from the first result with the majority verdict
        majority_result = next((r for r in results if r.verdict == majority_verdict), results[0])

        # Extract document_id and chunk_id from evidence if available
        doc_id = None
        chunk_id = None
        if chunks and majority_result.evidence and majority_result.evidence != "N/A":
            for chunk in chunks:
                if (
                    majority_result.evidence.strip('"').strip() in chunk.text
                    or chunk.text in majority_result.evidence
                ):
                    doc_id = chunk.doc_id
                    chunk_id = str(chunk.chunk_id)
                    break

        consistency_score = majority_count / len(results)

        return VerificationResult(
            claim=claim.text,
            claim_index=claim.index,
            verdict=majority_verdict,
            confidence=int(consistency_score * 100),
            evidence=majority_result.evidence,
            explanation=majority_result.explanation,
            document_id=doc_id,
            document_index=majority_result.document_index,
            chunk_id=chunk_id,
            consistency_score=consistency_score,
        )

    def _parse_result(self, claim: Claim, response: str) -> VerificationResult:
        """Parse the LLM verification response into a structured result.

        Handles both JSON output and legacy VERDICT:/CONFIDENCE: text format.
        """
        # Try JSON first
        result = self._try_parse_json_result(claim, response)
        if result is not None:
            return result

        # Fall back to legacy text format
        return self._parse_text_result(claim, response)

    def _try_parse_json_result(self, claim: Claim, response: str) -> VerificationResult | None:
        """Try to parse response as JSON."""
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        # Map verdict variants
        raw_verdict = str(data.get("verdict", "")).upper().strip()
        if "SUPPORTED" in raw_verdict and "NOT" not in raw_verdict:
            verdict = "supported"
        elif "CONTRADICTED" in raw_verdict:
            verdict = "contradicted"
        else:
            verdict = "not_enough_info"

        evidence = data.get("evidence", "N/A") or "N/A"
        explanation = data.get("explanation", "") or ""

        # Parse document_index (0-based)
        doc_index_raw = data.get("document_index")
        doc_index = int(doc_index_raw) if doc_index_raw is not None else None

        return VerificationResult(
            claim=claim.text,
            claim_index=claim.index,
            verdict=verdict,
            confidence=0,
            evidence=evidence,
            explanation=explanation,
            document_index=doc_index,
        )

    def _parse_text_result(self, claim: Claim, response: str) -> VerificationResult:
        """Parse legacy VERDICT:/CONFIDENCE: text format."""
        verdict = "not_enough_info"
        evidence = "N/A"
        explanation = ""

        verdict_match = re.search(r"VERDICT:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if verdict_match:
            raw_verdict = verdict_match.group(1).strip().upper()
            if "SUPPORTED" in raw_verdict and "NOT" not in raw_verdict:
                verdict = "supported"
            elif "CONTRADICTED" in raw_verdict:
                verdict = "contradicted"
            elif "NOT ENOUGH INFO" in raw_verdict or "NOT_ENOUGH" in raw_verdict:
                verdict = "not_enough_info"
            elif "SUPPORTED" in raw_verdict:
                verdict = "supported"

        evidence_match = re.search(
            r"EVIDENCE:\s*(.+?)(?:\n\n|\nEXPLANATION|\nVERDICT|\Z)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if evidence_match:
            evidence = evidence_match.group(1).strip()

        expl_match = re.search(
            r"EXPLANATION:\s*(.+?)(?:\Z)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if expl_match:
            explanation = expl_match.group(1).strip()

        return VerificationResult(
            claim=claim.text,
            claim_index=claim.index,
            verdict=verdict,
            confidence=0,
            evidence=evidence,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Batch verification
    # ------------------------------------------------------------------

    async def verify_batch(
        self,
        claims: list[Claim],
        documents: list[str] | list[dict[str, str]],
    ) -> list[VerificationResult]:
        """Verify multiple claims in batches.

        Groups claims into batches of ``batch_size`` and sends each batch
        as a single LLM call. Documents are sent once per batch (KV cache
        reuse), and all claims in the batch are verified together.

        Args:
            claims: Claims to verify.
            documents: Source documents.

        Returns:
            List of :class:`VerificationResult`, one per claim.
        """
        if not claims:
            return []

        if self.batch_size < 2:
            # Fall back to sequential verification
            return [
                await self.verify(claim, documents)
                for claim in claims
            ]

        # Group claims into batches
        batches = [
            claims[i : i + self.batch_size]
            for i in range(0, len(claims), self.batch_size)
        ]

        log.info(
            "verify_batch: %d claims in %d batches (batch_size=%d)",
            len(claims),
            len(batches),
            self.batch_size,
        )

        results: list[VerificationResult] = []
        for batch in batches:
            batch_results = await self._single_batch_verify(batch, documents)
            results.extend(batch_results)

        return results

    async def _single_batch_verify(
        self,
        claims: list[Claim],
        documents: list[str] | list[dict[str, str]],
    ) -> list[VerificationResult]:
        """Verify a single batch of claims in one LLM call."""
        indexed_claims = [(c.index, c.text) for c in claims]
        prompt = format_claim_verification_batch_prompt(indexed_claims, documents)

        # Use larger max_new_tokens for batch responses (more claims = more output)
        batch_tokens = max(self.max_new_tokens, len(claims) * 128)

        response = await self.llm.generate(
            prompt,
            max_new_tokens=batch_tokens,
            temperature=0.1,
        )

        parsed = self._parse_batch_response(response)

        # Build VerificationResult from parsed data + original Claim objects
        results = []
        for claim in claims:
            if claim.index in parsed:
                p = parsed[claim.index]
                results.append(
                    VerificationResult(
                        claim=claim.text,
                        claim_index=claim.index,
                        verdict=p["verdict"],
                        confidence=0,
                        evidence=p["evidence"],
                        explanation=p["explanation"],
                        document_index=p["document_index"],
                    )
                )
            else:
                log.warning(
                    "verify_batch: missing result for claim %d, defaulting to not_enough_info",
                    claim.index,
                )
                results.append(
                    VerificationResult(
                        claim=claim.text,
                        claim_index=claim.index,
                        verdict="not_enough_info",
                        confidence=0,
                        evidence="N/A",
                        explanation="Batch response missing this claim.",
                    )
                )
        return results

    def _parse_batch_response(self, response: str) -> dict[int, dict]:
        """Parse a batch verification JSON array response.

        Returns a dict mapping claim_index -> dict with verdict, evidence,
        explanation, document_index.  Claim text is filled in by the caller
        from the original Claim objects.
        """
        text = response.strip()
        # Strip markdown code fences if present (handle multi-line)
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            log.warning("verify_batch: failed to parse JSON response: %s", response[:200])
            return {}

        if not isinstance(data, list):
            log.warning("verify_batch: expected JSON array, got %s", type(data).__name__)
            return {}

        results: dict[int, dict] = {}
        for item in data:
            if not isinstance(item, dict):
                continue

            claim_index = item.get("claim_index")
            if claim_index is None:
                continue
            try:
                claim_index = int(claim_index)
            except (ValueError, TypeError):
                continue

            # Map verdict
            raw_verdict = str(item.get("verdict", "")).upper().strip()
            if "SUPPORTED" in raw_verdict and "NOT" not in raw_verdict:
                verdict = "supported"
            elif "CONTRADICTED" in raw_verdict:
                verdict = "contradicted"
            else:
                verdict = "not_enough_info"

            evidence = item.get("evidence", "N/A") or "N/A"
            explanation = item.get("explanation", "") or ""
            doc_index_raw = item.get("document_index")
            doc_index = int(doc_index_raw) if doc_index_raw is not None else None

            results[claim_index] = {
                "verdict": verdict,
                "evidence": evidence,
                "explanation": explanation,
                "document_index": doc_index,
            }

        return results


class RAGFactsChecker:
    """Main entry point for RAG answer fact-checking.

    Orchestrates the pipeline:
    1. Extract factual claims from the RAG answer
    2. Optionally retrieve relevant document chunks per claim
    3. Verify each claim against the (retrieved) source documents
    4. Aggregate results into a comprehensive report

    Example::

        from rag_facts_check import RAGFactsChecker, MockLLM

        llm = MockLLM()
        checker = RAGFactsChecker(llm)

        report = checker.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
        )
        print(report.to_dict())
    """

    # Verdict weights for aggregation: supported=1.0, not_enough_info=0.5, contradicted=0.0
    VERDICT_WEIGHTS = {
        "supported": 1.0,
        "not_enough_info": 0.5,
        "contradicted": 0.0,
    }

    def __init__(
        self,
        llm: LLM,
        max_claims: int | None = None,
        max_new_tokens: int = 512,
        max_extraction_tokens: int | None = None,
        max_docs_chars: int = 100000,
        max_chars_per_doc: int = 10000,
        num_consistency_runs: int = 1,
        evidence_first: bool = True,
        use_evidence_retrieval: bool = True,
        retriever: EvidenceRetriever | None = None,
        instructor_client=None,
        model: str = "gemma",
        temperature: float = 0.1,
        batch_size: int = 1,
    ):
        """Initialize the checker.

        Args:
            llm: LLM backend implementing the :class:`LLM` interface.
            max_claims: Maximum number of claims to verify (limits latency).
            max_new_tokens: Max tokens for LLM generation (verification phase).
            max_extraction_tokens: Max tokens for claim extraction. Defaults to
                2048 to allow thorough decomposition of compound statements.
            max_docs_chars: Maximum total characters of documents to include.
            max_chars_per_doc: Maximum characters per individual document.
            num_consistency_runs: Number of verification runs for self-consistency.
                If >1, runs verification multiple times with different temperatures
                and aggregates via majority vote.
            evidence_first: If True, use the evidence-first multi-step prompt.
            use_evidence_retrieval: If True, retrieve relevant document chunks
                per claim before verification (reduces context, improves accuracy).
            retriever: Custom :class:`EvidenceRetriever` instance. If None,
                a default one is created.
            instructor_client: Optional instructor-wrapped client for structured
                output with automatic retries. When provided, atomic agents are
                used instead of raw LLM calls.
            model: Model name (used when building atomic agents).
            temperature: Sampling temperature (used when building atomic agents).
            batch_size: Number of claims to verify in a single LLM call.
                When >1, claims are grouped into batches, reducing LLM calls.
                Default is 1 (sequential per-claim verification).
        """
        self.llm = llm
        self.max_claims = max_claims
        self.max_new_tokens = max_new_tokens
        self.max_extraction_tokens = max_extraction_tokens or 2048
        self.max_docs_chars = max_docs_chars
        self.max_chars_per_doc = max_chars_per_doc
        self.num_consistency_runs = num_consistency_runs
        self.evidence_first = evidence_first
        self.use_evidence_retrieval = use_evidence_retrieval
        if retriever is not None:
            self.retriever = retriever
        elif use_evidence_retrieval:
            # Default: LLM-based retrieval for semantic relevance judgment
            from .retriever import LLMEvidenceRetriever

            self.retriever = LLMEvidenceRetriever(llm=llm)
        else:
            self.retriever = EvidenceRetriever(top_k=3)

        self.extractor = ClaimExtractor(llm, max_new_tokens=self.max_extraction_tokens)
        self.verifier = ClaimVerifier(
            llm,
            max_new_tokens=max_new_tokens,
            max_docs_chars=max_docs_chars,
            max_chars_per_doc=max_chars_per_doc,
            num_consistency_runs=num_consistency_runs,
            evidence_first=evidence_first,
            batch_size=batch_size,
        )

        # Build atomic agents when instructor client is available
        if instructor_client is not None:
            from .agents import (
                make_claim_extraction_agent,
                make_verification_agent,
            )

            extraction_agent = make_claim_extraction_agent(
                client=instructor_client,
                model=model,
                system_prompt=CLAIM_EXTRACTION_SYSTEM,
                temperature=temperature,
            )
            verification_system = (
                CLAIM_VERIFICATION_EVIDENCE_FIRST_SYSTEM
                if evidence_first
                else CLAIM_VERIFICATION_SYSTEM
            )
            verification_agent = make_verification_agent(
                client=instructor_client,
                model=model,
                system_prompt=verification_system,
                temperature=temperature,
                evidence_first=evidence_first,
            )
            self.extractor.extraction_agent = extraction_agent
            self.verifier.verification_agent = verification_agent

    async def check(
        self,
        answer: str,
        documents: list[str] | list[dict[str, str]],
    ) -> CheckReport:
        """Run the full fact-checking pipeline on a RAG answer.

        Args:
            answer: The RAG-generated answer to verify.
            documents: List of source document strings, or list of dicts
                with ``{"doc_id": ..., "text": ...}`` entries.

        Returns:
            :class:`CheckReport` with overall confidence, verdict, per-claim
            results, multi-dimensional scores, and hallucination flags.
        """
        # Step 1: Extract claims
        claims = await self.extractor.extract(answer)

        # Compute claim spans in the original answer using original_text
        # (exact verbatim fragment) for reliable matching, falling back to
        # the rephrased claim text if original_text is not available.
        for claim in claims:
            search_text = claim.original_text or claim.text
            span = find_span_in_text(search_text, answer)
            if span:
                claim.span = Span(start=span[0], end=span[1])
            else:
                log.debug(
                    "check: could not locate claim[%d] in answer: %s",
                    claim.index,
                    search_text[:80],
                )

        if not claims:
            log.info("check: no claims extracted, returning no_claims report")
            return CheckReport(
                answer=answer,
                overall_confidence=0.0,
                overall_verdict="no_claims",
                claims=[],
                results=[],
                summary="No factual claims were detected in the answer.",
                hallucination_flags=[],
                dimensions={
                    "groundedness": 0.0,
                    "contradiction_rate": 0.0,
                    "hallucination_rate": 0.0,
                    "completeness": 0.0,
                },
            )

        # Limit number of claims to verify (for latency control)
        if self.max_claims is not None:
            claims = claims[: self.max_claims]

        # Step 2: Pre-chunk documents for evidence retrieval.
        # Skipped in batch mode — batch verification sends all documents
        # directly and does not use per-claim chunk narrowing.
        chunks = None
        if self.verifier.batch_size < 2 and self.use_evidence_retrieval and documents:
            chunks = self.retriever.chunk_documents(documents)

        # When evidence retrieval is disabled, pass original documents
        # (may be dicts with title) so format_documents can include titles
        docs_for_verifier = documents

        # Step 3: Verify claims
        if self.verifier.batch_size > 1:
            # Batch verification: one LLM call per batch of claims
            results = await self.verifier.verify_batch(claims, docs_for_verifier)
            # Compute evidence spans for batch results
            for result in results:
                evidence_span = self._find_evidence_span(result, documents, None)
                if evidence_span:
                    result.evidence_span = evidence_span
        else:
            # Sequential verification: one LLM call per claim
            results = []
            for claim in claims:
                # Retrieve relevant chunks for this claim
                relevant_chunks = None
                if chunks is not None:
                    relevant_chunks = await self.retriever.retrieve(claim.text, chunks)

                result = await self.verifier.verify(claim, docs_for_verifier, chunks=relevant_chunks)

                # Compute evidence span in source documents
                evidence_span = self._find_evidence_span(
                    result,
                    documents,
                    relevant_chunks,
                )
                if evidence_span:
                    result.evidence_span = evidence_span

                results.append(result)

        # Step 4: Aggregate
        return self._aggregate(answer, claims, results)

    def _find_evidence_span(
        self,
        result: VerificationResult,
        documents: list[str] | list[dict[str, str]],
        relevant_chunks: list["DocumentChunk"] | None,
    ) -> Span | None:
        """Find the evidence span for a verification result.

        Strategy:
        1. If the LLM provided a document_index, search that document first
        2. Fall back to searching all documents
        3. If evidence quote doesn't match, use the top retrieved chunk's offsets
        """
        evidence = result.evidence
        if not evidence or evidence == "N/A":
            return None

        # Step 1: Targeted search using document_index
        if result.document_index is not None:
            idx = result.document_index
            if 0 <= idx < len(documents):
                doc = documents[idx]
                doc_text = doc["text"] if isinstance(doc, dict) else doc
                span = find_evidence_span_in_doc(evidence, doc_text)
                if span is not None:
                    return Span(start=span[0], end=span[1])

        # Step 2: Search all documents
        all_match = find_evidence_span(evidence, documents)
        if all_match is not None:
            return Span(start=all_match[1], end=all_match[2])

        # Evidence quote not found in any document. Return None so the
        # segment is skipped — better than a bogus span pointing to
        # unrelated text.
        log.debug(
            "_find_evidence_span: evidence quote not found for claim[%d]: %s",
            result.claim_index,
            evidence[:80],
        )
        return None

    # Answer quality score constants
    _SCORE_NEI_WEIGHT = 0.4  # not_enough_info verdict weight
    _SCORE_CITATION_MAX_REDUCTION = 0.3  # max 30% reduction for uncited claims
    _SCORE_CONTRADICTION_MULTIPLIER = 1.5  # contradictions drive score to zero

    def _compute_answer_score(
        self,
        results: list[VerificationResult],
    ) -> float:
        """Compute a 0-10 answer quality score from verification results.

        Three multiplicative factors:
        1. Groundedness base — weighted average of verdicts scaled to 0-10
        2. Citation penalty — reduction for claims without evidence segments
        3. Contradiction penalty — harsh reduction for contradicted claims

        Returns:
            Score 0-10, rounded to 1 decimal place.
        """
        total = len(results)
        if total == 0:
            return 0.0

        # 1. Groundedness base (0-10) — verdict-based, no LLM confidence
        verdict_weights = {
            "supported": 1.0,
            "not_enough_info": self._SCORE_NEI_WEIGHT,
            "contradicted": 0.0,
        }
        weighted_sum = sum(verdict_weights.get(r.verdict, 0.5) for r in results)
        groundedness = weighted_sum / total * 10

        # 2. Citation penalty (1.0 = no penalty, 0.7 = max penalty)
        cited = sum(1 for r in results if r.evidence_span is not None)
        citation_ratio = cited / total
        citation_penalty = 1.0 - (
            self._SCORE_CITATION_MAX_REDUCTION * (1 - citation_ratio)
        )

        # 3. Contradiction penalty (1.0 = no penalty, 0 = max penalty)
        contradicted = sum(1 for r in results if r.verdict == "contradicted")
        contradiction_penalty = max(
            0.0, 1.0 - (self._SCORE_CONTRADICTION_MULTIPLIER * contradicted / total)
        )

        # Compose
        raw = groundedness * citation_penalty * contradiction_penalty
        return round(max(0.0, min(10.0, raw)), 1)

    def _aggregate(
        self,
        answer: str,
        claims: list[Claim],
        results: list[VerificationResult],
    ) -> CheckReport:
        """Aggregate per-claim results into a comprehensive report."""
        total = len(results)
        supported = sum(1 for r in results if r.verdict == "supported")
        contradicted = sum(1 for r in results if r.verdict == "contradicted")
        not_enough = sum(1 for r in results if r.verdict == "not_enough_info")

        # Overall confidence: percentage of claims that are supported
        overall_confidence = supported / total * 100 if total > 0 else 0.0

        # Overall verdict
        support_ratio = supported / total if total > 0 else 0
        contradiction_ratio = contradicted / total if total > 0 else 0

        if total == 0:
            overall_verdict = "no_claims"
        elif support_ratio == 1.0:
            overall_verdict = "fully_supported"
        elif support_ratio >= 0.7 and contradiction_ratio == 0:
            overall_verdict = "mostly_supported"
        elif support_ratio >= 0.3:
            overall_verdict = "partially_supported"
        else:
            overall_verdict = "largely_unsupported"

        # Identify hallucination flags (contradicted or not_enough_info)
        hallucination_flags = [
            r for r in results if r.verdict in ("contradicted", "not_enough_info")
        ]

        # Multi-dimensional scoring
        dimensions = {
            "groundedness": round(supported / total * 100, 1) if total > 0 else 0.0,
            "contradiction_rate": round(contradicted / total * 100, 1) if total > 0 else 0.0,
            "hallucination_rate": round((contradicted + not_enough) / total * 100, 1)
            if total > 0
            else 0.0,
            "completeness": round(supported / total * 100, 1) if total > 0 else 0.0,
        }

        # Build summary
        summary = self._build_summary(
            total, supported, contradicted, not_enough, overall_confidence, overall_verdict
        )

        # Compute answer quality score
        answer_score = self._compute_answer_score(results)

        return CheckReport(
            answer=answer,
            answer_score=answer_score,
            overall_confidence=overall_confidence,
            overall_verdict=overall_verdict,
            claims=claims,
            results=results,
            summary=summary,
            hallucination_flags=hallucination_flags,
            dimensions=dimensions,
        )

    def _build_summary(
        self,
        total: int,
        supported: int,
        contradicted: int,
        not_enough: int,
        confidence: float,
        verdict: str,
    ) -> str:
        """Build a human-readable summary of the check results."""
        verdict_labels = {
            "fully_supported": "Fully supported by source documents",
            "mostly_supported": "Mostly supported by source documents",
            "partially_supported": "Partially supported — some claims lack evidence",
            "largely_unsupported": "Largely unsupported — many claims lack evidence",
            "no_claims": "No factual claims detected",
        }

        label = verdict_labels.get(verdict, verdict)
        parts = [
            f"Overall confidence: {confidence:.0f}%",
            f"Verdict: {label}",
            f"Claims: {total} total, {supported} supported, "
            f"{contradicted} contradicted, {not_enough} need more info.",
        ]

        if contradicted > 0:
            parts.append(f"⚠️  {contradicted} claim(s) are contradicted by the source documents.")
        if not_enough > 0:
            parts.append(
                f"ℹ️  {not_enough} claim(s) could not be verified — "
                f"the source documents do not contain sufficient evidence."
            )

        return "\n".join(parts)
