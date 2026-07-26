"""
Core fact-checking pipeline: claim extraction, per-claim verification,
and result aggregation.
"""

import re
from typing import List, Optional

from .models import Claim, VerificationResult, CheckReport
from .llm import LLM
from .prompts import (
    format_claim_extraction_prompt,
    format_claim_verification_prompt,
    format_documents,
)


class ClaimExtractor:
    """Extracts atomic factual claims from a RAG-generated answer.

    Uses an LLM to parse the answer and identify individual verifiable
    statements.
    """

    def __init__(self, llm: LLM, max_new_tokens: int = 512):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def extract(self, answer: str) -> List[Claim]:
        """Extract factual claims from *answer*.

        Args:
            answer: The RAG-generated answer text.

        Returns:
            List of :class:`Claim` objects.
        """
        if not answer or not answer.strip():
            return []

        prompt = format_claim_extraction_prompt(answer)
        response = self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
        )

        return self._parse_claims(response)

    def _parse_claims(self, response: str) -> List[Claim]:
        """Parse the LLM response into a list of claims.

        Expected format: "CLAIM N: <text>" per line.
        """
        claims = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            match = re.match(r"^CLAIM\s+(\d+):\s*(.+)$", line, re.IGNORECASE)
            if match:
                index = int(match.group(1))
                text = match.group(2).strip()
                if text:
                    claims.append(Claim(text=text, index=index))

        # If no claims were parsed, try to interpret the response as a single claim
        if not claims and response.strip() and "NO CLAIMS" not in response.upper():
            claims.append(Claim(text=response.strip(), index=1))

        return claims


class ClaimVerifier:
    """Verifies individual claims against a set of source documents.

    For each claim, determines whether it is *supported*, *contradicted*,
    or has *not enough info* based on the provided documents.
    """

    def __init__(
        self,
        llm: LLM,
        max_new_tokens: int = 512,
        max_docs_chars: int = 8000,
        max_chars_per_doc: int = 2000,
    ):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.max_docs_chars = max_docs_chars
        self.max_chars_per_doc = max_chars_per_doc

    def verify(self, claim: Claim, documents: List[str]) -> VerificationResult:
        """Verify a single claim against the source documents.

        Args:
            claim: The claim to verify.
            documents: List of source document strings.

        Returns:
            :class:`VerificationResult` with verdict, confidence, evidence,
            and explanation.
        """
        formatted_docs = format_documents(
            documents,
            max_chars_per_doc=self.max_chars_per_doc,
            max_total_chars=self.max_docs_chars,
        )

        prompt = format_claim_verification_prompt(claim.text, documents)
        response = self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
        )

        return self._parse_result(claim, response)

    def _parse_result(self, claim: Claim, response: str) -> VerificationResult:
        """Parse the LLM verification response into a structured result."""
        verdict = "not_enough_info"
        confidence = 50
        evidence = "N/A"
        explanation = ""

        # Parse VERDICT
        verdict_match = re.search(
            r"VERDICT:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
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

        # Parse CONFIDENCE
        conf_match = re.search(
            r"CONFIDENCE:\s*(\d+)", response, re.IGNORECASE
        )
        if conf_match:
            confidence = min(100, max(0, int(conf_match.group(1))))

        # Parse EVIDENCE
        evidence_match = re.search(
            r"EVIDENCE:\s*(.+?)(?:\n\n|\nEXPLANATION|\Z)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if evidence_match:
            evidence = evidence_match.group(1).strip()

        # Parse EXPLANATION
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
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
        )


class RAGFactsChecker:
    """Main entry point for RAG answer fact-checking.

    Orchestrates the pipeline:
    1. Extract factual claims from the RAG answer
    2. Verify each claim against the source documents
    3. Aggregate results into a comprehensive report

    Example::

        from rag_facts_check import RAGFactsChecker, MockLLM

        llm = MockLLM()
        checker = RAGFactsChecker(llm)

        report = checker.check(
            answer="Paris is the capital of France.",
            documents=["Paris is the capital of France. It is known for the Eiffel Tower."],
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
        max_claims: Optional[int] = None,
        max_new_tokens: int = 512,
        max_docs_chars: int = 8000,
        max_chars_per_doc: int = 2000,
    ):
        """Initialize the checker.

        Args:
            llm: LLM backend implementing the :class:`LLM` interface.
            max_claims: Maximum number of claims to verify (limits latency).
            max_new_tokens: Max tokens for LLM generation.
            max_docs_chars: Maximum total characters of documents to include.
            max_chars_per_doc: Maximum characters per individual document.
        """
        self.llm = llm
        self.max_claims = max_claims
        self.max_new_tokens = max_new_tokens
        self.max_docs_chars = max_docs_chars
        self.max_chars_per_doc = max_chars_per_doc

        self.extractor = ClaimExtractor(llm, max_new_tokens=max_new_tokens)
        self.verifier = ClaimVerifier(
            llm,
            max_new_tokens=max_new_tokens,
            max_docs_chars=max_docs_chars,
            max_chars_per_doc=max_chars_per_doc,
        )

    def check(self, answer: str, documents: List[str]) -> CheckReport:
        """Run the full fact-checking pipeline on a RAG answer.

        Args:
            answer: The RAG-generated answer to verify.
            documents: List of source document strings retrieved by the RAG system.

        Returns:
            :class:`CheckReport` with overall confidence, verdict, per-claim
            results, and hallucination flags.
        """
        # Step 1: Extract claims
        claims = self.extractor.extract(answer)

        if not claims:
            return CheckReport(
                answer=answer,
                overall_confidence=0.0,
                overall_verdict="no_claims",
                claims=[],
                results=[],
                summary="No factual claims were detected in the answer.",
                hallucination_flags=[],
            )

        # Limit number of claims to verify (for latency control)
        if self.max_claims is not None:
            claims = claims[: self.max_claims]

        # Step 2: Verify each claim
        results = []
        for claim in claims:
            result = self.verifier.verify(claim, documents)
            results.append(result)

        # Step 3: Aggregate
        return self._aggregate(answer, claims, results)

    def _aggregate(
        self,
        answer: str,
        claims: List[Claim],
        results: List[VerificationResult],
    ) -> CheckReport:
        """Aggregate per-claim results into a comprehensive report."""
        total = len(results)
        supported = sum(1 for r in results if r.verdict == "supported")
        contradicted = sum(1 for r in results if r.verdict == "contradicted")
        not_enough = sum(1 for r in results if r.verdict == "not_enough_info")

        # Overall confidence: weighted average of per-claim confidence
        weighted_sum = sum(
            r.confidence * self.VERDICT_WEIGHTS.get(r.verdict, 0.5)
            for r in results
        )
        max_possible = sum(
            100 * self.VERDICT_WEIGHTS.get(r.verdict, 0.5)
            for r in results
        )
        overall_confidence = (weighted_sum / max_possible * 100) if max_possible > 0 else 0.0

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

        # Build summary
        summary = self._build_summary(
            total, supported, contradicted, not_enough, overall_confidence, overall_verdict
        )

        return CheckReport(
            answer=answer,
            overall_confidence=overall_confidence,
            overall_verdict=overall_verdict,
            claims=claims,
            results=results,
            summary=summary,
            hallucination_flags=hallucination_flags,
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
            parts.append(
                f"⚠️  {contradicted} claim(s) are contradicted by the source documents."
            )
        if not_enough > 0:
            parts.append(
                f"ℹ️  {not_enough} claim(s) could not be verified — "
                f"the source documents do not contain sufficient evidence."
            )

        return "\n".join(parts)
