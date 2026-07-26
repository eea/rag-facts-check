"""
Mock LLM implementations for testing and development.

These mocks return predefined responses based on keywords in the prompt,
allowing the full fact-checking pipeline to be tested without a real LLM.
"""

import re
from typing import List

from ..llm import LLM


class MockLLM(LLM):
    """Mock LLM for testing and development.

    Returns predefined responses based on keywords in the prompt.
    Handles claim extraction, claim verification, and environmental
    topic verification.

    Example::

        from rag_facts_check.testing import MockLLM
        from rag_facts_check import RAGFactsChecker

        llm = MockLLM()
        checker = RAGFactsChecker(llm)
    """

    def __init__(self):
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        self.call_count += 1
        lower = prompt.lower()

        if "extract all factual claims" in lower or "extract claims from the following" in lower:
            return self._mock_claims_response(prompt)
        elif "verify whether a claim is supported" in lower or "verify the following claim" in lower:
            return self._mock_verification_response(prompt)
        else:
            return "Mock response for: " + prompt[:100]

    def _mock_claims_response(self, prompt: str) -> str:
        """Mock claim extraction: split text into sentences and treat each as a claim."""
        # Extract the text portion from the prompt (between "Text:\n" and "\nList each claim")
        text_start = prompt.find("Text:")
        if text_start >= 0:
            text_start += 5  # skip "Text:"
            # Find the end: look for the instructions that follow the text
            text_end = prompt.find("List each claim", text_start)
            if text_end < 0:
                text_end = prompt.find("Claims:", text_start)
            if text_end < 0:
                text_end = len(prompt)
            text = prompt[text_start:text_end].strip()
        else:
            text = "Paris is the capital of France. The Eiffel Tower was built in 1889."

        # Simple sentence-based claim extraction for mock
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for i, s in enumerate(sentences, 1):
            s = s.strip().rstrip('.')
            if s and len(s) > 5:
                claims.append(f"CLAIM {i}: {s}.")
        if not claims:
            return "NO CLAIMS"
        return "\n".join(claims)

    def _mock_verification_response(self, prompt: str) -> str:
        """Mock verification: check if claim is supported/contradicted by docs."""
        # Extract claim from prompt
        claim_match = re.search(r'Claim:\s*\n(.+?)\n\nSource Documents:', prompt, re.DOTALL)
        claim = claim_match.group(1).strip() if claim_match else "unknown claim"

        # Extract documents from prompt (handles both standard and evidence-first formats)
        docs_match = re.search(
            r'Source Documents:\s*\n(.+?)(?:\n\nInstructions:|\n\nStep 1:|\Z)',
            prompt, re.DOTALL
        )
        docs_text = docs_match.group(1).strip() if docs_match else ""

        # Simple keyword-based mock verification
        lower_claim = claim.lower()
        lower_docs = docs_text.lower()

        # ── General knowledge claims ──
        # Check for contradiction (claim says one thing, docs say another)
        if "berlin" in lower_claim and "paris" in lower_docs:
            return """VERDICT: CONTRADICTED
CONFIDENCE: 85
EVIDENCE: "The Louvre Museum is one of the world's largest museums, located in Paris, France."
EXPLANATION: The source documents state the Louvre is in Paris, but the claim says it is in Berlin. This is a direct contradiction."""

        if "capital of france" in lower_claim and "capital of france" in lower_docs:
            return """VERDICT: SUPPORTED
CONFIDENCE: 95
EVIDENCE: "Paris is the capital of France."
EXPLANATION: The source document explicitly states that Paris is the capital of France, which directly supports the claim."""

        if "eiffel tower" in lower_claim and "1889" in lower_docs:
            return """VERDICT: SUPPORTED
CONFIDENCE: 90
EVIDENCE: "The Eiffel Tower was constructed between 1887 and 1889."
EXPLANATION: The source document confirms the Eiffel Tower was built in 1889, supporting the claim."""

        # ── Environmental topic checks ──
        # Climate change: 5.7°C or 6.2°C contradicts 1.5-4°C in docs
        if any(x in lower_claim for x in ["5.7", "6.2"]) and "temperature" in lower_claim:
            return """VERDICT: CONTRADICTED
CONFIDENCE: 85
EVIDENCE: "Current climate models estimate a 2-4°C rise by the end of the century."
EXPLANATION: The source documents project a 2-4°C temperature rise, but the claim states 5.7°C, which is a direct contradiction."""

        # Renewable energy: 30% supported, 42% supported, 78% contradicted
        if "30%" in lower_claim and "30%" in lower_docs:
            return """VERDICT: SUPPORTED
CONFIDENCE: 95
EVIDENCE: "In 2023, renewable energy sources accounted for 30% of global electricity generation, with solar and wind leading the growth."
EXPLANATION: The source document explicitly states that renewables accounted for 30% of global electricity in 2023, supporting the claim."""

        if "42%" in lower_claim and "42%" in lower_docs:
            return """VERDICT: SUPPORTED
CONFIDENCE: 90
EVIDENCE: "The International Renewable Energy Agency (IRENA) projects renewables will reach 42% of global electricity by 2028."
EXPLANATION: The source document confirms IRENA projects 42% renewable electricity by 2028, supporting the claim."""

        if "78%" in lower_claim and "renewable" in lower_claim:
            return """VERDICT: CONTRADICTED
CONFIDENCE: 85
EVIDENCE: "In 2023, renewable energy sources accounted for 30% of global electricity generation."
EXPLANATION: The source documents state renewables were 30% in 2023, but the claim says 78%, which is a significant contradiction."""

        # General number matching for environmental claims
        claim_numbers = set(re.findall(r'\d+(?:\.\d+)?%', lower_claim))
        doc_numbers = set(re.findall(r'\d+(?:\.\d+)?%', lower_docs))
        if claim_numbers and claim_numbers & doc_numbers:
            matched = claim_numbers & doc_numbers
            return f"""VERDICT: SUPPORTED
CONFIDENCE: 85
EVIDENCE: "A matching statistic appears in the source documents."
EXPLANATION: The claim contains the statistic {matched.pop()}, which appears in the source documents, supporting the claim."""

        # Default: not enough info
        return """VERDICT: NOT ENOUGH INFO
CONFIDENCE: 60
EVIDENCE: N/A
EXPLANATION: The source documents do not contain sufficient information to verify this claim."""
