"""
Prompt templates for claim extraction and verification.

These prompts are designed to work well with instruction-tuned local
LLMs (Llama-2-Chat, Mistral-Instruct, Gemma, etc.).  Adjust the
system context and examples as needed for your specific model.
"""

# ---------------------------------------------------------------------------
# Claim Extraction
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_SYSTEM = """You are an expert fact-checker. Your task is to extract all factual claims from the given text.

A factual claim is a statement that can be verified as true or false based on evidence.
Do NOT extract:
- Opinions, beliefs, or subjective statements
- Questions
- General greetings or small talk
- Statements that are clearly hypothetical
- Meta-text about the answer itself

Each claim should be:
- Atomic (a single fact, not a compound statement with multiple facts)
- Verifiable (can be checked against source documents)
- Specific (not vague or general)

If the text contains no factual claims, output "NO CLAIMS"."""

CLAIM_EXTRACTION_PROMPT = """{system_prompt}

Extract claims from the following text:

Text:
{text}

List each claim on a separate line, prefixed with "CLAIM N: " where N is the claim number starting from 1.
If there are no factual claims, output "NO CLAIMS".

Claims:"""


def format_claim_extraction_prompt(text: str) -> str:
    """Build the full claim extraction prompt."""
    return CLAIM_EXTRACTION_PROMPT.format(system_prompt=CLAIM_EXTRACTION_SYSTEM, text=text)


# ---------------------------------------------------------------------------
# Claim Verification — Standard
# ---------------------------------------------------------------------------

CLAIM_VERIFICATION_SYSTEM = """You are an expert fact-checker. Your task is to verify whether a claim is supported by the provided source documents.

For each claim, you must:
1. Read the claim carefully
2. Search through ALL source documents for relevant evidence
3. Determine the verdict:
   - SUPPORTED: The documents contain clear, direct evidence that the claim is true
   - CONTRADICTED: The documents contain clear evidence that the claim is false
   - NOT ENOUGH INFO: The documents do not contain sufficient information to verify the claim (neither support nor contradict)

4. Quote the specific evidence from the documents
5. Provide a confidence score (0-100)

Be strict: if the evidence is ambiguous or indirect, lean toward "NOT ENOUGH INFO"."""

CLAIM_VERIFICATION_PROMPT = """{system_prompt}

Claim:
{claim}

Source Documents:
{documents}

Instructions:
- Read the claim and the source documents carefully
- Search for evidence that supports or contradicts the claim
- Provide your response in the EXACT format below:

VERDICT: [SUPPORTED|CONTRADICTED|NOT ENOUGH INFO]
CONFIDENCE: [0-100]
EVIDENCE: [exact quote from source documents, or "N/A" if not enough info]
EXPLANATION: [brief explanation, max 2 sentences]"""


def format_claim_verification_prompt(claim: str, documents: list[str]) -> str:
    """Build the full claim verification prompt.

    Args:
        claim: The claim text to verify.
        documents: List of document strings.

    Returns:
        Formatted prompt string.
    """
    formatted_docs = format_documents(documents)
    return CLAIM_VERIFICATION_PROMPT.format(
        system_prompt=CLAIM_VERIFICATION_SYSTEM,
        claim=claim,
        documents=formatted_docs,
    )


# ---------------------------------------------------------------------------
# Claim Verification — Evidence-First (Multi-Step)
# ---------------------------------------------------------------------------

CLAIM_VERIFICATION_EVIDENCE_FIRST_SYSTEM = """You are an expert fact-checker. Your task is to verify whether a claim is supported by the provided source documents.

Follow these steps in order:

Step 1: Extract relevant evidence
Read the source documents and extract any passages that are relevant to the claim. Quote them exactly.

Step 2: Compare evidence to claim
Compare the extracted evidence to the claim. Does the evidence support, contradict, or fail to address the claim?

Step 3: Verdict
Classify the claim as:
- SUPPORTED: Evidence clearly supports the claim
- CONTRADICTED: Evidence clearly contradicts the claim
- NOT ENOUGH INFO: Evidence is insufficient to determine

Step 4: Output
Provide your response in the EXACT format below:

VERDICT: [SUPPORTED|CONTRADICTED|NOT ENOUGH INFO]
CONFIDENCE: [0-100]
EVIDENCE: [exact quote from source documents, or "N/A"]
EXPLANATION: [brief explanation, max 2 sentences]

Be strict: if the evidence is ambiguous or indirect, lean toward "NOT ENOUGH INFO"."""

CLAIM_VERIFICATION_EVIDENCE_FIRST_PROMPT = """{system_prompt}

Claim:
{claim}

Source Documents:
{documents}"""


def format_claim_verification_evidence_first_prompt(claim: str, documents: list[str]) -> str:
    """Build the evidence-first multi-step verification prompt.

    This prompt explicitly asks the model to extract evidence first,
    then compare it to the claim, then provide a verdict. This reduces
    hallucinated evaluations where the verifier makes up evidence.

    Args:
        claim: The claim text to verify.
        documents: List of document strings.

    Returns:
        Formatted prompt string.
    """
    formatted_docs = format_documents(documents)
    return CLAIM_VERIFICATION_EVIDENCE_FIRST_PROMPT.format(
        system_prompt=CLAIM_VERIFICATION_EVIDENCE_FIRST_SYSTEM,
        claim=claim,
        documents=formatted_docs,
    )


# ---------------------------------------------------------------------------
# Document Formatting
# ---------------------------------------------------------------------------


def format_documents(
    documents: list[str],
    max_chars_per_doc: int = 2000,
    max_total_chars: int = 8000,
) -> str:
    """Format a list of documents into a single string for the LLM.

    Args:
        documents: List of document strings.
        max_chars_per_doc: Maximum characters per document (truncated).
        max_total_chars: Maximum total characters across all documents.

    Returns:
        Formatted documents string.
    """
    if not documents:
        return "(No source documents provided)"

    parts = []
    total_chars = 0

    for i, doc in enumerate(documents):
        if total_chars >= max_total_chars:
            parts.append("\n[Remaining documents truncated to fit context window]")
            break

        # Truncate individual document
        if len(doc) > max_chars_per_doc:
            doc = doc[:max_chars_per_doc] + "... [truncated]"

        remaining = max_total_chars - total_chars
        if len(doc) > remaining:
            doc = doc[:remaining] + "... [truncated]"

        parts.append(f"Document {i + 1}:\n{doc}\n")
        total_chars += len(doc)

    return "\n".join(parts)
