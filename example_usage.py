"""
Example usage of the RAG Facts Check system.

Demonstrates the full pipeline with the MockLLM and shows how to
integrate with a real local model.
"""

from rag_facts_check import RAGFactsChecker, MockLLM, EvidenceRetriever


def print_report(report):
    """Pretty-print a check report."""
    print(f"Answer: {report.answer}")
    print(f"Overall Confidence: {report.overall_confidence:.1f}%")
    print(f"Overall Verdict: {report.overall_verdict}")
    print(f"\nMulti-dimensional scores:")
    for dim, score in report.dimensions.items():
        print(f"  {dim}: {score}")
    print(f"\nSummary:\n{report.summary}")
    print(f"\nClaims extracted: {len(report.claims)}")
    for claim in report.claims:
        print(f"  [{claim.index}] {claim.text}")

    print(f"\nVerification results: {len(report.results)}")
    for result in report.results:
        print(f"  [{result.claim_index}] {result.claim}")
        print(f"      Verdict: {result.verdict}")
        print(f"      Confidence: {result.confidence}%")
        print(f"      Evidence: {result.evidence}")
        if result.document_id:
            print(f"      Document: {result.document_id}, Chunk: {result.chunk_id}")
        if result.consistency_score is not None:
            print(f"      Consistency: {result.consistency_score:.0%}")
        print(f"      Explanation: {result.explanation}")

    if report.hallucination_flags:
        print(f"\n⚠️  Hallucination flags ({len(report.hallucination_flags)}):")
        for flag in report.hallucination_flags:
            print(f"  Claim [{flag.claim_index}]: {flag.claim}")
            print(f"    Verdict: {flag.verdict}")
            print(f"    Evidence: {flag.evidence}")


def main():
    print("=" * 70)
    print("RAG Facts Check — Example Usage")
    print("=" * 70)

    # ── Basic example with MockLLM ─────────────────────────────────────
    print("\n[1] Basic check with MockLLM\n")
    llm = MockLLM()
    checker = RAGFactsChecker(llm, max_claims=10)

    answer = (
        "Paris is the capital of France. "
        "The Eiffel Tower was built in 1889. "
        "The Louvre Museum is located in Berlin."
    )

    documents = [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "The Eiffel Tower was constructed between 1887 and 1889.",
        "The Louvre Museum is one of the world's largest museums, located in Paris, France.",
    ]

    report = checker.check(answer, documents)
    print_report(report)

    # ── Self-consistency example ────────────────────────────────────────
    print("\n[2] Self-consistency (3 runs)\n")
    llm2 = MockLLM()
    checker2 = RAGFactsChecker(
        llm2,
        max_claims=10,
        num_consistency_runs=3,
        evidence_first=True,
    )
    report2 = checker2.check(answer, documents)
    print_report(report2)

    # ── Evidence retrieval example ──────────────────────────────────────
    print("\n[3] Evidence retrieval (retriever with top_k=2)\n")
    llm3 = MockLLM()
    retriever = EvidenceRetriever(chunk_size=50, top_k=2)
    checker3 = RAGFactsChecker(
        llm3,
        max_claims=10,
        retriever=retriever,
        evidence_first=True,
    )
    report3 = checker3.check(answer, documents)
    print_report(report3)

    # ── JSON output ─────────────────────────────────────────────────────
    print("\n[4] JSON output:\n")
    import json
    print(json.dumps(report.to_dict(), indent=2))

    # ── Integration example with a real local model ─────────────────────
    print("\n[5] Integration example (pseudo-code):\n")
    print("""
    # For Hugging Face Transformers:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from rag_facts_check import HuggingFaceLLM, RAGFactsChecker

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    llm = HuggingFaceLLM(model, tokenizer, chat_format=True)
    checker = RAGFactsChecker(llm)

    # For HTTP API (vLLM, Ollama, llama.cpp server):
    from rag_facts_check import APILLM, RAGFactsChecker
    llm = APILLM("http://localhost:8000/v1/completions", model_name="my-model")
    checker = RAGFactsChecker(llm)

    # For custom local model:
    from rag_facts_check import LLM, RAGFactsChecker
    class MyLLM(LLM):
        def generate(self, prompt, max_new_tokens=512, temperature=0.1, **kwargs):
            # Your model inference here
            return "generated text"
    llm = MyLLM()
    checker = RAGFactsChecker(llm)
    """)


if __name__ == "__main__":
    main()
