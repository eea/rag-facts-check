"""
Example usage of the RAG Facts Check system.

Demonstrates the full pipeline with the MockLLM and shows how to
integrate with a real local model.
"""

from rag_facts_check import RAGFactsChecker, MockLLM


def main():
    print("=" * 70)
    print("RAG Facts Check — Example Usage")
    print("=" * 70)

    # ── Mock LLM example ──────────────────────────────────────────────
    print("\n[1] Using MockLLM (for testing)\n")
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

    print(f"Answer: {report.answer}")
    print(f"Overall Confidence: {report.overall_confidence:.1f}%")
    print(f"Overall Verdict: {report.overall_verdict}")
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
        print(f"      Explanation: {result.explanation}")

    if report.hallucination_flags:
        print(f"\n⚠️  Hallucination flags ({len(report.hallucination_flags)}):")
        for flag in report.hallucination_flags:
            print(f"  Claim [{flag.claim_index}]: {flag.claim}")
            print(f"    Verdict: {flag.verdict}")
            print(f"    Evidence: {flag.evidence}")

    # ── JSON output ───────────────────────────────────────────────────
    print("\n[2] JSON output:\n")
    import json
    print(json.dumps(report.to_dict(), indent=2))

    # ── Integration example with a real local model ───────────────────
    print("\n[3] Integration example (pseudo-code):\n")
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
