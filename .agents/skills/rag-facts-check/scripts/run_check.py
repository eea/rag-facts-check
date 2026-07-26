#!/usr/bin/env python3
"""
CLI script for running the RAG Facts Check pipeline.

Usage:
    python .agents/skills/rag-facts-check/scripts/run_check.py \
        --answer "Paris is the capital of France." \
        --documents "Paris is the capital of France." \
        --output report.json

    python .agents/skills/rag-facts-check/scripts/run_check.py \
        --answer-file answer.txt \
        --documents-file documents.json \
        --output report.json \
        --num-consistency-runs 3 \
        --evidence-first \
        --use-evidence-retrieval
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
# Script is at: <root>/.agents/skills/rag-facts-check/scripts/run_check.py
# Project root is 4 levels up from the script file
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rag_facts_check import (
    RAGFactsChecker,
    MockLLM,
    EvidenceRetriever,
)


def load_answer(args) -> str:
    """Load the answer from --answer or --answer-file."""
    if args.answer:
        return args.answer
    if args.answer_file:
        return Path(args.answer_file).read_text()
    raise ValueError("Either --answer or --answer-file is required")


def load_documents(args) -> list[str]:
    """Load documents from --documents or --documents-file."""
    if args.documents:
        # With action="append", args.documents is already a list of strings
        return args.documents
    if args.documents_file:
        content = Path(args.documents_file).read_text()
        try:
            docs = json.loads(content)
            if isinstance(docs, list):
                return docs
            if isinstance(docs, dict) and "documents" in docs:
                return docs["documents"]
        except json.JSONDecodeError:
            # Treat as plain text, one document
            return [content]
    raise ValueError("Either --documents or --documents-file is required")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG Facts Check pipeline on a RAG-generated answer."
    )
    parser.add_argument(
        "--answer",
        type=str,
        help="The RAG-generated answer to verify (inline string).",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        help="Path to a file containing the answer text.",
    )
    parser.add_argument(
        "--documents",
        type=str,
        action="append",
        help="Source document text (inline string). Can be specified multiple times.",
    )
    parser.add_argument(
        "--documents-file",
        type=str,
        help="Path to a JSON file containing documents as a list of strings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report.json",
        help="Output file path for the JSON report (default: report.json).",
    )
    parser.add_argument(
        "--num-consistency-runs",
        type=int,
        default=1,
        help="Number of verification runs for self-consistency (default: 1).",
    )
    parser.add_argument(
        "--evidence-first",
        action="store_true",
        default=False,
        help="Use evidence-first multi-step prompting (default: False).",
    )
    parser.add_argument(
        "--use-evidence-retrieval",
        action="store_true",
        default=True,
        help="Enable evidence retrieval (default: True).",
    )
    parser.add_argument(
        "--no-evidence-retrieval",
        action="store_true",
        default=False,
        help="Disable evidence retrieval (pass all documents to verifier).",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=None,
        help="Maximum number of claims to verify (limits latency).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens for LLM generation (default: 512).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use MockLLM for testing (no real LLM required).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    # Load inputs
    answer = load_answer(args)
    documents = load_documents(args)

    if args.verbose:
        print(f"Answer: {answer[:100]}...")
        print(f"Documents: {len(documents)} document(s)")
        print(f"Consistency runs: {args.num_consistency_runs}")
        print(f"Evidence-first: {args.evidence_first}")
        print(f"Evidence retrieval: {args.use_evidence_retrieval}")

    # Initialize LLM
    if args.mock:
        llm = MockLLM()
    else:
        # User must provide their own LLM implementation
        print("ERROR: --mock flag not set. You must implement a custom LLM adapter.")
        print("See README.md for integration examples (HuggingFaceLLM, APILLM, etc.)")
        print("For testing, add --mock flag.")
        sys.exit(1)

    # Initialize checker
    retriever = None
    if args.use_evidence_retrieval:
        retriever = EvidenceRetriever(top_k=3)

    checker = RAGFactsChecker(
        llm,
        max_claims=args.max_claims,
        max_new_tokens=args.max_new_tokens,
        num_consistency_runs=args.num_consistency_runs,
        evidence_first=args.evidence_first,
        use_evidence_retrieval=args.use_evidence_retrieval,
        retriever=retriever,
    )

    # Run the check
    if args.verbose:
        print("\nRunning fact-checking pipeline...")

    report = checker.check(answer, documents)

    # Output report
    report_dict = report.to_dict()

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report_dict, indent=2))

    if args.verbose:
        print(f"\nReport written to: {output_path}")
        print(f"Overall confidence: {report.overall_confidence:.1f}%")
        print(f"Overall verdict: {report.overall_verdict}")
        print(f"Claims: {len(report.claims)}")
        print(f"Results: {len(report.results)}")
        print(f"Hallucination flags: {len(report.hallucination_flags)}")
        print(f"\nDimensions:")
        for dim, score in report.dimensions.items():
            print(f"  {dim}: {score}")
    else:
        print(json.dumps(report_dict, indent=2))


if __name__ == "__main__":
    main()
