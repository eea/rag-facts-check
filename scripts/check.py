#!/usr/bin/env python3
"""
Run the fact-checking pipeline on a single dataset and print results.

The dataset can be:
  - A generate-request.json from artifacts/debug-data/<session>/
  - A mock dataset (mock_datasets/*.json)
  - Any JSON with ``{"answer": ..., "documents": [...]}`` or
    ``{"answer": ..., "sources": [{"text": ..., "title": ...}, ...]}``

Usage:
    python scripts/check.py artifacts/debug-data/eu-doing-combat-climate-change/generate-request.json
    python scripts/check.py --verbose mock_datasets/climate_change_hallucinated.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import os

from rag_facts_check import AsyncAPILLM, RAGFactsChecker

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> dict:
    """Load a dataset JSON file, normalising to {answer, documents}."""
    with open(path) as f:
        data = json.load(f)

    answer = data.get("answer", "")
    documents = data.get("documents") or data.get("sources") or []

    # Normalise sources (dicts with text/title) to the format checker expects
    normalised = []
    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            normalised.append({"doc_id": doc.get("doc_id", str(i)), "text": doc["text"]})
        else:
            normalised.append({"doc_id": str(i), "text": str(doc)})

    return {"answer": answer, "documents": normalised}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(name: str, report, verbose: bool = False, elapsed: float = 0) -> None:
    d = report.to_dict()
    print(f"\n{'=' * 60}")
    print(f"  {name} ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Answer score: {d.get('answer_score', 'N/A')}")
    print(f"Verdict:      {d['overall_verdict']}")
    print(f"Confidence:   {d['overall_confidence']:.1f}%")
    print(f"Claims:       {len(d['claims'])} total, {len(d['hallucination_flags'])} flagged")
    if d.get("dimensions"):
        dims = d["dimensions"]
        print(
            f"Dimensions:   groundedness={dims['groundedness']:.1f}%, "
            f"contradiction={dims['contradiction_rate']:.1f}%, "
            f"hallucination={dims['hallucination_rate']:.1f}%"
        )

    print(f"\nPer-claim results:")
    for r in d["results"]:
        claim = r["claim"]
        print(f"  [{r['claim_index']:>2}] {r['verdict'].upper():15} | {claim}")

    if d["hallucination_flags"]:
        print(f"\nHallucination flags ({len(d['hallucination_flags'])}):")
        for f_item in d["hallucination_flags"]:
            claim = f_item["claim"]
            print(f"  [!] [{f_item['claim_index']}] {claim}")

    if verbose:
        print(f"\nDetailed evidence:")
        for r in d["results"]:
            print(f"\n  Claim {r['claim_index']}: {r['claim']}")
            print(f"  Verdict:    {r['verdict']}")
            print(f"  Evidence:   {r['evidence']}")
            if r.get("explanation"):
                print(f"  Explanation: {r['explanation']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(dataset_path: str, verbose: bool = False) -> dict:
    data = load_dataset(dataset_path)

    name = Path(dataset_path).stem
    print(f"Dataset: {name}")
    print(f"Answer:  {len(data['answer'])} chars, ~{len(data['answer'].split())} words")
    print(
        f"Docs:    {len(data['documents'])} documents, "
        f"~{sum(len(d['text']) for d in data['documents'])} chars"
    )

    # LLM setup
    base = os.getenv("LLM_API_BASE", "http://localhost:4002/v1")
    url = base.rstrip("/") + "/chat/completions"
    model = os.getenv("LLM_MODEL", "gemma")
    api_key = os.getenv("LLM_API_KEY", "not-needed")

    if verbose:
        print(f"LLM: {model} at {url}")

    llm = AsyncAPILLM(url, model_name=model, api_key=api_key, chat_mode=True)
    checker = RAGFactsChecker(llm)

    t0 = time.time()
    report = await checker.check(answer=data["answer"], documents=data["documents"])
    elapsed = time.time() - t0

    print_report(name, report, verbose=verbose, elapsed=elapsed)
    return report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fact-checking pipeline on a dataset.",
    )
    parser.add_argument("dataset", help="Path to dataset JSON file.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-claim evidence details."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save results to a JSON file.",
    )
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(run(args.dataset, verbose=args.verbose))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
