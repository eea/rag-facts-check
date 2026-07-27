#!/usr/bin/env python3
"""
Check a mock dataset against the fact-checking pipeline using a real LLM.

Usage:
    python scripts/check_dataset.py mock_datasets/climate_change_hallucinated.json
    python scripts/check_dataset.py --verbose mock_datasets/*.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_facts_check import APILLM, RAGFactsChecker


def load_env(env_path: str = ".env") -> dict:
    """Load environment variables from a .env file."""
    env_vars = {}
    path = Path(env_path)
    if not path.exists():
        return env_vars
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()
    return env_vars


def check_dataset(dataset_path: str, verbose: bool = False):
    """Check a dataset using real LLM from .env."""
    with open(dataset_path) as f:
        data = json.load(f)
    env = load_env()
    if not env:
        print("ERROR: No .env file found. Copy .env.example to .env and configure.")
        sys.exit(1)

    base = env.get("LLM_API_BASE", "http://localhost:4002/v1")
    url = base.rstrip("/") + "/chat/completions"
    model = env.get("LLM_MODEL", "gemma")
    api_key = env.get("LLM_API_KEY")

    if verbose:
        print(f"Using LLM: {model} at {url}")

    llm = APILLM(url, model_name=model, api_key=api_key, chat_mode=True)
    checker = RAGFactsChecker(llm)
    report = checker.check(answer=data["answer"], documents=data["documents"])
    return report


def print_report(name: str, report, verbose: bool = False):
    """Print a formatted report."""
    d = report.to_dict()
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"Verdict:    {d['overall_verdict']}")
    print(f"Confidence: {d['overall_confidence']:.1f}%")
    print(f"Claims:     {len(d['claims'])} total, {len(d['hallucination_flags'])} flagged")
    print(f"Dimensions: {d['dimensions']}")

    print("\nPer-claim results:")
    for r in d["results"]:
        claim_preview = r["claim"][:75] + "..." if len(r["claim"]) > 75 else r["claim"]
        verdict = r["verdict"].upper()
        print(f"  [{r['claim_index']}] {verdict:15} conf={r['confidence']:3}% | {claim_preview}")

    if d["hallucination_flags"]:
        print("\nHallucination flags:")
        for f in d["hallucination_flags"]:
            claim_preview = f["claim"][:65] + "..." if len(f["claim"]) > 65 else f["claim"]
            print(f"  ⚠️  [{f['claim_index']}] {claim_preview}")

    if verbose:
        print("\nDetailed evidence:")
        for r in d["results"]:
            print(f"\n  Claim {r['claim_index']}: {r['claim'][:80]}")
            print(f"  Verdict:    {r['verdict']}")
            print(f"  Confidence: {r['confidence']}%")
            print(f"  Evidence:   {r['evidence'][:120]}")
            if r.get("explanation"):
                print(f"  Explanation: {r['explanation'][:150]}")


def main():
    parser = argparse.ArgumentParser(description="Check a dataset with real LLM.")
    parser.add_argument("datasets", nargs="+", help="Dataset JSON file(s) to check.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    for dataset_path in args.datasets:
        name = Path(dataset_path).stem
        try:
            report = check_dataset(dataset_path, verbose=args.verbose)
            print_report(name, report, verbose=args.verbose)
        except Exception as e:
            print(f"ERROR checking {dataset_path}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
