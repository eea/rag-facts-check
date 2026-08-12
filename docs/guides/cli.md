---
type: Guide
title: CLI
description: Ad-hoc fact-checking from the command line.
tags: [cli, tool, scripts]
timestamp: '2025-01-01T00:00:00Z'
---

# CLI: `scripts/check.py`

Run the fact-checking pipeline on a single dataset from the command line. Useful for ad-hoc debugging, quick checks, and iterating on datasets without writing code.

## Usage

```bash
python scripts/check.py <dataset.json> [options]
```

## Dataset Format

The dataset can be any JSON file with one of these shapes:

**Standard format** (`mock_datasets/`):

```json
{
  "answer": "Paris is the capital of France.",
  "documents": ["Paris is the capital of France and home to the Eiffel Tower."]
}
```

**Sources format** (from `generate-request.json` artifacts):

```json
{
  "answer": "Paris is the capital of France.",
  "sources": [
    {"text": "Paris is the capital...", "title": "Wikipedia"},
    {"text": "France's capital city...", "title": "Britannica"}
  ]
}
```

Both `documents` (string array) and `sources` (dict array with `text` + optional `title`) are supported.

## Options

| Flag | Short | Description |
|---|---|---|
| `dataset` | — | Path to dataset JSON file (required) |
| `--verbose` | `-v` | Show per-claim evidence and explanations |
| `--output <file>` | `-o` | Save full results to a JSON file |
| `--batch-size <n>` | `-b` | Claims per LLM call (default: 1) |

## Examples

**Basic check:**

```bash
python scripts/check.py mock_datasets/climate_change_hallucinated.json
```

**Verbose output with per-claim evidence:**

```bash
python scripts/check.py -v mock_datasets/climate_change_hallucinated.json
```

**Save results to JSON:**

```bash
python scripts/check.py -o results.json mock_datasets/renewable_energy_supported.json
```

**Batch verification (10 claims per LLM call):**

```bash
python scripts/check.py -b 10 artifacts/debug-data/eu-doing-combat-climate-change/generate-request.json
```

## Output

```
============================================================
  climate_change_hallucinated (12.3s)
============================================================
Answer score: 3.2
Verdict:      partially_supported
Confidence:   45.3%
Claims:       5 total, 3 flagged
Dimensions:   groundedness=40.0%, contradiction=20.0%, hallucination=40.0%

Per-claim results:
  [ 1] SUPPORTED       | Global temperatures have risen in the past century.
  [ 2] CONTRADICTED    | Temperatures will rise by 5.7°C by 2100.
  [ 3] NOT ENOUGH INFO | The GCMA reported extreme weather events doubled.
  ...
```

## Environment

The CLI reads LLM configuration from a `.env` file in the project root (or environment variables):

| Variable | Default | Description |
|---|---|---|
| `LLM_API_BASE` | `http://localhost:4002/v1` | Base URL for the LLM API |
| `LLM_MODEL` | `gemma` | Model name |
| `LLM_API_KEY` | `not-needed` | API key (if required) |

The CLI always uses a live LLM — there is no mock mode. For testing without a real LLM, use the pytest test suite instead.

## Output JSON Schema

When using `--output`, the saved JSON matches the `CheckReport.to_dict()` schema. See [Output Format](/architecture/output-format.md) for field details.
