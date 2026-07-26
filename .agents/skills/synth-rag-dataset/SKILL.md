---
name: synth-rag-dataset
description: >
  Generate synthetic RAG datasets for environmental topics. Each dataset sample
  contains a user question, AI-generated answer, and document chunks retrieved
  by the search retriever. Supports controlled hallucination rates, topic
  selection, and answer difficulty levels. Use when asked to create test data
  for RAG fact-checking, pipeline testing, hallucination detection, or prompt
  iteration on environmental topics.
---

# Synth RAG Dataset

Generate synthetic RAG datasets for environmental topics. Each sample contains:
- A user question about an environmental topic
- Document chunks (source documents retrieved by the search retriever)
- An AI-generated answer (optionally with controlled hallucinations)
- Metadata (topic, hallucination flag, difficulty)

## Prerequisites

- Python 3.10+
- `rag_facts_check` package on the Python path
- An LLM backend (local or API) implementing the `LLM` interface
- For LLM generation: `.env` with `LLM_API_KEY`, `LLM_MODEL`, `LLM_API_BASE`
  (or use `--mock` for testing without a real LLM)

## Quick Start

```bash
# Generate 10 synthetic samples with 30% hallucination rate (MockLLM)
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 10 \
    --hallucination-rate 0.3 \
    --mock \
    -o output/synth-rag-samples.jsonl

# Validate with the fact-checking pipeline
python .agents/skills/rag-facts-check/scripts/run_check.py \
    --answer-file output/synth-rag-samples.jsonl \
    --documents-file output/synth-rag-samples.jsonl \
    --mock --verbose
```

## Workflow

### Step 1: Decide the scenario

Determine what the dataset is needed for:

| Use case | Approach |
|----------|----------|
| Quick structural test | 5-10 samples, `--mock`, default topics |
| Hallucination detection | 20-50 samples, `--hallucination-rate 0.5`, real LLM |
| Pipeline regression | 100+ samples, multiple topics, controlled difficulty |
| Prompt iteration | YAML config with specific topics, hallucination patterns |
| Difficulty testing | `--difficulty easy|medium|hard` to control answer complexity |

### Step 2: Generate the dataset

#### Simple generation (CLI flags)

```bash
# 20 random environmental samples, 40% hallucination rate
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 20 \
    --hallucination-rate 0.4 \
    --mock \
    -o output/synth-rag-20.jsonl

# Specific topics only
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 15 \
    --topics climate_change,renewable_energy,biodiversity \
    --mock \
    -o output/synth-rag-topics.jsonl

# With difficulty control
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 10 \
    --difficulty hard \
    --hallucination-rate 0.5 \
    --mock \
    -o output/synth-rag-hard.jsonl
```

#### YAML config generation (full control)

Create a config file:

```yaml
dataset_name: "env-rag-test"
sample_count: 20
seed: 42

hallucination_rate: 0.35

topics:
  - climate_change
  - renewable_energy
  - biodiversity
  - pollution
  - carbon_emissions

difficulty: medium

num_documents_per_sample: 3
doc_chunk_size: 80  # words per document chunk

hallucination_patterns:
  - "fabricated statistics"
  - "invented dates"
  - "phantom organizations"
  - "cross-topic conflation"

metadata:
  source: "synthetic"
  domain: "environmental"
  version: "1.0"
```

Then run:

```bash
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -c <config.yaml> \
    --mock \
    -o output/synth-rag-config.jsonl
```

#### With a real LLM

```bash
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 20 \
    --hallucination-rate 0.3 \
    --llm-backend api \
    --api-url "http://localhost:8000/v1/completions" \
    --model-name "my-rag-model" \
    -o output/synth-rag-real.jsonl
```

### Step 3: Inspect the output

```bash
# View first 3 samples
head -3 output/synth-rag-samples.jsonl | python3 -m json.tool

# Count hallucinated vs non-hallucinated
python3 -c "
import json
with open('output/synth-rag-samples.jsonl') as f:
    samples = [json.loads(line) for line in f]
h = sum(1 for s in samples if s['metadata']['has_hallucination'])
print(f'Total: {len(samples)}, Hallucinated: {h}, Clean: {len(samples)-h}')
"
```

### Step 4: Run fact-checking

```bash
# Check each sample with the fact-checking pipeline
python .agents/skills/rag-facts-check/scripts/run_check.py \
    --answer-file output/synth-rag-samples.jsonl \
    --documents-file output/synth-rag-samples.jsonl \
    --mock --verbose \
    --num-consistency-runs 3 \
    --evidence-first
```

## Output Convention

**Always write to `output/`** — this is the only valid output directory for generated datasets.

### Output format (JSONL)

Each line is a JSON object:

```json
{
  "question": "What is the projected global temperature increase by 2100?",
  "documents": [
    "The IPCC 2023 report projects a global temperature increase of 1.5°C by 2040 under moderate emission scenarios...",
    "Current climate models estimate a 2-4°C rise by the end of the century if emissions continue at current rates...",
    "Arctic ice sheet data from 2020-2023 shows accelerating melt patterns consistent with warming trends..."
  ],
  "answer": "The global temperature is projected to increase by 3.2°C by 2100, according to the latest IPCC projections.",
  "metadata": {
    "topic": "climate_change",
    "has_hallucination": true,
    "hallucination_type": "fabricated_statistics",
    "difficulty": "medium",
    "num_documents": 3,
    "sample_id": "climate_change_001"
  }
}
```

## Topic Reference

| Topic | Description | Example question |
|-------|-------------|-----------------|
| `climate_change` | Global warming, temperature trends, climate models | "What is the projected temperature increase by 2100?" |
| `renewable_energy` | Solar, wind, hydro, geothermal adoption | "What percentage of global electricity comes from renewables?" |
| `biodiversity` | Species extinction, habitat loss, conservation | "How many species went extinct in 2023?" |
| `pollution` | Air, water, soil contamination | "What are the main sources of ocean plastic?" |
| `carbon_emissions` | CO2 emissions, carbon footprint, sequestration | "Which countries are the largest CO2 emitters?" |
| `sustainable_agriculture` | Regenerative farming, soil health, food systems | "How does regenerative agriculture reduce emissions?" |
| `ocean_conservation` | Marine protected areas, coral bleaching, fisheries | "What percentage of oceans are protected?" |
| `forest_protection` | Deforestation, reforestation, carbon sinks | "How much forest was lost in the Amazon last year?" |
| `air_quality` | Particulate matter, pollution sources, health impacts | "What are the health effects of PM2.5?" |
| `water_resources` | Freshwater availability, drought, water stress | "Which countries face severe water stress?" |

## Configuration Reference

### Top-level config fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | string | `env-rag-synth` | Name for the dataset |
| `sample_count` | int | 10 | Number of samples to generate |
| `seed` | int | — | Random seed for reproducibility |
| `hallucination_rate` | float | 0.3 | Fraction of samples with hallucinated answers |
| `topics` | list[str] | all 10 | Topics to sample from |
| `difficulty` | string | `medium` | Answer difficulty: `easy`, `medium`, `hard` |
| `num_documents_per_sample` | int | 3 | Number of document chunks per sample |
| `doc_chunk_size` | int | 80 | Target word count per document chunk |
| `hallucination_patterns` | list[str] | default | Types of hallucinations to inject |
| `metadata` | dict | — | Extra metadata to include in each sample |

### CLI flags

| Flag | Description |
|------|-------------|
| `-n, --sample-count` | Number of samples to generate |
| `--hallucination-rate` | Fraction of samples with hallucinations (0.0-1.0) |
| `--topics` | Comma-separated list of topics |
| `--difficulty` | `easy`, `medium`, or `hard` |
| `--num-docs` | Number of document chunks per sample |
| `--mock` | Use MockLLM (no real LLM required) |
| `--llm-backend` | `mock`, `hf`, or `api` |
| `--api-url` | API endpoint for APILLM |
| `--model-name` | Model name for APILLM |
| `--seed` | Random seed |
| `-c, --config` | YAML config file |
| `-o, --output` | Output file path (JSONL) |
| `--verbose`, `-v` | Verbose output |

### Valid hallucination patterns

| Pattern | Description |
|---------|-------------|
| `fabricated_statistics` | Invented numbers, percentages, or data points |
| `invented_dates` | Made-up dates or timeframes |
| `phantom_organizations` | Non-existent organizations or entities |
| `cross_topic_conflation` | Facts from one topic attributed to another |
| `exaggerated_claims` | Overstated impact or severity |
| `temporal_hallucinations` | Incorrect timeframes or sequences |
| `causal_hallucinations` | Invented cause-effect relationships |
| `negative_claims` | Statements that something "did not happen" when docs are silent |

## Example Scenarios

### Minimal 5-sample test

```bash
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 5 --mock -o output/synth-minimal.jsonl
```

### Hallucination detection benchmark

```bash
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 50 \
    --hallucination-rate 0.5 \
    --difficulty hard \
    --mock \
    -o output/synth-hallucination-bench.jsonl
```

### Full-topic dataset

```bash
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -n 100 \
    --hallucination-rate 0.3 \
    --topics climate_change,renewable_energy,biodiversity,pollution,carbon_emissions \
    --difficulty medium \
    --mock \
    -o output/synth-full.jsonl
```

### YAML config with specific patterns

```bash
python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
    -c synth-rag-dataset/examples/hard-hallucinations.yaml \
    --mock \
    -o output/synth-hard.jsonl
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: rag_facts_check` | Ensure the package is on the Python path or install with `pip install -e .` |
| MockLLM produces repetitive data | Use a real LLM backend (`--llm-backend api` or `--llm-backend hf`) |
| Generated answers don't match documents | Check `--hallucination-rate` — higher rates produce more divergence |
| JSONL output is malformed | Run with `--verbose` to see per-sample generation; check LLM output format |
| Topics not recognized | Use exact topic names from the Topic Reference table |
| All samples have same difficulty | Check `--difficulty` flag; `hard` requires more complex reasoning |

## See Also

- [SKILL.md](../rag-facts-check/SKILL.md) — Fact-checking skill for reviewing generated answers
- [debug-check.md](../../docs/debug-prompts/debug-check.md) — Debug command for the fact-checking pipeline
- [LLM interface](../../rag_facts_check/llm.py) — LLM backend adapters (HF, API, Chat, Mock)
