#!/usr/bin/env python3
"""
Generate synthetic RAG datasets for environmental topics.

Each sample contains:
- A user question about an environmental topic
- Document chunks (source documents)
- An AI-generated answer (optionally with controlled hallucinations)
- Metadata (topic, hallucination flag, difficulty)

Requires a real LLM backend. Configure via .env file (copy .env.example).

Usage:
    python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
        -n 10 --hallucination-rate 0.3 -o output/synth-rag.jsonl

    python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
        -n 20 --hallucination-rate 0.4 --topics climate_change,renewable_energy \
        --difficulty hard -o output/synth-rag.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
# Script is at: <root>/.agents/skills/synth-rag-dataset/scripts/generate_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rag_facts_check import LLM


# ─── Environment Loading ─────────────────────────────────────────────────────

def load_env(env_path: Optional[str] = None) -> dict:
    """Load environment variables from a .env file.

    Reads KEY=VALUE pairs (ignoring comments and blank lines).
    Does NOT require python-dotenv.
    """
    env_vars = {}
    if env_path is None:
        # Default: look for .env in the project root (4 levels up from this script)
        env_path = str(Path(__file__).resolve().parents[4] / ".env")

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


def get_llm_from_env(env_vars: dict, api_url: Optional[str] = None,
                     model_name: Optional[str] = None) -> "LLM":
    """Create an APILLM instance from environment variables.

    Reads LLM_API_BASE, LLM_API_KEY, LLM_MODEL from env_vars.
    Command-line arguments override environment values.
    """
    from rag_facts_check import APILLM

    base_url = api_url or env_vars.get("LLM_API_BASE", "http://localhost:8000/v1")
    model = model_name or env_vars.get("LLM_MODEL", "gemma")
    api_key = env_vars.get("LLM_API_KEY")

    # Build the completions URL
    if not base_url.endswith("/completions") and not base_url.endswith("/chat/completions"):
        final_url = base_url.rstrip("/") + "/chat/completions"
        chat_mode = True
    elif "/chat/completions" in base_url:
        final_url = base_url
        chat_mode = True
    else:
        final_url = base_url
        chat_mode = False

    return APILLM(final_url, model_name=model, api_key=api_key, chat_mode=chat_mode)


# ─── Environmental Topics ────────────────────────────────────────────────────

TOPICS = [
    "climate_change",
    "renewable_energy",
    "biodiversity",
    "pollution",
    "carbon_emissions",
    "sustainable_agriculture",
    "ocean_conservation",
    "forest_protection",
    "air_quality",
    "water_resources",
]

TOPIC_DESCRIPTIONS = {
    "climate_change": "global warming, temperature trends, climate models, carbon targets",
    "renewable_energy": "solar, wind, hydro, geothermal, energy storage, grid integration",
    "biodiversity": "species extinction, habitat loss, conservation efforts, endangered species",
    "pollution": "air, water, soil contamination, plastic waste, industrial emissions",
    "carbon_emissions": "CO2 output, carbon footprint, emission reduction targets, carbon accounting",
    "sustainable_agriculture": "regenerative farming, soil health, food systems, sustainable practices",
    "ocean_conservation": "marine protected areas, coral bleaching, fisheries, ocean acidification",
    "forest_protection": "deforestation, reforestation, carbon sinks, protected areas",
    "air_quality": "particulate matter, PM2.5, air pollution sources, health impacts",
    "water_resources": "freshwater availability, drought, water stress, conservation",
}

# ─── Generation Prompts ───────────────────────────────────────────────────────

QUESTION_PROMPT_TEMPLATE = """You are an expert in environmental science. Generate a specific, factual question about the following environmental topic: {topic}

The topic context: {description}

The question should be answerable using factual information about the topic. It should not be a yes/no question. It should be specific enough to require looking up data.

Question:"""

DOCUMENT_PROMPT_TEMPLATE = """You are an expert in environmental science. Generate {num_docs} short document chunks (about {chunk_size} words each) about the following topic: {topic}

The topic context: {description}

These documents should contain factual information that could answer this question:
"{question}"

Include specific data, dates, statistics, and facts. Each chunk should be self-contained and factual.

Format as:
DOC 1: <text>
DOC 2: <text>
DOC 3: <text>"""

ANSWER_PROMPT_TEMPLATE = """You are a helpful assistant. Given the following question and source documents, generate a factual answer based ONLY on the information in the documents.

Question: {question}

Source Documents:
{documents}

Answer:"""

HALLUCINATED_ANSWER_PROMPT_TEMPLATE = """You are a helpful assistant. Given the following question and source documents, generate an answer that contains at least one factual claim NOT supported by the documents. The answer should sound plausible and confident, but include at least one hallucinated fact (e.g., a fabricated statistic, invented date, or phantom organization).

Question: {question}

Source Documents:
{documents}

Answer:"""


# ─── Dataset Generator ────────────────────────────────────────────────────────

class DatasetGenerator:
    """Generates synthetic RAG datasets for environmental topics."""

    def __init__(
        self,
        llm: LLM,
        topics: List[str] = None,
        hallucination_rate: float = 0.3,
        difficulty: str = "medium",
        num_docs: int = 3,
        doc_chunk_size: int = 80,
        seed: int = 42,
    ):
        self.llm = llm
        self.topics = topics or TOPICS
        self.hallucination_rate = hallucination_rate
        self.difficulty = difficulty
        self.num_docs = num_docs
        self.doc_chunk_size = doc_chunk_size
        self.seed = seed
        random.seed(seed)

    def generate_sample(self, topic: str = None) -> dict:
        """Generate a single synthetic RAG sample."""
        topic = topic or random.choice(self.topics)
        description = TOPIC_DESCRIPTIONS.get(topic, topic)

        # Generate question
        question_prompt = QUESTION_PROMPT_TEMPLATE.format(
            topic=topic, description=description
        )
        question = self.llm.generate(question_prompt, max_new_tokens=128, temperature=0.7)

        # Generate documents (include question so docs are relevant)
        doc_prompt = DOCUMENT_PROMPT_TEMPLATE.format(
            topic=topic,
            description=description,
            num_docs=self.num_docs,
            chunk_size=self.doc_chunk_size,
            question=question,
        )
        doc_response = self.llm.generate(doc_prompt, max_new_tokens=512, temperature=0.7)
        documents = self._parse_documents(doc_response)

        # Decide if this sample should have a hallucination
        has_hallucination = random.random() < self.hallucination_rate

        # Generate answer
        if has_hallucination:
            answer_prompt = HALLUCINATED_ANSWER_PROMPT_TEMPLATE.format(
                question=question,
                documents="\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(documents)),
            )
        else:
            answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
                question=question,
                documents="\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(documents)),
            )

        answer = self.llm.generate(answer_prompt, max_new_tokens=256, temperature=0.7)

        # Build metadata
        hallucination_type = "fabricated_statistics" if has_hallucination else None

        return {
            "question": question.strip(),
            "documents": documents,
            "answer": answer.strip(),
            "metadata": {
                "topic": topic,
                "has_hallucination": has_hallucination,
                "hallucination_type": hallucination_type,
                "difficulty": self.difficulty,
                "num_documents": len(documents),
                "sample_id": f"{topic}_{self._sample_counter():04d}",
            },
        }

    def _sample_counter(self) -> int:
        """Return a counter for sample IDs."""
        if not hasattr(self, "_counter"):
            self._counter = 0
        self._counter += 1
        return self._counter

    def _parse_documents(self, response: str) -> List[str]:
        """Parse document chunks from the LLM response."""
        docs = []
        lines = response.strip().split("\n")
        for line in lines:
            match = re.match(r"^DOC\s+\d+:\s*(.+)$", line, re.IGNORECASE)
            if match:
                docs.append(match.group(1).strip())
        # If no docs were parsed, treat the entire response as one document
        if not docs and response.strip():
            docs = [response.strip()]
        return docs

    def generate_dataset(self, sample_count: int) -> List[dict]:
        """Generate a dataset of synthetic samples."""
        samples = []
        for i in range(sample_count):
            topic = random.choice(self.topics)
            sample = self.generate_sample(topic)
            samples.append(sample)
            if hasattr(self.llm, "call_count"):
                pass  # MockLLM tracks call count
        return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RAG datasets for environmental topics."
    )
    parser.add_argument(
        "-n", "--sample-count",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10).",
    )
    parser.add_argument(
        "--hallucination-rate",
        type=float,
        default=0.3,
        help="Fraction of samples with hallucinated answers (0.0-1.0, default: 0.3).",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=",".join(TOPICS),
        help=f"Comma-separated list of topics (default: all {len(TOPICS)} topics).",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Answer difficulty level (default: medium).",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=3,
        help="Number of document chunks per sample (default: 3).",
    )
    parser.add_argument(
        "--doc-chunk-size",
        type=int,
        default=80,
        help="Target word count per document chunk (default: 80).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["env", "hf", "api"],
        default="env",
        help="LLM backend to use (default: env, reads from .env file).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file (default: <project-root>/.env).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API endpoint for APILLM backend (overrides .env).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for APILLM backend (overrides .env).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/synth-rag-dataset.jsonl",
        help="Output file path (JSONL format).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    # Parse topics
    topics = [t.strip() for t in args.topics.split(",")]
    for t in topics:
        if t not in TOPICS:
            print(f"ERROR: Unknown topic '{t}'. Valid topics: {', '.join(TOPICS)}")
            sys.exit(1)

    # Initialize LLM
    if args.llm_backend == "env":
        env_vars = load_env(args.env_file)
        if not env_vars:
            print("ERROR: No .env file found. Copy .env.example to .env and configure.")
            sys.exit(1)
        if args.verbose:
            print(f"Loaded .env: API_BASE={env_vars.get('LLM_API_BASE', 'N/A')}, MODEL={env_vars.get('LLM_MODEL', 'N/A')}")
        llm = get_llm_from_env(env_vars, api_url=args.api_url, model_name=args.model_name)
    elif args.llm_backend == "api":
        from rag_facts_check import APILLM
        llm = APILLM(args.api_url, model_name=args.model_name)
    elif args.llm_backend == "hf":
        from rag_facts_check import HuggingFaceLLM
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("ERROR: --llm-backend hf requires manual setup. See README.")
        sys.exit(1)

    # Initialize generator
    generator = DatasetGenerator(
        llm=llm,
        topics=topics,
        hallucination_rate=args.hallucination_rate,
        difficulty=args.difficulty,
        num_docs=args.num_docs,
        doc_chunk_size=args.doc_chunk_size,
        seed=args.seed,
    )

    # Generate dataset
    if args.verbose:
        print(f"Generating {args.sample_count} samples...")
        print(f"Topics: {', '.join(topics)}")
        print(f"Hallucination rate: {args.hallucination_rate}")
        print(f"Difficulty: {args.difficulty}")
        print(f"Documents per sample: {args.num_docs}")
        print()

    samples = generator.generate_dataset(args.sample_count)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if len(samples) == 1:
            # Single sample: pretty-printed JSON
            json.dump(samples[0], f, indent=2)
            f.write("\n")
        else:
            # Multiple samples: JSONL
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    # Summary
    h_count = sum(1 for s in samples if s["metadata"]["has_hallucination"])
    c_count = len(samples) - h_count

    if args.verbose:
        print(f"Dataset written to: {output_path}")
        print(f"Total samples: {len(samples)}")
        print(f"  Hallucinated: {h_count} ({h_count/len(samples)*100:.0f}%)")
        print(f"  Clean: {c_count} ({c_count/len(samples)*100:.0f}%)")
        print(f"\nTopic distribution:")
        from collections import Counter
        topic_counts = Counter(s["metadata"]["topic"] for s in samples)
        for topic, count in sorted(topic_counts.items()):
            print(f"  {topic}: {count}")
    else:
        print(f"Generated {len(samples)} samples ({h_count} hallucinated) -> {output_path}")


if __name__ == "__main__":
    main()
