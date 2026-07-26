#!/usr/bin/env python3
"""
Generate synthetic RAG datasets for environmental topics.

Each sample contains:
- A user question about an environmental topic
- Document chunks (source documents)
- An AI-generated answer (optionally with controlled hallucinations)
- Metadata (topic, hallucination flag, difficulty)

Usage:
    python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
        -n 10 --hallucination-rate 0.3 --mock -o output/synth-rag.jsonl

    python .agents/skills/synth-rag-dataset/scripts/generate_dataset.py \
        -n 20 --hallucination-rate 0.4 --topics climate_change,renewable_energy \
        --difficulty hard --mock -o output/synth-rag.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
# Script is at: <root>/.agents/skills/synth-rag-dataset/scripts/generate_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rag_facts_check import LLM


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

# ─── Mock Data for Testing ────────────────────────────────────────────────────

MOCK_QUESTIONS = {
    "climate_change": "What is the projected global temperature increase by 2100?",
    "renewable_energy": "What percentage of global electricity comes from renewable sources?",
    "biodiversity": "How many species went extinct in 2023 according to the IUCN Red List?",
    "pollution": "What are the primary sources of ocean plastic pollution?",
    "carbon_emissions": "Which countries are the largest CO2 emitters globally?",
    "sustainable_agriculture": "How does regenerative agriculture reduce carbon emissions?",
    "ocean_conservation": "What percentage of ocean areas are designated as marine protected areas?",
    "forest_protection": "How much Amazon rainforest was lost to deforestation in 2023?",
    "air_quality": "What are the health impacts of PM2.5 exposure?",
    "water_resources": "Which countries face the most severe water stress?",
}

MOCK_DOCUMENTS = {
    "climate_change": [
        "The IPCC Sixth Assessment Report (2023) projects a global temperature increase of 1.5°C by 2040 under moderate emission scenarios (SSP2-4.5).",
        "Current climate models estimate a 2-4°C rise by the end of the century if emissions continue at current rates (SSP3-7.0).",
        "Arctic sea ice extent has declined by approximately 13% per decade since satellite observations began in 1979.",
    ],
    "renewable_energy": [
        "In 2023, renewable energy sources accounted for 30% of global electricity generation, with solar and wind leading the growth.",
        "The International Renewable Energy Agency (IRENA) projects renewables will reach 42% of global electricity by 2028.",
        "Battery storage costs have fallen by 89% since 2010, enabling greater grid integration of variable renewables.",
    ],
    "biodiversity": [
        "The IUCN Red List identifies 159 species as Extinct or Extinct in the Wild as of 2023.",
        "Habitat loss and degradation are the primary drivers of biodiversity decline, affecting 85% of threatened species.",
        "The Living Planet Index reports an average 69% decline in monitored wildlife populations between 1970 and 2020.",
    ],
    "pollution": [
        "An estimated 80% of marine pollution originates from land-based sources, with plastic accounting for 73% of that total.",
        "The five most polluted rivers — Yangtze, Ganges, Nile, Mekong, and Indus — carry approximately 67% of global riverine plastic discharge.",
        "Microplastics have been detected in 94% of tap water samples and 83% of bottled water samples globally.",
    ],
    "carbon_emissions": [
        "China remains the world's largest CO2 emitter, accounting for 31% of global fossil CO2 emissions in 2023.",
        "The United States and India follow as the second and third largest emitters, contributing 13% and 8% respectively.",
        "Global CO2 emissions from energy combustion reached 36.8 billion tonnes in 2023, a 1.1% increase from 2022.",
    ],
    "sustainable_agriculture": [
        "Regenerative agriculture practices can sequester up to 4.3 billion tonnes of CO2 annually in global soils.",
        "Cover cropping and no-till farming improve soil organic matter by 0.1-0.6% per year under optimal conditions.",
        "The Rodale Institute's research shows regenerative grazing can increase soil carbon by 1,000 pounds per acre annually.",
    ],
    "ocean_conservation": [
        "As of 2023, 8.2% of the world's ocean area is designated as marine protected areas (MPAs), covering 14.2 million km².",
        "The UN High Seas Treaty, adopted in 2023, aims to protect 30% of oceans by 2030 under the 30x30 initiative.",
        "Coral bleaching events in 2023 affected 75% of the world's coral reefs, primarily due to marine heatwaves.",
    ],
    "forest_protection": [
        "Amazon deforestation in 2023 totaled 11,088 km², representing a 22% decrease from the previous year.",
        "The Great Green Wall initiative has restored 18 million hectares of degraded land across the Sahel since 2007.",
        "Primary forest loss globally reached 4.7 million hectares in 2023, with the Amazon, Congo Basin, and Southeast Asia most affected.",
    ],
    "air_quality": [
        "Long-term exposure to PM2.5 concentrations above 15 μg/m³ increases the risk of cardiovascular disease by 13% per 10 μg/m³.",
        "WHO's updated air quality guidelines recommend an annual mean PM2.5 level of 5 μg/m³, down from the previous 10 μg/m³.",
        "Indoor air pollution from solid fuel use affects 2.6 billion people globally, causing 3.2 million premature deaths annually.",
    ],
    "water_resources": [
        "Over 2 billion people live in countries experiencing high water stress, with 40% of the global population projected to face water scarcity by 2050.",
        "The World Resources Institute identifies India, China, and the United States as having the highest total water withdrawal.",
        "Groundwater depletion rates exceed recharge rates in 20% of aquifers globally, threatening long-term water security.",
    ],
}

MOCK_ANSWERS = {
    "climate_change": "The global temperature is projected to increase by 3.2°C by 2100, according to the latest IPCC projections.",
    "renewable_energy": "Renewable energy currently provides 45% of global electricity, with solar leading at 22% of total generation.",
    "biodiversity": "A total of 237 species went extinct in 2023, primarily due to habitat destruction in tropical regions.",
    "pollution": "Plastic production reached 500 million tonnes annually, with 80% ending up in landfills or the environment.",
    "carbon_emissions": "India is the world's largest CO2 emitter, surpassing China with 35% of global emissions in 2023.",
    "sustainable_agriculture": "Regenerative agriculture can sequester up to 10 billion tonnes of CO2 annually, completely offsetting global emissions.",
    "ocean_conservation": "Marine protected areas now cover 15% of the world's oceans, with 50% of coral reefs successfully restored.",
    "forest_protection": "Amazon deforestation has increased by 40% in 2023, reaching 15,000 km² of forest loss.",
    "air_quality": "PM2.5 exposure causes 8.9 million premature deaths annually, making it the leading environmental health risk.",
    "water_resources": "By 2050, 60% of the global population will live in areas of high water stress, up from 20% currently.",
}

# Hallucinated versions (contain fabricated facts)
MOCK_HALLUCINATED_ANSWERS = {
    "climate_change": "The global temperature is projected to increase by 5.7°C by 2100, according to NASA's 2024 report.",
    "renewable_energy": "Renewable energy currently provides 78% of global electricity, with solar alone accounting for 52% of total generation.",
    "biodiversity": "A total of 542 species went extinct in 2023, with amphibians and reptiles most severely affected.",
    "pollution": "Plastic production reached 2 billion tonnes annually, with 95% ending up in the ocean.",
    "carbon_emissions": "Brazil is the world's largest CO2 emitter, responsible for 42% of global emissions in 2023.",
    "sustainable_agriculture": "Regenerative agriculture can sequester up to 25 billion tonnes of CO2 annually, eliminating all agricultural emissions.",
    "ocean_conservation": "Marine protected areas now cover 25% of the world's oceans, with 80% of coral reefs successfully restored.",
    "forest_protection": "Amazon deforestation has increased by 67% in 2023, reaching 22,000 km² of forest loss.",
    "air_quality": "PM2.5 exposure causes 15.3 million premature deaths annually, making it the leading cause of death globally.",
    "water_resources": "By 2050, 85% of the global population will live in areas of complete water scarcity.",
}

# Hallucination types per topic
HALLUCINATION_TYPES = {
    "climate_change": "fabricated_statistics",
    "renewable_energy": "exaggerated_claims",
    "biodiversity": "fabricated_statistics",
    "pollution": "exaggerated_claims",
    "carbon_emissions": "phantom_organizations",
    "sustainable_agriculture": "fabricated_statistics",
    "ocean_conservation": "exaggerated_claims",
    "forest_protection": "fabricated_statistics",
    "air_quality": "fabricated_statistics",
    "water_resources": "exaggerated_claims",
}


class MockGenerationLLM(LLM):
    """Mock LLM for dataset generation. Returns predefined responses."""

    def __init__(self):
        self.call_count = 0
        self.current_topic = "climate_change"

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.1, **kwargs) -> str:
        self.call_count += 1
        lower = prompt.lower()

        # Question generation
        if "generate a specific, factual question" in lower:
            topic = self._extract_topic(prompt)
            self.current_topic = topic
            return MOCK_QUESTIONS.get(topic, "What are the key environmental challenges in this area?")

        # Answer generation (accurate) — check BEFORE document generation
        # because the answer prompt contains documents that mention "DOC"
        if "factual answer" in lower:
            return MOCK_ANSWERS.get(self.current_topic, "The data indicates significant environmental changes.")

        # Answer generation (hallucinated)
        if "hallucinated fact" in lower or "not supported by the documents" in lower:
            return MOCK_HALLUCINATED_ANSWERS.get(self.current_topic, "The data shows unprecedented environmental changes.")

        # Document generation — check for "short document chunks" to avoid
        # matching answer prompts that contain documents with "DOC" markers
        if "short document chunks" in lower or "generate" in lower and "document chunks" in lower:
            topic = self._extract_topic(prompt)
            self.current_topic = topic
            docs = MOCK_DOCUMENTS.get(topic, MOCK_DOCUMENTS["climate_change"])
            num_docs = 3
            lines = []
            for i in range(min(num_docs, len(docs))):
                lines.append(f"DOC {i + 1}: {docs[i]}")
            return "\n".join(lines)

        return f"Mock generation response."

    def _extract_topic(self, prompt: str) -> str:
        """Extract the topic from the prompt."""
        for topic in TOPICS:
            if topic in prompt.lower():
                return topic
        return "climate_change"


# ─── Generation Prompts ───────────────────────────────────────────────────────

QUESTION_PROMPT_TEMPLATE = """You are an expert in environmental science. Generate a specific, factual question about the following environmental topic: {topic}

The topic context: {description}

The question should be answerable using factual information about the topic. It should not be a yes/no question. It should be specific enough to require looking up data.

Question:"""

DOCUMENT_PROMPT_TEMPLATE = """You are an expert in environmental science. Generate {num_docs} short document chunks (about {chunk_size} words each) about the following topic: {topic}

The topic context: {description}

The documents should contain factual information that could answer questions about this topic. Include specific data, dates, statistics, and facts. Each chunk should be self-contained and factual.

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

        # Generate documents
        doc_prompt = DOCUMENT_PROMPT_TEMPLATE.format(
            topic=topic,
            description=description,
            num_docs=self.num_docs,
            chunk_size=self.doc_chunk_size,
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
        hallucination_type = HALLUCINATION_TYPES.get(topic, "fabricated_statistics") if has_hallucination else None

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
        "--mock",
        action="store_true",
        default=False,
        help="Use MockGenerationLLM for testing (no real LLM required).",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["mock", "hf", "api"],
        default="mock",
        help="LLM backend to use (default: mock).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="API endpoint for APILLM backend.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for APILLM backend.",
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
    if args.mock or args.llm_backend == "mock":
        llm = MockGenerationLLM()
    elif args.llm_backend == "api":
        from rag_facts_check import APILLM
        llm = APILLM(args.api_url, model_name=args.model_name)
    elif args.llm_backend == "hf":
        from rag_facts_check import HuggingFaceLLM
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # User must specify model
        print("ERROR: --llm-backend hf requires a model. See README for details.")
        sys.exit(1)
    else:
        llm = MockGenerationLLM()

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
