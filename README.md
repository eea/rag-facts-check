# RAG Facts Check

A modular system for verifying RAG-generated answers against their source documents using claim extraction + per-claim verification.

## How It Works

1. **Extract** atomic factual claims from the generated answer
2. **Retrieve** relevant document chunks for each claim (optional)
3. **Verify** each claim against the source documents
4. **Aggregate** results into a confidence score, verdict, and detailed report

## Quick Start

```python
import asyncio
from rag_facts_check import RAGFactsChecker, MockLLM

async def main():
    llm = MockLLM()
    checker = RAGFactsChecker(llm)

    answer = "Paris is the capital of France. The Eiffel Tower was built in 1889."
    documents = [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "The Eiffel Tower was constructed between 1887 and 1889.",
    ]

    report = await checker.check(answer, documents)
    print(report.to_dict())

asyncio.run(main())
```

## Documentation

Full documentation is in [`docs/`](docs/index.md):

- **[Overview](docs/overview/overview.md)** — what the project is and how it works
- **[Fact-Checking Approaches](docs/overview/approaches.md)** — survey of possible strategies
- **[Architecture](docs/architecture/architecture.md)** — module layout and data flow
- **[LLM Integration](docs/guides/llm-integration.md)** — plugging in your model
- **[Configuration](docs/guides/configuration.md)** — tuning parameters
- **[Web Service](docs/guides/web-service.md)** — FastAPI endpoints
- **[Testing](docs/guides/testing.md)** — test suite and MockLLM

## Requirements

- Python 3.10+

No core dependencies beyond the standard library and `requests`. Install optional groups as needed:

```bash
pip install -e ".[test,dev]"       # pytest, ruff
pip install -e ".[test,dev,server]" # + fastapi, uvicorn, atomic-agents, instructor
```

## License

MIT
