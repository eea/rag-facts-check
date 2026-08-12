# RAG Facts Check

A modular system for verifying RAG-generated answers against their source documents using claim extraction + per-claim verification.

## How It Works

1. **Extract** atomic factual claims from the generated answer
2. **Retrieve** relevant document chunks for each claim (optional)
3. **Verify** each claim against the source documents
4. **Aggregate** results into a confidence score, verdict, and detailed report

## Quick Start

The primary entry point is the FastAPI web service. It accepts RAG answers and source documents, extracts claims, verifies each against the sources, and returns a detailed report.

```bash
# Development (auto-reload)
make serve

# Production
docker build -t rag-fact-check .
docker run -p 8000:8000 rag-fact-check
```

See [Web Service](docs/guides/web-service.md) for endpoint details and request/response schemas.

## Documentation

Full documentation is in [`docs/`](docs/index.md):

- **[Overview](docs/overview/overview.md)** — what the project is and how it works
- **[Fact-Checking Approaches](docs/overview/approaches.md)** — survey of possible strategies
- **[Architecture](docs/architecture/architecture.md)** — module layout and data flow
- **[CLI](docs/guides/cli.md)** — ad-hoc fact-checking from the command line
- **[LLM Integration](docs/guides/llm-integration.md)** — plugging in your model
- **[Configuration](docs/guides/configuration.md)** — tuning parameters
- **[Web Service](docs/guides/web-service.md)** — FastAPI endpoints
- **[Testing](docs/guides/testing.md)** — test suite and AsyncMock fixtures

## Requirements

- Python 3.10+

No core dependencies beyond the standard library and `requests`. Install optional groups as needed:

```bash
pip install -e ".[test,dev]"       # pytest, ruff
pip install -e ".[test,dev,server]" # + fastapi, uvicorn, atomic-agents, instructor
```

## License

MIT
