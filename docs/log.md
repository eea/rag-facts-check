# Documentation Change Log

## 2025-01-01
* **Restructure**: Refactored documentation to [OKF v0.1](../reference/knowledge-catalog/okf/SPEC.md) format. Split monolithic `README.md` into concept documents under `docs/`.
* **Creation**: Added [Project Overview](/overview/overview.md) as the high-level introduction.
* **Creation**: Added [Fact-Checking Approaches](/overview/approaches.md) clarifying that the table lists *possible* approaches, not all implemented here.
* **Creation**: Added [System Architecture](/architecture/architecture.md) documenting module layout and component responsibilities.
* **Creation**: Added [Data Flow](/architecture/data-flow.md) with Mermaid diagram of the pipeline.
* **Creation**: Added [Output Format](/architecture/output-format.md) documenting CheckReport structure and verdicts.
* **Creation**: Added [LLM Integration](/guides/llm-integration.md) for plugging in models.
* **Creation**: Added [Configuration](/guides/configuration.md) for checker tuning parameters.
* **Creation**: Added [Web Service](/guides/web-service.md) for FastAPI endpoint reference.
* **Creation**: Added [Testing](/guides/testing.md) for test suite and MockLLM usage.
* **Creation**: Added [docs/index.md](index.md) as the OKF directory listing.
* **Update**: Added YAML frontmatter to [Architecture Tension](architecture-tension.md) and [Debug Command](debug-prompts/debug-check.md).
* **Update**: Replaced ASCII art diagrams with Mermaid in [Data Flow](/architecture/data-flow.md).
* **Update**: Root `README.md` shortened to brief overview linking to `docs/index.md`.
* **Creation**: Added [docs/AGENTS.md](AGENTS.md) with documentation contributing conventions.
