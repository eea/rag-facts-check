# RAG Facts Check — Documentation Index

## Getting Started

* [Project Overview](/overview/overview.md) — What RAG Facts Check is and how it works.
* [CLI](/guides/cli.md) — Ad-hoc fact-checking from the command line.
* [Web Service](/guides/web-service.md) — FastAPI endpoints, request/response schemas, and integration guide.
* [Configuration](/guides/configuration.md) — Tuning the checker: token budgets, retrieval, consistency runs.

## Architecture

* [System Architecture](/architecture/architecture.md) — Module layout and component responsibilities.
* [Data Flow](/architecture/data-flow.md) — Step-by-step pipeline from answer input to final report.
* [Output Format](/architecture/output-format.md) — CheckReport structure, verdicts, and dimensions.
* [Answer Quality Score](/architecture/answer-quality-score.md) — 0-10 numeric grade for overall answer quality.

## Reference

* [Fact-Checking Approaches](/overview/approaches.md) — Survey of possible RAG fact-checking strategies.
* [LLM Integration](/guides/llm-integration.md) — Plugging in HTTP API or custom models.
* [Testing](/guides/testing.md) — Test suite, MockLLM, and mock datasets.
* [Architecture Tension: Multi-Turn Conversations](/architecture-tension.md) — Design considerations for multi-turn chat scenarios.
* [Debug Artifacts](/guides/debug-artifacts.md) — Local-only HAR extraction format reference.
* [Debug Command: debug-check](/debug-prompts/debug-check.md) — CLI debug pipeline reference.
