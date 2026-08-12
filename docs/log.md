# Documentation Change Log

## 2025-08-04
* **Creation**: Added [Debug Artifacts](/guides/debug-artifacts.md) — format reference for local HAR extraction artifacts.

## 2026-07-26
* **Update**: Updated [Web Service](/guides/web-service.md) — added HalloumiSource schema (structured sources with title, source_type, link) and categorical claim scores.
* **Update**: Updated [Configuration](/guides/configuration.md) — added max_extraction_tokens parameter, fixed evidence_first and use_evidence_retrieval defaults.
* **Update**: Updated [Data Flow](/architecture/data-flow.md) — documented claim dedup, 2048 extraction token budget, title headers in verification prompts, zero-length span skipping.
* **Update**: Updated [docs/index.md](index.md) — reorganised with Getting Started section (overview, web service, config) before Architecture and Reference.
* **Update**: Updated [Project Overview](/overview/overview.md) — replaced library quickstart with server run instructions, mentioned halloumi endpoint and categorical scores.

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
