# RAG Facts Check — Documentation Index

## Overview

* [Project Overview](/overview/overview.md) - What RAG Facts Check is and how it works at a high level.
* [Fact-Checking Approaches](/overview/approaches.md) - Survey of possible RAG fact-checking strategies and where this project fits.

## Architecture

* [System Architecture](/architecture/architecture.md) - Module layout and component responsibilities.
* [Data Flow](/architecture/data-flow.md) - Step-by-step pipeline from answer input to final report.
* [Output Format](/architecture/output-format.md) - CheckReport structure, verdicts, and dimensions.

## Guides

* [LLM Integration](/guides/llm-integration.md) - Plugging in Hugging Face, HTTP API, or custom models.
* [Configuration](/guides/configuration.md) - Checker options and tuning parameters.
* [Web Service](/guides/web-service.md) - FastAPI endpoints for async fact-checking.
* [Testing](/guides/testing.md) - Test suite, MockLLM, and mock datasets.

## Reference

* [Architecture Tension: Multi-Turn Conversations](/architecture-tension.md) - Design considerations for multi-turn chat scenarios.
* [Debug Command: debug-check](/debug-prompts/debug-check.md) - CLI debug pipeline reference.
