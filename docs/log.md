# Changelog

## Unreleased

### Server & Deployment
- `make serve` auto-bootstraps venv via `setup-dev` prerequisite
- README rewritten: server-first narrative, Dockerfile for production
- MIT License (European Environment Agency, 2026)
- Docs reorganised: Getting Started section leads with web service and configuration

### Claim Extraction
- Token budget increased: 512 → 2048 (new `max_extraction_tokens` parameter)
- Prompt tightened: requires complete statements (subject + predicate), not bare noun phrases
- Claims with identical answer spans are deduplicated
- Prompt covers every table row explicitly

### Verification & Spans
- Structured sources: `HalloumiSource` schema (`text`, `title`, `source_type`, `link`) — titles appear as LLM prompt headers without polluting raw text
- `_to_halloumi_format`: maps per-document evidence spans into the frontend's joined sources string
- Zero-length evidence spans are skipped (no bogus highlights)
- Chunk-offset fallback removed from `_find_evidence_span`
- Categorical claim scores: High (1.0), Low (0.4), Failed (0.0)

## 0.2.0 (2025-07)

### Architecture
- Structured claim extraction with multi-turn refinement (up to 3 rounds for unmatchable `original_text`)
- Atomic agents (atomic-agents + instructor) for structured LLM I/O with automatic retries
- Answer quality score (0-10) with verdict-based scoring
- Span-level grounding for claims and evidence (character offsets for clickable highlighting)
- Halloumi-compatible endpoint (`POST /halloumi/generate`) for frontend drop-in replacement
- Async pipeline: `RAGFactsChecker.check()` is `async` to support `AsyncAPILLM`
- Production Dockerfile

### Data Model
- `Claim.original_text` — exact verbatim fragment for span matching
- `VerificationResult.evidence_span` — character offsets of evidence in source documents
- `VerificationResult.document_index` — LLM-identified source document index
- `CheckReport.answer_score` — 0-10 numeric grade
- `CheckReport.dimensions` — groundedness, contradiction_rate, hallucination_rate, completeness

### Infrastructure
- Ruff per-file ignores for prompt strings and mock responses
- Branch naming convention (hyphens, no slashes)
- Greenfield project: no backward compatibility required
- OKF-formatted documentation structure

## 0.1.0 (2025-01)

Initial release.

- Claim extraction from RAG answers
- Per-claim verification with evidence-first prompting
- Self-consistency (multiple runs, majority vote)
- Evidence retrieval (chunk-based lexical matching)
- Multi-dimensional scoring (groundedness, contradiction rate, hallucination rate)
- FastAPI web service (`POST /check`)
- MockLLM for testing
- Synthetic test datasets
