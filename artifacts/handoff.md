# RAG Facts Check — Handoff Document

**Date:** 2026-07-26
**Status:** Active development, integrated with volto-eea-chatbot frontend
**Repo:** github.com:eea/rag-facts-check (transferring from tiberiuichim)
**Branch:** `main`

---

## What This Is

A standalone FastAPI service for verifying RAG-generated answers against source
documents. Deployed at EEA to power the climate-adapt chatbot quality checks.

Extracts factual claims from an answer, verifies each against provided documents,
returns a structured report with per-claim verdicts, categorical scores, evidence
quotes, and character-level spans for UI highlighting.

---

## Architecture

```
prompts/                    # Plain-text prompt templates (edit directly)
rag_facts_check/
├── __init__.py             # Package exports
├── models.py               # Claim, VerificationResult, CheckReport, Span
├── llm.py                  # LLM interface + adapters (AsyncAPILLM is primary)
├── prompts.py              # Loads prompts/ files, format_documents() with title support
├── retriever.py            # Chunk-based lexical evidence retrieval (DocumentChunk has title)
├── spans.py                # Span matching (claim→answer, evidence→document)
├── checker.py              # Core pipeline: Extract → Verify → Aggregate
├── agents.py               # Pydantic schemas for structured LLM I/O (atomic-agents)
└── server.py               # FastAPI service (POST /check, POST /halloumi/generate)
tests/                      # pytest suite (162 tests, all passing)
mock_datasets/              # Synthetic test datasets (JSON)
docs/                       # OKF-format documentation
```

### Pipeline

```
Answer → ClaimExtractor (2048 tokens) → [Claim(text, original_text, span)]
                              │                         (deduped by span)
                         EvidenceRetriever (optional, chunk-based)
                              │
                         ClaimVerifier → [VerificationResult]
                              │              (evidence spans mapped to joined sources)
                         Aggregator → CheckReport
                              │
                         _to_halloumi_format → {answer_score, claims, segments}
```

### Key Design Decisions

- **Greenfield project** — no backward compatibility required (see AGENTS.md).
- **Server-first** — primary consumer is the EEA chatbot frontend via HTTP.
- **Async checker** — `check()`, `extract()`, `verify()` are `async def`.
- **Prompts as data files** — all LLM prompts in `prompts/*.txt`, loaded at runtime.
- **Structured sources** — `HalloumiSource` schema: `{text, title, source_type, link}`.
  Titles appear as LLM prompt headers without polluting raw text (span-safe).
- **Categorical scores** — High (1.0 = supported), Low (0.4 = not enough info),
  Failed (0.0 = contradicted). Frontend renders these labels, not percentages.
- **Multi-turn extraction** — claims include `original_text` (verbatim fragment).
  If unmatchable, sent back to LLM for refinement (up to 3 rounds).
- **Span safety** — zero-length or unmatched evidence spans are skipped (no bogus highlights).

---

## API

### POST /halloumi/generate (primary)

Frontend endpoint. Drop-in replacement for halloumi middleware.

**Request:**
```json
{
  "answer": "Paris is the capital of France.",
  "sources": [
    { "text": "Paris is the capital...", "title": "Paris overview" },
    { "text": "The Eiffel Tower...", "title": "Eiffel Tower" }
  ],
  "max_context_segments": 0
}
```

Sources accept plain strings (backward compat) or structured `HalloumiSource`
objects with `text`, `title`, `source_type`, `link` fields.

**Response:**
```json
{
  "answer_score": 7.5,
  "claims": [
    {
      "claimString": "Paris is the capital of France.",
      "startOffset": 0,
      "endOffset": 31,
      "segmentIds": ["0"],
      "score": 1.0,
      "rationale": "Document states this explicitly."
    }
  ],
  "segments": {
    "0": { "id": 0, "startOffset": 0, "endOffset": 31 }
  }
}
```

Field contract with frontend (volto-eea-chatbot):
- `answer_score` — 0-10 overall grade (Excellent/Good/Acceptable/Poor/Failing)
- `claimString` — displayed in ClaimModal popup quote
- `startOffset`/`endOffset` — character offsets in the answer for highlighting
- `segmentIds` — references into `segments` for evidence quotes
- `score` — 1.0 (High), 0.4 (Low), 0.0 (Failed)
- `rationale` — displayed in ClaimModal rationale section
- `segments[id].id` — numeric id for citation chip display

### POST /check

Full fact-checking endpoint. Returns complete CheckReport with dimensions,
hallucination flags, etc.

### GET /health

Returns `{"status": "ok", "version": "0.2.0"}`.

---

## Configuration

Server reads from `.env`:

```env
LLM_API_BASE=http://localhost:4002/v1
LLM_API_KEY=not-needed
LLM_MODEL=gemma
LLM_TEMPERATURE=0.1
```

Checker parameters (in `RAGFactsChecker.__init__`):
- `max_claims` — limit claims for latency control (default: None)
- `max_new_tokens` — LLM tokens for verification (default: 512)
- `max_extraction_tokens` — LLM tokens for claim extraction (default: 2048)
- `num_consistency_runs` — self-consistency (default: 1)
- `evidence_first` — multi-step evidence-first prompting (default: True)
- `use_evidence_retrieval` — chunk-based lexical retrieval (default: True)

---

## Frontend Integration

Frontend repo:
`/home/tibi/work/eea.docker.plone-climateadapt/cca/frontend/src/addons/volto-eea-chatbot/`

Key integration points:
- `AIMessage.tsx` — builds `halloumiSource` objects from `final_documents`,
  sends to `/_ha/generate`. Falls back to `doc.content` when tool packets
  lack content (critical fix: prevents empty sources).
- `useQualityMarkers.js` — calls `/_ha/generate`, passes structured sources
  (prefers `halloumiSource` over legacy `halloumiContext` strings).
- `ClaimModal.jsx` — displays claim quote, score label (High/Low/Failed), rationale.
- `ClaimSegments.jsx` — evidence segments with citation chips, builds
  `joinedSources` string from `halloumiContext` for span highlighting.

Middleware proxy: `halloumi/middleware.js` forwards `/_ha/generate` to
`RAG_FACT_CHECKER_URL/halloumi/generate`.

---

## Known Issues & Open Questions

### 1. LLM verdict/rationale inconsistency

**Problem:** The LLM sometimes returns `SUPPORTED` with 100% confidence but
the rationale says the claim cannot be verified.

**Current state:** Band-aid `_normalise_result()` in checker.py scans rationale
for uncertainty signals. Not a proper fix.

**Options:** verdict-before-rationale prompt, two-step verification,
validate-and-retry, or accept and show rationale prominently.

### 2. Evidence retrieval is lexical only

**Problem:** Keyword overlap scoring. Semantically similar but lexically
different text suffers.

**Options:** embedding-based retrieval (sentence-transformers) or accept.

### 3. Duplicate sources from frontend

**Problem:** HAR analysis shows 1-3 unique documents out of 50-205 sources,
with the rest being exact duplicates. E.g. europe-6.har: 24 copies of the
same 12 KB document, 1 copy of a 7 KB document, 25 empty strings.

**Root cause:** Frontend builds sources array from streaming packets without
deduplication. Same document appears in multiple packets (MESSAGE_START,
SEARCH_TOOL_DELTA, etc.) and gets added each time.

**Impact:** Wastes context window, confuses `_find_source_index` (picks first
match which may be wrong), inflates source counts in UI.

**Fix needed:** Deduplicate sources in frontend before sending to fact-checker.
Must preserve order and keep the first occurrence. Critical: texts must remain
byte-identical between frontend (for span highlighting) and backend (for
evidence span computation).

---

## Running

```bash
# Development (auto-bootstraps venv)
make serve

# Production
docker build -t rag-fact-check .
docker run -p 8000:8000 rag-fact-check

# Tests
make test-coverage     # pytest with coverage report (69% overall)

# Lint
.venv/bin/ruff check rag_facts_check/ tests/
.venv/bin/ruff format rag_facts_check/ tests/
```

### Dependencies

- `[test]` — pytest, pytest-cov, pytest-asyncio
- `[dev]` — ruff
- `[server]` — fastapi, uvicorn, httpx, python-dotenv, atomic-agents, instructor, openai

### Prompt editing

Edit `prompts/*.txt` directly. Restart server. No rebuild.

Template variables: `{system_prompt}`, `{text}`, `{claim}`, `{documents}`.

---

## Recent Changes

| Date | Change |
|------|--------|
| 2026-07-26 | Docs reorganised: server-first narrative, OKF changelog |
| 2026-07-26 | MIT License (European Environment Agency) |
| 2026-07-26 | README: server-first, Dockerfile for production |
| 2026-07-26 | `make serve` auto-bootstraps venv |
| 2026-07-25 | Structured sources with document titles in prompts |
| 2026-07-25 | Span offset fix: map to joined sources string |
| 2026-07-25 | Claim extraction: 2048 tokens, dedup by span, complete statements |
| 2026-07-25 | Zero-length evidence spans skipped |
| 2026-07-25 | Categorical scores: High/Low/Failed |
| 2026-07-25 | Frontend fix: use doc.content from final_documents |

---

## People

- **Tiberiu Ichim** — project owner, tiberiu.ichim@gmail.com
