# RAG Facts Check — Handoff Document

**Date:** 2026-07-27
**Status:** Active development, integrated with volto-eea-chatbot frontend
**Branch:** `main` on github.com:tiberiuichim/rag-facts-check

---

## What This Is

A modular Python service that verifies RAG-generated answers against source
documents. It extracts factual claims from an answer, verifies each claim
against provided documents, and returns a structured report with per-claim
verdicts, confidence scores, evidence quotes, and character-level spans for
UI highlighting.

Used by the **climate-adapt chatbot** (volto-eea-chatbot frontend) via a
halloumi-compatible HTTP endpoint.

---

## Architecture

```
prompts/                    # Plain-text prompt templates (edit directly, no rebuild)
rag_facts_check/
├── __init__.py             # Package exports
├── models.py               # Claim, VerificationResult, CheckReport, Span
├── llm.py                  # LLM interface + adapters (HF, API, AsyncAPI)
├── prompts.py              # Loads prompts/ files, formats them with variables
├── retriever.py            # Chunk-based lexical evidence retrieval
├── spans.py                # Span matching (claim→answer, evidence→document)
├── checker.py              # Core pipeline: Extract → Verify → Aggregate
├── agents.py               # Pydantic schemas for structured LLM I/O (atomic-agents)
└── server.py               # FastAPI service (POST /check, POST /halloumi/generate)
tests/                      # pytest suite (132 tests, all passing)
mock_datasets/              # Synthetic test datasets (JSON)
docs/                       # OKF-format documentation
```

### Pipeline

```
Answer → ClaimExtractor → [Claim(text, original_text, span)]
                              │
                         EvidenceRetriever (optional, chunk-based)
                              │
                         ClaimVerifier → [VerificationResult]
                              │
                         Aggregator → CheckReport
                              │
                         _to_halloumi_format → {claims, segments}
```

### Key Design Decisions

- **Greenfield project** — no backward compatibility required (see AGENTS.md).
- **Async checker** — `check()`, `extract()`, `verify()` are `async def` because
  the server uses `AsyncAPILLM` (httpx). Tests use pytest-asyncio.
- **Prompts as data files** — all LLM prompts live in `prompts/*.txt`, loaded at
  runtime. Edit directly, restart server.
- **Structured JSON output** — prompts ask the LLM to return JSON. Parser handles
  both JSON and legacy text format (VERDICT:/CONFIDENCE:).
- **Multi-turn claim extraction** — claims include `original_text` (verbatim
  fragment from answer) for reliable span matching. If original_text can't be
  found, sends problematic claims back to LLM for refinement (up to 3 rounds).

---

## API

### POST /halloumi/generate

Frontend endpoint. Drop-in replacement for halloumi middleware.

**Request:**
```json
{
  "answer": "Paris is the capital of France.",
  "sources": ["Paris is the capital...", "The Eiffel Tower..."],
  "max_context_segments": 100
}
```

**Response:**
```json
{
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
- `claimString` — displayed in ClaimModal popup quote
- `startOffset`/`endOffset` — character offsets in the answer for highlighting
- `segmentIds` — references into `segments` for evidence quotes
- `score` — 0-1 float (confidence/100)
- `rationale` — displayed in ClaimModal rationale section
- `segments[id].id` — numeric id for citation chip display (#0, #1, ...)

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
- `max_claims` — limit claims for latency control
- `num_consistency_runs` — self-consistency (1 = single pass, 3 = majority vote)
- `evidence_first` — multi-step evidence-first prompting
- `use_evidence_retrieval` — chunk-based lexical retrieval per claim
- `retriever` — custom EvidenceRetriever instance

---

## Known Issues & Open Questions

### 1. LLM verdict/rationale inconsistency

**Problem:** The LLM sometimes returns `verdict: SUPPORTED, confidence: 100`
but the rationale clearly says the claim cannot be verified. Example:

```
claim: "France generated roughly 340 Mt of waste in 2022"
verdict: SUPPORTED
confidence: 100
rationale: "The documents contain percentage trends but not absolute mass.
            Cannot be converted to specific tonnage without baseline."
```

**Current state:** A `_normalise_result()` function in checker.py attempts to
detect this by scanning the rationale for uncertainty signals and flipping the
verdict. This is a band-aid, not a proper fix.

**Options discussed:**
- Better prompt: force the LLM to decide verdict BEFORE writing rationale
- Two-step verification: extract evidence first, then render verdict
- Validate + retry: detect contradiction, send refinement prompt
- Accept it: show rationale prominently so human sees the mismatch

**Decision needed:** Pick an approach and implement properly.

### 2. Span matching for paraphrased claims

**Problem:** The LLM rephrases claims during extraction. Even with
`original_text` and multi-turn refinement, some claims still don't match
the original answer text exactly.

**Current state:** Falls back to full answer range (startOffset=0,
endOffset=len(answer)) when span matching fails. This works but means
the UI highlights the entire answer instead of the specific claim.

**Options:**
- Use fuzzy matching (fuzzywuzzy, rapidfuzz) for span detection
- Ask the LLM to return character offsets directly
- Accept it: full-answer highlighting is imperfect but functional

### 3. Evidence retrieval is lexical only

**Problem:** The EvidenceRetriever uses keyword overlap scoring. For
semantically similar but lexically different text, retrieval quality
suffers.

**Options:**
- Add embedding-based retrieval (sentence-transformers)
- Accept it: lexical retrieval works well enough for exact quotes

### 4. No structured output enforcement

**Problem:** Prompts ask for JSON but the LLM may return malformed JSON
or the legacy text format. The parser handles both but this is fragile.

**Options:**
- Use instructor's response_model to enforce Pydantic schemas at the LLM level
- Add JSON repair (json-repair library) for malformed responses
- Accept it: dual-format parser works for now

---

## Frontend Integration

The frontend is at:
`/home/tibi/work/eea.docker.plone-climateadapt/cca/frontend/src/addons/volto-eea-chatbot/`

Key components:
- `ClaimModal.jsx` — popup showing claim quote, score, rationale
- `ClaimSegments.jsx` — evidence segments with citation chips
- `Citation.jsx` — source citation rendering
- `index.jsx` (markdown/) — spans processor that wraps claims in ClaimModal

The frontend calls `POST /_ha/generate` which proxies to our
`POST /halloumi/generate` endpoint.

---

## Development

```bash
# Setup
make setup-dev          # Creates .venv, installs all deps

# Run server
make serve              # uvicorn with auto-reload, --log-level info

# Test
.venv/bin/pytest tests/ -v

# Lint
.venv/bin/ruff check rag_facts_check/ tests/
.venv/bin/ruff format rag_facts_check/ tests/
```

### Dependencies

- `[test]` — pytest, pytest-cov, pytest-asyncio
- `[dev]` — ruff
- `[server]` — fastapi, uvicorn, httpx, python-dotenv, atomic-agents, instructor

### Prompt editing

Edit files in `prompts/*.txt` directly. Restart the server. No rebuild needed.

Template variables: `{system_prompt}`, `{text}`, `{claim}`, `{documents}`.

---

## Recent Changes (git log)

| Commit | Description |
|--------|-------------|
| c75ce3c | Include id field in segments for citation chip display |
| 6abc232 | Include claimString in halloumi response for frontend display |
| 3b89cb8 | Note greenfield project in AGENTS.md |
| 0881977 | Structured claim extraction with multi-turn refinement |
| de50fc6 | Extract LLM prompts into prompts/ folder |
| 223f763 | Include claims without spans + verbose server logging |
| 1e0543c | Make checker pipeline async to support AsyncAPILLM |
| 1d2a25a | Refactor docs to OKF format |

---

## People

- **Tiberiu Ichim** — project owner, tiberiu.ichim@gmail.com
