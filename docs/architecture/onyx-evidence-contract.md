---
type: Architecture
title: Onyx Evidence Contract
description: Where the answer's evidence actually lives in an Onyx v3 chat stream, and how the fact-checker obtains real document text instead of search blurbs.
tags: [onyx, evidence, sources, chunk-info, hallucination, v3-stream]
timestamp: '2026-10-08T00:00:00Z'
---

# Onyx Evidence Contract

The fact-checker verifies an answer against the `sources` the frontend sends. In an
Onyx **v3** stream those sources are **search blurbs, not document text** — which
makes the checker manufacture false hallucination verdicts. This document records
what the stream does and does not carry, what the answer generator actually read, and
how to obtain the real evidence using existing Onyx endpoints.

Verified against Onyx `v4.3.9-eea.0.0.118` (branch `merge_v4.3.9`) and four captured
streams (Aug–Oct 2026).

## 1. The failure mode

A 7,009-character, 10-row answer was fact-checked against **1,628 characters** of
blurbs. Five phrases the checker called unsupported — *Copernicus*, *European
Investment Bank*, *Climate Awareness Bond*, *Forest Information System*, *"28
headline indicators"* — are all present in the real chunk text Onyx read. Only one
row (an uncited "European Climate Law & Fit for 55 dashboard") was a genuine
hallucination candidate.

**Thin sources do not merely lower scores, they invert verdicts.** A claim absent
from a 500-character snippet is not evidence of hallucination.

## 2. What the stream carries

`SearchDoc` (`onyx/context/search/models.py:244`) is the payload type for both
`search_tool_documents_delta` and `message_start.final_documents`. It has **no
`content` field**:

```
document_id, chunk_ind, semantic_identifier, link, blurb, source_type, boost,
hidden, metadata, score, is_relevant, relevance_explanation, match_highlights,
updated_at, primary_owners, secondary_owners, is_internet, file_id
```

`metadata` is PDF provenance (Author/CreationDate/Producer/…), `match_highlights`
was `[]` and `relevance_explanation` `null` in every captured chunk. Content exists
only on `SavedSearchDocWithContent` and `InferenceSection.combined_content` — neither
is streamed, and session reload (`session_loading.py:511`) rebuilds deltas from
`SavedSearchDoc`, so replaying an old chat never yields text either.

Packet inventory for the reference capture:

```
message_delta 1314, reasoning_delta 950, reasoning_done 3, citation_info 3,
search_tool_start 2, search_tool_queries_delta 2, search_tool_documents_delta 2,
section_end 2, reasoning_start 2, message_start 1, stop 1
```

No packet outside `message_delta` / `reasoning_delta` contains a string longer than
300 characters. The 1,314 `message_delta` packets sum to exactly the answer length.

Two captures of the same conversation from August (pre-v3) *did* carry `content` on
18/18 documents — the field disappeared with the v3 stream, it was never removed from
the frontend, which still does `content: doc.content || doc.blurb`
(`messageProcessor.ts:extractToolCall`). That silent substitution is what made the
gap invisible.

## 3. What the answer generator actually read

`search_tool.py:run()` returns two different payloads from the same sections:

```python
docs_str, citation_mapping = convert_inference_sections_to_llm_string(
    top_sections=merged_sections,          # LLM-selected + context-expanded
    limit=override_kwargs.max_llm_chunks,  # MAX_CHUNKS_FED_TO_CHAT = 25
)
return ToolResponse(
    rich_response=SearchDocsResponse(search_docs=search_docs, ...),  # → browser
    llm_facing_response=docs_str,                                    # → answer LLM
)
```

The pipeline, with its hard caps:

| stage | cap | source |
|---|---|---|
| retrieval per query | `NUM_RETURNED_HITS = 50` sections | `chat_configs.py:5` |
| trim before selection | `25 × 512 = 12,800` tokens, ≤3 chunks/section | `search_tool.py:1037` |
| LLM section selection | ≤10 sections | `document_filter.py:191` |
| context expansion | one shot: drop / centre / ±2 / ±5 | `search_utils.py:385`, `constants.py:23` |
| merge | adjacent/overlapping sections of the same doc merged | `search_utils.py:208` |
| final payload | ≤25 sections (`MAX_CHUNKS_FED_TO_CHAT`) | `search_tool.py:1153` |
| agent loop | `MAX_LLM_CYCLES = 6` search calls per turn | `chat_configs.py:14` |

Expansion is **one classification pass per section, never escalated**: the classifier
answers 0/1/2/3 (`MAIN_SECTION_ONLY` is the default — no widening), and nothing
re-widens if the answer turns out thin. The only escalation in Onyx is the answer LLM
choosing to search again. `chunk_ind` on a streamed `SearchDoc` is the **centre** of
its section (`models.py:294`), so a multi-chunk section exposes only one chunk index.

In the reference capture: 100 retrieved chunks / 37 documents, of which the selection
LLM forwarded **4 unique documents** across 2 search calls. Those 4 — not the 37 — are
the answer's evidence base.

## 4. `final_documents` is not the evidence set

Every `final_documents` entry carries `document_id` + `chunk_ind`, so all of them are
fetchable (100 entries → 60 unique pairs). Fetching them all is still wrong:

| selected doc | centre | its `final_documents` chunk_inds | overlap with centre±5 |
|---|---|---|---|
| 8th EAP monitoring report | 63 | 2, 4, 10, 13, 16, 49, 55, 63 | 1/11 |
| Policy corner | 0 | 0 | 1/6 |
| 8th EAP report (pdf) | 60 | 4, 8, 10, 23, 56, 60, 66, 72 | 2/11 |
| Green bonds indicator | 4 | 1, 4 | 2/10 |

Retrieved chunk indices are scattered query hits, not neighbourhoods: only **6 of the
38** chunks in the selected docs' centre±5 windows appear in `final_documents`. The
set also includes 33 documents the selection LLM dropped and never fed to the answer.
Cost: 60 requests and ~144k characters, over the checker's 100k
`max_total_chars` budget.

**Fetch the selected documents (`search_tool_documents_delta`) around their centre
chunk.** That is the set Onyx chose to feed.

## 5. The evidence endpoint: `GET /document/chunk-info`

`onyx/server/documents/document.py:73`

```python
@router.get("/chunk-info", dependencies=[Depends(require_vector_db)])
def get_chunk_info(
    document_id: str = Query(...),
    chunk_id: int = Query(...),
    user: User = Depends(require_permission(Permission.BASIC_ACCESS)),
    ...
) -> ChunkInfo:   # {"content": str, "num_tokens": int}
```

Why this is the right lever:

- The two query params are exactly the two fields the stream already ships for every
  document: `document_id` and `chunk_ind`.
- **Auth is free** — `BASIC_ACCESS` is in `NON_TOGGLEABLE_PERMISSIONS`, so the cookie
  or API key the Volto `_da` proxy already holds suffices. No curator/admin role.
- ACL is enforced per calling user (`build_access_filters_for_user`).
- Same source of truth as the answer: the chunk comes straight out of the vector
  index the retrieval pipeline read from.
- No Onyx change, no DB credentials, no extra LLM call.

Design constraints:

1. **One chunk per call** — the endpoint pins `min_chunk_ind == max_chunk_ind`, so a
   chunk is ~512 tokens (`DOC_EMBEDDING_CONTEXT_SIZE`). Match Onyx's expansion by
   calling `chunk_ind-window … chunk_ind+window`.
2. **404 is normal** at `chunk_ind` beyond the document's chunk count — treat it as
   the document edge, not an error. It also happens if a reindex/prune removed the
   chunk between answering and checking.
3. **Gateway rate limits.** A gateway in front of `aihub` (not Onyx — its 429 body is
   not in the Onyx codebase) throttles at roughly 30 req/s. 8 workers × 38 requests
   produced 4–10 failures; **one retry after 1s clears every 429**. Keep workers ≤4.
4. **The Onyx web UI does not use this endpoint**, so it is internal — pin it with an
   integration test so an Onyx upgrade breaks loudly instead of silently degrading
   fact-check scores.

### Measured (2026-10-07, `https://aihub.eea.europa.eu`)

Reproduce with:

```bash
python scripts/probe_chunk_info.py --stream capture.jsonl --window 2
```

| mode | requests | wall time | evidence chars | vs blurbs |
|---|---|---|---|---|
| centre chunk only (`--window 0`) | 4/4 | 0.6 s | 10,671 (1,997 tok) | 4.8× |
| ±2 chunks (`--window 2`) | 17/18 (1× 404) | 2.2 s | 40,073 (8,216 tok) | 18.1× |
| ±5 chunks (`--window 5`) | 34/38 (4× 404) | 2.9 s | 76,122 (16,420 tok) | 34.3× |

Per-document character counts (the frontend sends one source per document, its
chunks joined):

| document | blurb | ±2 | ±5 |
|---|---|---|---|
| 8th EAP monitoring report (centre 63) | 561 | 12,074 | 25,116 |
| Policy corner (centre 0) | 566 | 7,904 | 13,974 |
| 8th EAP report pdf (centre 60) | 592 | 11,473 | 24,397 |
| Green bonds indicator (centre 4) | 501 | 8,622 | 12,635 |

Window 0 is **not sufficient** — at ±0 four of the five previously-"missing" phrases
are still absent, because the claims sit in the neighbourhood of the matched chunk,
not in it. ±2 recovers all five and matches Onyx's `INCLUDE_ADJACENT_SECTIONS`
expansion exactly; ±5 doubles the corpus for the same coverage on this example.

### Character budgets

Batch verification (the default, `batch_size=20`, which is what `/halloumi/generate`
uses) skips per-claim chunk narrowing and sends **every source whole**, so the source
budget is what decides how much of this evidence the judge actually sees.

`format_documents` caps the corpus at `max_docs_chars = 100000` and applies
`max_chars_per_doc = 10000` as a **fairness guard only** — it kicks in when the corpus
overflows the total, so one huge source cannot starve the others. A corpus that fits is
passed whole. Both are env-tunable on the server (`CHECKER_MAX_DOCS_CHARS`,
`CHECKER_MAX_CHARS_PER_DOC`); see [Configuration](../guides/configuration.md).

| window | corpus | vs `max_docs_chars` | truncated |
|---|---|---|---|
| ±2 | 40,073 chars | 40% | nothing |
| ±5 | 76,122 chars | 76% | nothing |

So the whole ±5 expansion already fits, and the binding constraint is the total budget
(100k characters ≈ 20k tokens), not the per-document cap. Widening beyond ±5, or
raising `MAX_CHUNKS_FED_TO_CHAT` in Onyx, is what would force `CHECKER_MAX_DOCS_CHARS`
up — with a directly proportional cost in judge latency and tokens.

## 6. Frontend contract

Implemented in `volto-eea-chatbot`:

- `services/chunkEvidence.ts` fetches `chunk_ind ± window` (default **2**) per
  selected document, worker pool of 4, one 429 retry, 404 = document edge.
- `hooks/useChunkEvidence.ts` gates the fact-check on `evidenceSettled`, so the
  checker never fires on blurbs in the same render that started the fetch.
- Each source carries `kind`: `"chunk"` (real text) or `"snippet"` (blurb fallback).
- `middleware.js` allowlists `GET /document/chunk-info` on the `_da` proxy.

## 7. Fallback ladder the checker follows

`/halloumi/generate` reads `kind` from every structured source and returns a
`context_quality` block describing what it was actually given:

| `level` | when | how to read the score |
|---|---|---|
| `full` | every source is `chunk` text | verify normally, score 0–10 |
| `partial` | at least one source is a `snippet` | report as **partial context** — a claim missing from a snippet is not evidence of hallucination |
| `unknown` | client declared no `kind` | older client; do not infer thin evidence |
| `none` | no sources survived | cannot verify, not a low score |

In `partial` mode each `not_enough_info` claim additionally carries
`"context_limited": true`, so the UI can distinguish "the answer may be wrong" from
"we could not see enough of the answer's sources". The numeric `answer_score` is left
untouched so runs stay comparable — the label, not the number, carries the caveat.

## 8. Byte-exact alternatives

Chunk re-retrieval approximates the corpus. The byte-exact corpus is persisted:
`llm_loop.py:1120` stores `tool_response.llm_facing_response` — literally the JSON the
answer LLM received, `content` per section — in `tool_call.tool_call_response`
(Postgres `Text`, `db/models.py:2977`), joined on
`parent_chat_message_id = assistant_message.id`. The join key is already in the
stream's first packet as `reserved_assistant_message_id`.

```sql
SELECT t.name,
       r ->> 'document'  AS citation_number,
       r ->> 'title'     AS title,
       length(r ->> 'content') AS content_chars,
       r ->> 'content'   AS content
FROM tool_call tc
JOIN tool t ON t.id = tc.tool_id
CROSS JOIN LATERAL jsonb_array_elements((tc.tool_call_response)::jsonb -> 'results') AS r
WHERE tc.parent_chat_message_id = :assistant_message_id
  AND t.name IN ('internal_search', 'web_search');
```

Cast to jsonb only after filtering to search tools — the column is plain text and is
not JSON for other tools. Scope to `parent_tool_call_id IS NULL` for top-level calls.
Older turns' tool responses are *not* in the final prompt: on history replay they are
replaced by `TOOL_CALL_RESPONSE_CROSS_MESSAGE` ("the results are no longer
accessible", `chat_prompts.py:95`), so the current message's rows are the faithful
corpus.

Other options: `POST /api/search` (`server/features/search/api.py:187`) returns the
same `llm_facing_response` but re-runs retrieval, selection and expansion and costs an
extra LLM call; a custom `/chat/evidence` endpoint in the EEA fork would be byte-exact
over HTTP but requires an Onyx change.

## 9. Citation numbering caveat

Citation numbers are assigned per search call (`starting_citation_num` continues
across calls), and `citation_processor.py:508` deliberately skips `CitationInfo` for a
document already cited:

```python
if doc_id in self.recent_cited_documents:
    continue
```

So one document can be both citation 1 and citation 5, with number 5 carrying no
`CitationInfo`. The citation map built from `citation_info` packets is **structurally
incomplete** (numbers 3, 5, 6 unmapped in the reference capture), so citation-mode
source selection can silently drop cited documents. A missing citation number is not
by itself a hallucination signal.

## Related

* [Data Flow](data-flow.md) — pipeline from answer input to report.
* [Answer Quality Score](answer-quality-score.md) — how the 0–10 grade is computed.
* [Configuration](../guides/configuration.md) — `max_total_chars` and token budgets.
