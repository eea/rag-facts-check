---
title: "RAG Facts Check"
subtitle: "Automated Claim-Level Verification for Enterprise RAG"
author: "European Environment Agency (EEA) / Plone Chatbot"
date: "September 2026"
---

## The Enterprise RAG "Trust Gap"

::::: columns
::: {.column width="50%"}
### The Core Challenges
- **Subtle, High-Stakes Hallucinations**
  - Not obvious gibberish, but shifted policy dates (2035 vs 2050) or temperature baselines (5.7°C vs 2.0–4.0°C).
- **The Token Generation Limit**
  - Rich answers with 8+ row comparison tables exceed LLM output budgets, truncating one-pass extraction mid-answer.
- **Unstructured Source Disconnect**
  - Scraped PDFs and HTML have hard line breaks, OCR shifts, and hyphenation that break naive string matching.
:::
::: {.column width="50%"}
### Operational Impact
- **Loss of Stakeholder Trust**
  - A single incorrect regulatory date invalidates the entire response.
- **Latency Bottlenecks**
  - Checking 20 claims serially against 30KB of documents takes 45–60s+, breaking interactive chat UX.
- **The Need**
  - Granular, sub-sentence verification with exact character attribution and real-time speed.
:::
:::::

::: notes
Standard RAG is great at finding relevant documents, but the generation step remains a black box. In policy and regulatory contexts, hallucinations are rarely obvious nonsense. They are subtle: an LLM might say a net-zero milestone is legally binding by 2035 instead of 2050, or quote a temperature rise of 5.7°C instead of 2.0 to 4.0°C. That single discrepancy undermines the entire platform's credibility.

When teams try to build verification for this, they hit three practical walls:
First, long answers—especially rich markdown tables comparing climate impacts across sectors—routinely hit model output token ceilings, cutting verification off halfway through the response.
Second, source documents scraped from official portals have erratic formatting, linebreaks, and OCR noise, causing exact string matching to fail completely.
And third, calling an LLM sequentially for dozens of individual claims takes 45 to 60 seconds, which kills interactive chat UX.

We designed RAG Facts Check specifically to overcome every single one of these production hurdles.
:::

## System Architecture: End-to-End Pipeline

| Stage | Component | Core Function |
|---|---|---|
| **1. Ingestion** | `POST /halloumi/generate` | Receives answer text + cited source documents |
| **2. Chunking** | `split_answer_into_chunks` | Preserves markdown tables & replicates header rows |
| **3. Extraction** | `ClaimExtractor` | Extracts atomic claims with answer span coordinates |
| **4. Verification** | `ClaimVerifier` | Batches claims & verifies against docs via KV-cache reuse |
| **5. Alignment** | `find_evidence_span_in_doc` | 4-stage hierarchy maps evidence quotes to exact doc offsets |
| **6. Aggregation** | `RAGFactsChecker._aggregate`| Computes 0–10 quality score & formats Volto segments |

::: notes
Here is the architectural blueprint. RAG Facts Check runs as a lightweight, containerized FastAPI service. In our deployment with the European Environment Agency, Plone and the LLM generate the answer along with cited sources.

Immediately after generation, the Volto frontend issues an asynchronous request to our /halloumi/generate endpoint.

Inside the engine, the request flows through four modular stages managed by RAGFactsChecker:
First, the answer text is chunked without destroying table formatting.
Second, atomic claims are extracted with character offsets mapped to the master answer.
Third, the claims are batched and verified against the source documents in a single pass.
Fourth, verbatim evidence quotes are resolved into exact character spans in the original source texts.
Finally, a calibrated 0-to-10 score is computed, and the payload is returned to the frontend.

Let's look closer at how we solved the long-answer truncation problem in step one.
:::

## Innovation 1: Markdown-Aware Chunked Extraction

::::: columns
::: {.column width="55%"}
### Preserving Table Semantics
- **The Ceiling:** Answers >2,000 characters with large comparison tables exceed `max_new_tokens = 2048`.
- **Header Replication:** `split_answer_into_chunks()` splits large tables row-by-row while **copying the header row into every chunk**.
- **Context Retention:** The LLM always understands column definitions for every row, even in chunk 3 or 4.
- **Section Merging:** Small headings and lead paragraphs are merged to ensure sufficient context.
:::
::: {.column width="45%"}
### Coordinate Recalibration
- **Offset Tracking:** Each chunk tracks its offset relative to the master answer:
  $$\text{master\_offset} = \text{chunk\_start} + \text{local\_span}$$
- **Span Deduplication:** Overlapping claims extracted at chunk boundaries are deduplicated and re-indexed.
- **Guarantee:** 100% claim coverage across long answers and tables.
:::
:::::

::: notes
One of our earliest discoveries was that chatbot answers frequently include markdown comparison tables. If you send an 8-row table to an LLM extractor in one prompt, the model runs out of output tokens before it reaches the summary at the bottom. The table is only half-checked, and concluding recommendations aren't checked at all.

If you naively slice a markdown table by character count, the second chunk loses the header row, meaning the LLM has no idea what column a cell belongs to.

Our solution, implemented in split_answer_into_chunks, parses markdown table structures. When a table must be split, it preserves row integrity and automatically duplicates the header row into subsequent chunks.

We then recalibrate the character offsets back into the master answer coordinates and deduplicate overlapping claims. This gives us 100% extraction coverage on long, structured policy responses.
:::

## Innovation 2: KV-Cache Prefix Reuse & Batch Verification

::::: columns
::: {.column width="50%"}
### The Performance Breakthrough
- **The Latency Trap:**
  - Sequential check: 20 claims × 2.5s = **50 seconds** (Unviable in chat).
  - Batched check: 20 claims in 1 prompt = **5–12 seconds** (>70% faster).
- **Static Prefix Ordering:**
  - Source documents placed at the **very start** of the verification prompt.
  - Local engines (`vLLM`, `llama.cpp`) compute document tokens once and reuse the pre-computed KV-cache prefix.
:::
::: {.column width="50%"}
### Targeted Context Filtering
- **Context Selection (`qualityCheckContext`):**
  - `'all'`: 15–50 search chunks (~150KB) $\rightarrow$ High prefill overhead.
  - `'citations'` *(Recommended)*: Only cited sources (2–4 docs, ~25KB) $\rightarrow$ Instant prefill.
- Matches the exact context used by the generation model.
- Eliminates noise from unconsulted documents.
:::
:::::

::: notes
Verification latency is make-or-break for a chatbot. If a user has to wait a minute for verification highlights, they will navigate away.

In ClaimVerifier, we solve this with two key techniques:

First, batching. Rather than firing one LLM call per claim, we evaluate batches of up to 20 claims simultaneously.
Second, prompt architecture designed for KV-cache prefix reuse. We place the static reference documents at the beginning of the prompt. Modern inference engines like vLLM and llama.cpp recognize that this prefix hasn't changed, so GPU memory reuses the pre-computed key-value cache rather than re-ingesting tens of thousands of document tokens.

In addition, our frontend passes only the documents explicitly cited in the answer rather than the full set of 50 search chunks. Together, these optimizations drop verification latency from over 45 seconds down to 5 to 12 seconds.
:::

## Innovation 3: 4-Stage Resilient Evidence Span Matching

::::: columns
::: {.column width="55%"}
### The Matching Hierarchy (`spans.py`)
1. **Quote & Ellipsis Stripping:**
   - Cleans surrounding quotes (`"`, `'`, `“`, `”`) and ellipses (`...`).
2. **Exact Match:**
   - Instant substring index lookup.
3. **Whitespace Regex (`\s+`):**
   - Matches quotes across linebreaks (`\n`), tabs, and multiple spaces.
4. **Punctuation Regex (`[\W_]+`):**
   - Tolerates hyphenation across lines and punctuation shifts.
5. **Fuzzy Sequence Alignment:**
   - Fallback via `difflib.SequenceMatcher` for minor paraphrasing.
:::
::: {.column width="45%"}
### Why It Matters
- **Messy Data Reality:**
  - Scraped PDFs and HTML briefings have formatting quirks.
- **LLM Tendencies:**
  - Models frequently add decorative quotes or truncate citations.
- **Outcome:**
  - Robust `{startOffset, endOffset}` coordinates for UI tooltips without brittle exact-match failures.
:::
:::::

::: notes
Once the model identifies evidence, it cites a verbatim snippet. But anyone who has worked with real document pipelines knows what happens next: string matching fails.

Official documents converted from PDF or HTML are full of hard newlines, soft hyphens, and weird spaces. Furthermore, LLMs love adding smart quotes or ellipses to their evidence citations.

To ensure citations never break, we developed a 4-stage matching hierarchy in find_evidence_span_in_doc.
It first strips framing quotes and ellipses. It tries an exact match. If that fails, it compiles a whitespace-flexible regular expression that treats line breaks and multiple spaces as single wildcards. If that fails, it allows punctuation variations. And as a final safety net, it uses fuzzy sequence alignment.

This guarantees that whenever the LLM cites true source evidence, we resolve the exact character coordinates in the original document so the user can inspect it.
:::

## Calibrated Quality Scoring & Frontend UX

::::: columns
::: {.column width="50%"}
### Deterministic 0–10 Formula
$$\text{Score} = \text{clamp}(\text{Base} \times P_{\text{cite}} \times P_{\text{contra}},\, 0,\, 10)$$

- **Groundedness Base (0–10):**
  - Supported = $1.0$ | Unverified = $0.4$ | Contradicted = $0.0$
- **Citation Penalty ($P_{\text{cite}}$):**
  - Up to 30% reduction if claims lack matched source spans.
- **Contradiction Penalty ($P_{\text{contra}}$):**
  - Severe penalty; 33%+ contradicted floors score to 0.
:::
::: {.column width="50%"}
### Interactive Volto UI Experience
- **Overall Score Badge:**
  - e.g. `8.5 / 10 Good` (instant calibration).
- **Inline Text Highlights:**
  - 🟢 **Green:** Verified claim with source evidence tooltip.
  - 🟡 **Yellow:** Unverified or low-confidence statement.
  - 🔴 **Red:** Statement contradicted by source documents.
- **Evidence Tooltips:**
  - Hovering reveals the verbatim excerpt from the official document.
:::
:::::

::: notes
Users do not want an opaque 'Pass' or 'Fail'. They need calibrated, actionable transparency.

In models.py and checker.py, we compute a deterministic 0-to-10 Answer Quality Score. Supported claims earn full credit, uncited claims receive a proportional penalty, and direct contradictions hit the score with a heavy penalty multiplier.

In the Volto user interface, this translates into immediate visual clarity:
An overall score badge—like '8.5 / 10 Good'—sits at the top.
Every sentence in the answer is highlighted. Green means fully verified, and hovering over it reveals the exact quote from the European legislation. Yellow alerts the user that a claim lacks direct source backing, and red immediately flags a contradiction.

This shifts the user dynamic from blind skepticism to confident, auditable reading.
:::

## Production Architecture & Engineering Rigor

::::: columns
::: {.column width="50%"}
### Enterprise Packaging
- **FastAPI Core:**
  - High concurrency, async I/O, automatic OpenAPI documentation.
- **Drop-In Compatibility:**
  - Native `/halloumi/generate` endpoint requires zero frontend code changes.
- **Containerized:**
  - Dockerized deployment ready for Kubernetes or cloud environments.
- **Model Agnostic:**
  - Compatible with LiteLLM, vLLM, llama.cpp, Ollama, Gemma, Llama, and Mistral.
:::
::: {.column width="50%"}
### Testing & Quality Assurance
- **190 Automated Tests (100% Pass):**
  - 87% coverage on `checker.py`
  - 88% coverage on `spans.py`
  - 95% coverage on `retriever.py`
  - 100% coverage on `models.py`
- **Structured LLM Output:**
  - Guaranteed JSON schema enforcement via Instructor and Pydantic.
:::
:::::

::: notes
RAG Facts Check is not a prototype; it is engineered for production reliability.

Because it provides native /halloumi/generate endpoint compatibility, the frontend team didn't need to rebuild their UI. They simply pointed their proxy to our service.

Every LLM response is enforced via structured Pydantic schemas using Instructor, eliminating JSON parsing errors. The codebase is backed by 190 automated tests with over 87% core module coverage, verifying everything from table splitting edge cases to regex span tolerance. It builds into a clean Docker container deployable in any Kubernetes or cloud environment.
:::

## Summary & Future Roadmap

::::: columns
::: {.column width="50%"}
### Key Takeaways
- **Claim-Level Granularity:**
  - Replaces blind faith with verifiable, sub-sentence auditing.
- **Real-World Resiliency:**
  - Table-preserving chunking and 4-stage span matching conquer messy data.
- **Production Performance:**
  - KV-cache prefix reuse makes rigorous fact-checking feasible in live chat.
:::
::: {.column width="50%"}
### What's Next
- **Specialized Local NLI Models:**
  - Transitioning routine claim checks to lightweight NLI models for sub-second classification.
- **Cross-Document Consensus:**
  - Detecting contradictions across multiple source documents.
- **Q&A:**
  - Thank you! Open for questions.
:::
:::::

::: notes
To wrap up: RAG Facts Check bridges the trust gap in enterprise RAG. By pairing markdown-aware claim extraction with KV-cache batching and 4-stage span matching, we've delivered a fact-checking pipeline that is accurate, resilient to noisy data, and fast enough for interactive use.

Looking ahead, we are exploring dedicated local NLI models to push verification latencies even lower, as well as cross-document consensus checking.

Thank you for your time, and I look forward to your questions.
:::
