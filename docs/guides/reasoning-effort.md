---
type: Guide
title: Disabling Reasoning on Qwen 3.8
description: Backend comparison (vLLM, LiteLLM gateway, EdenAI, llama.cpp) and per-request thinking control for Qwen 3.8.
tags: [llm, qwen, reasoning, vllm, litellm, edenai]
timestamp: '2026-09-14T00:00:00Z'
---

# Disabling reasoning on `edenai/qwen3.8-27b` via the EEA LiteLLM gateway

**Date:** 2026-09-14
**Model:** `edenai/qwen3.8-27b` (Qwen3.8-27B, reasoning model)
**Gateway:** `https://llmgw.eea.europa.eu/v1` (LiteLLM proxy 1.91.0 → EdenAI v3 `api.eu.edenai.run/v3` → tensorx)
**Goal:** test the model for the hallucination (Halloumi) check with reasoning **off**, since
thinking tokens count against `max_tokens` and cost significant latency.

---

## 1. Background: why reasoning off matters

- Thinking tokens consume the `max_tokens` budget. With the default 512-token budget the model
  spent the entire budget on thinking and returned `content: null` /
  `finish_reason: "length"` — no output at all.
- Even at 16384 tokens, easy prompts are fine but complex prompts (claim extraction /
  verification) generated 10K–40K chars of thinking, making single calls take 30 s–3 min and
  full fact-check requests take ~8 minutes.
- The Qwen3.8-27B [model card](https://huggingface.co/Qwen/Qwen3.8-27B) states thinking "is on
  by default and can be disabled per request":
  - non-thinking mode: `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`
  - top-level `reasoning_effort` with levels `xhigh` (default) / `medium` / `low`
  - `preserve_thinking` (multi-turn only, not relevant to our single-turn calls)
- The [EdenAI API docs](https://www.edenai.co/docs/api-reference/chat/chat-completions#body-reasoning-effort-one-of-0)
  document a top-level `reasoning_effort` enum:
  `minimal | low | medium | high | max | xhigh | disable | none`.

## 2. Test methodology

- All tests: raw `POST /v1/chat/completions` against the gateway, non-streaming.
- **Cache contamination was a real hazard:** the gateway/provider returns *identical cached
  responses* (in ~1 s) for identical prompts. Every comparison used a unique prompt suffix
  per run (e.g. `(run 0-delta)`).
- Two standard prompts were used:
  - **Easy:** *"Explain the EU's 2030 emissions reduction target in three sentences."*
  - **Hard:** the real claim-extraction prompt built from
    `mock_datasets/climate_change_hallucinated.json` (~711 prompt tokens).
- Thinking volume measured as `len(reasoning_content)` (chars); baseline = same prompt, no
  reasoning flags.

## 3. Results

### 3.1 Flags that do not work

| Parameter (as sent) | Result |
|---|---|
| top-level `enable_thinking: false` | ignored (thinking continues) |
| top-level `reasoning: false`, `thinking: false` | ignored / 400 `UnsupportedParamsError` |
| top-level `reasoning_effort: "low"/"medium"` (official Qwen levels) | **400** — LiteLLM validates against the OpenAI spec and rejects the param for this model before it reaches the provider |
| top-level `reasoning_effort: "none"` | 400 (same validation) |
| top-level `max_reasoning_tokens: 0 / 10` | ignored (identical full thinking both times) |
| `chat_template_kwargs: {"enable_thinking": false}` (top-level) | ignored |
| `/no_think` token in user or system message | ignored (only marginally shorter thinking) |

### 3.2 The pass-through mechanism: `extra_body`

LiteLLM merges unknown parameters into the provider request's `extra_body` for
OpenAI-compatible providers — so **params nested under `extra_body` bypass the proxy's
param validation and are forwarded verbatim to EdenAI**. This is the only way to reach
EdenAI's native `reasoning_effort` field through the gateway:

```json
{"extra_body": {"reasoning_effort": "disable"}}
```

### 3.3 Effect of the working knobs (hard extraction prompt)

| Run | Flag | thinking (chars) | time |
|---|---|---|---|
| A | none (baseline) | 31,260 | 31 s |
| B | `reasoning: {"effort": "minimal"}` | 3,387 | 55 s |
| C | `reasoning: {"effort": "off"}` | 14,353 | 103 s |
| D | `reasoning: {"effort": "disable"}` | 19,591 | 147 s |
| E | `extra_body: {"reasoning_effort": "disable"}` | 3,203 | 41 s |
| F | `extra_body: {"reasoning_effort": "none"}` | 11,969 | 21 s |
| G | `extra_body: {"enable_thinking": false}` | 3,260 | 45 s |
| H | `extra_body` combo (`reasoning_effort=disable` + `enable_thinking=false` + `chat_template_kwargs`) | 40,606 / 3,288 | 41–45 s |

Controlled A/B (same prompt, unique suffixes):

| Pair | `extra_body.reasoning_effort=disable` | baseline |
|---|---|---|
| 1 | 3,256 | 19,574 |
| 2 | 25,563 | 32,453 |

### 3.4 Interpretation

- `reasoning: {"effort": ...}` (LiteLLM's own field) is **not** mapped to EdenAI's
  `reasoning_effort`; the gateway passes it through and EdenAI interprets it loosely:
  `minimal`/`off`-like values reduce thinking, `disable`/`none` in *this* field do not.
- `extra_body.reasoning_effort=disable` (EdenAI's native field) **consistently reduces**
  thinking (lower in both A/B pairs, typically ~80–90% less on good runs) **but does not
  turn it off** — run 2 still produced 25.5K chars, and stacking more disable signals made
  one run *worse* (40K).
- Thinking volume is inherently variable (baseline itself ranged 19K–32K across pairs).
- `extra_body: {"thinking": {...}}` was also probed: EdenAI's schema accepts the field
  (it demands a dict — bools/strings 422 with "Input should be a valid dictionary") but
  the backend ignores every tested shape (`{type: disabled}`, `{enabled: false}`,
  `{effort: none}`) — all produced full baseline-level thinking (~19–21K chars).
- `extra_body: {"reasoning": false}`: accepted, full thinking (14.5K).
- Conclusion: **no client-side encoding of EdenAI's documented
  `reasoning_effort: disable|none` (or any other flag) turns thinking off on the
  `edenai/qwen3.8-27b` route.** The tensorx deployment does not implement true
  per-request non-thinking mode; all flags are, at best, "think less" nudges.
  The same model behind `Inhouse-LLM/qwen3.8-27b` (vLLM) disables cleanly — use that
  route. If disabling on the EdenAI route is ever required, ask EdenAI whether
  `reasoning_effort=disable` is actually implemented for Qwen3.8 via their tensorx
  integration; the parameter is accepted but observably no-ops there.

### 3.5 Direct EdenAI access (2026-09-15)

With a direct EdenAI API key (`api.eu.edenai.run/v3`, model
`tensorx/qwen/qwen3.8-27b` — the same model ID and `provider: tensorx` as the
gateway route), the disable flags were retested without LiteLLM in the path:

| Parameter | Result |
|---|---|
| `reasoning_effort: "disable"` | accepted (200), thinking continues (29 reasoning tokens on "Say OK") |
| `reasoning_effort: "none"` | accepted (200), identical |
| `thinking: {"type": "disabled"}` | accepted (200), identical |

**Direct access changes nothing** — the no-op lives in EdenAI's tensorx
integration, not in LiteLLM. The direct route adds no capability over
`.env.llmgw-qwen` (kept as `.env.edenai` for reference).

### 3.6 Working EdenAI disable: nested `extra_body` provider pass-through (2026-09-15)

The missing piece: LiteLLM **flattens** the request's `extra_body` into the top
level of the body it forwards to EdenAI. So sending
`extra_body: {"chat_template_kwargs": ...}` put the flag at EdenAI's top level,
where it was ignored. EdenAI's documented provider pass-through is *its own*
`extra_body` field — reachable only by **double nesting**:

```json
"extra_body": {"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}
```

Results on `edenai/qwen3.8-27b` via llmgw (hard extraction prompt):

| Run | Time | reasoning_content |
|---|---|---|
| 1 | 11.2s | NONE |
| 2 | 53.1s | 4,551 chars (one miss — likely routed to a node without the flag applied) |
| 3–6 | 7.7–12.1s | NONE each |

5/6 clean, ~10s per call. Full fact-check pipeline: 178–259s (the route is
slow even without thinking) — so `.env.llmgw-inhouse` (10.4s) remains the
preferred public backend; `.env.llmgw-qwen` now carries this config as a
working-but-slow alternative.

Also verified with this fix in place: the taskman-307516 workaround
(`allowed_openai_params: ["reasoning_effort"]` in the request body) does make
LiteLLM accept top-level `reasoning_effort` — but `disable` is still a no-op
on the tensorx backend (22.8K chars thinking on the hard prompt), confirming
the docs gap. The permanent fix remains the proxy-side
`model_info.allowed_openai_params` configuration.

## 4. Backend comparison (2026-09-14, same dataset: `climate_change_hallucinated`)

The EdenAI/LiteLLM path was not a dead end unique to its flags — the winning move was
**bypassing it entirely** and using the vLLM deployment that powers the gateway directly:

| Backend | Thinking | 1 hard call | Total time | Score | Flags |
|---|---|---|---|---|---|
| EdenAI via llmgw, `reasoning.effort=minimal` | reduced, not off | 30–55s | 109.7s | 6.0 | 3/5 |
| EdenAI via llmgw, nested `extra_body` pass-through (§3.6) | off (5/6 clean) | 7.7–12.1s (53s once) | 178–259s | 6.0 | 3/5 |
| EdenAI **direct** (`api.eu.edenai.run`, §3.5) | disable flags no-op | n/a (key expired after tiny prompts) | n/a | n/a | n/a |
| llama.cpp local :4000 (startup `--reasoning on`) | on (4096 budget) | 90s | 306.5s | 6.0 | 3/5 |
| llama.cpp local :4000 + per-request `enable_thinking=false` | *unverified — flag did not apply without a server restart* | 8s* | 42.3s | 7.6 | 2/5 |
| **vLLM `gpu01.pdmz.eea:9000` direct** | **genuinely off per request** | **2s** | **10.0s** | 7.0 | 3/6 |
| **llmgw `Inhouse-LLM/qwen3.8-27b`** + `chat_template_kwargs` | **genuinely off per request** | **3s** | **10.4s** | 8.0 | 2/6 |

\* llama.cpp per-request 8s is not reliable (flag may not have applied without a restart).

**On "llmgw vs EdenAI direct" speed:** they run the *same tensorx backend* — the
responses carry identical model ID and `provider: tensorx` — and LiteLLM adds only
~8ms overhead per call (measured in gateway headers:
`x-litellm-overhead-duration-ms: 8.158`). So there is **no meaningful speed
difference between calling EdenAI directly or through llmgw**; the large gaps in the
table come from the *backends* (tensorx vs vLLM vs llama.cpp) and from whether
thinking is actually off, not from the proxy.

- The `Inhouse-LLM/qwen3.8-27b` route on the public gateway is the **same gpu01 vLLM
  backend**, and it accepts the thinking-off flag through LiteLLM: top-level
  `chat_template_kwargs: {"enable_thinking": false}` (and the `extra_body`-nested
  variant) return in ~3s with **no reasoning content** on the hard probe, vs 30s / 21.7K
  chars baseline. This answers the open question: **LiteLLM is not the problem** — it
  passes `chat_template_kwargs` through verbatim; the EdenAI route failed because the
  tensorx backend ignores the flag, not because the proxy dropped it.
- On this route, top-level `enable_thinking: false` is still ignored (29.6K thinking) and
  top-level `reasoning_effort` is still rejected with 400 — `chat_template_kwargs` is the
  only working form.
- Best overall: llmgw `Inhouse-LLM/qwen3.8-27b` — public reachability, 10.4s, highest
  score (8.0). Equivalent to direct vLLM (10s, 7.0) plus no intranet dependency.

- vLLM honours `chat_template_kwargs: {"enable_thinking": false}` per request (single-call
  probe: 2s vs 29s). No proxy, no restart, no API key.
- EdenAI's documented `provider_params` passthrough was also tested through the gateway
  (`enable_thinking`, `chat_template_kwargs`, `reasoning_effort=disable`, `thinking=false`
  under `provider_params.tensorx`): all ignored (25K–52K chars thinking, some worse than
  baseline).
- `glm-5.3-flash` (via llmgw) was tested as an alternative model: slower (264s) and
  over-strict (flagged 5/5 claims including directly-supported ones) — rejected.
- Quality across all qwen backends is comparable; the fabricated GCMA/SSP2-4.5 claims are
  flagged in every configuration.

**Production config (recommended, public):**

```bash
LLM_API_BASE=https://llmgw.eea.europa.eu/v1
LLM_API_KEY=sk-...
LLM_MODEL=Inhouse-LLM/qwen3.8-27b
LLM_MAX_TOKENS=16384
LLM_TIMEOUT=300
LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
```

Equivalent intranet config (no key needed):

```bash
LLM_API_BASE=http://gpu01.pdmz.eea:9000/v1
LLM_API_KEY=not-needed
LLM_MODEL=qwen/qwen3.8-27b
LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
```

Server-side `/halloumi/generate` measured at **6.6s** with the intranet config (was
~8 min via the llmgw edenai route). All variants live in `.env.*` files, switched with
`make use-env ENV=<name>`.

## 5. Recommendations

1. ~~Keep the current config (`minimal`)~~ — superseded by §4: use the **vLLM direct
   endpoint** with `enable_thinking=false`; the llmgw config is the fallback.
2. **Ask the gateway admin** to make non-thinking mode real, e.g.:
   - add the model to LiteLLM's `model_list` with
     `extra_params: {"chat_template_kwargs": {"enable_thinking": false}}`, or add a second
     route (e.g. `qwen3.8-27b-instruct`) with that config; and/or
   - redeploy via vLLM/SGLang where the Qwen3.8 chat template actually honors
     `enable_thinking`; and/or
   - configure LiteLLM to pass through `reasoning_effort` / `chat_template_kwargs`
     instead of validating them out.
3. **Latency mitigations available now** (per request, `options`):
   - `use_evidence_retrieval: false` — removes one LLM call per claim (biggest win on small
     doc sets);
   - `max_claims` — caps verification count;
   - keep the server's default `batch_size: 20` for batched verification.
4. **Gateway admin evidence** to attach: the A/B table in §3.3 plus the 400 error text
   `litellm.UnsupportedParamsError: openai does not support parameters: ['reasoning_effort']`.

## 6. Appendix: request examples

```bash
# Baseline
curl https://llmgw.eea.europa.eu/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"edenai/qwen3.8-27b","messages":[{"role":"user","content":"..."}],
       "max_tokens":16384,"temperature":0.1}'

# Best client-side knob (current production config)
curl ... -d '{"model":"edenai/qwen3.8-27b","messages":[...],"max_tokens":16384,
              "temperature":0.1,"reasoning":{"effort":"minimal"}}'

# EdenAI-native param via pass-through (reduces but does not disable)
curl ... -d '{"model":"edenai/qwen3.8-27b","messages":[...],"max_tokens":16384,
              "temperature":0.1,"extra_body":{"reasoning_effort":"disable"}}'
```
