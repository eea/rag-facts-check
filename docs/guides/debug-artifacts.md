---
type: Guide
title: Debug Artifacts
description: Local-only HAR extraction artifacts for debugging fact-check sessions.
tags: [debug, har, artifacts, local]
timestamp: '2025-08-04T00:00:00Z'
---

# Debug Artifacts

> **Local only.** The `artifacts/` directory is gitignored (`artifacts/*.har`) and
> exists only for local development debugging. It is **not** committed to the
> repository.

## Purpose

Captured HTTP Archive (`.har`) exports from browser dev tools, extracted into
per-session subfolders under `artifacts/debug-data/`. Each subfolder holds the
request/response payloads for a single fact-check session, making them easy to
inspect, diff, and replay against the API.

## Directory layout

```
artifacts/debug-data/
└── <session-name>/
    ├── send-chat-message-response.jsonl   # Chatbot streaming response
    ├── generate-request.json               # Halloumi fact-check request
    └── generate-response.json              # Halloumi fact-check response
```

Session names are descriptive slugs (e.g. `eu-doing-combat-climate-change`).

## File formats

### `send-chat-message-response.jsonl`

JSONL capture of the `/send-chat-message` SSE stream. Each line is a JSON
object representing one streaming event.

**Line 1** — Session metadata:

```json
{"user_message_id": 740626, "reserved_assistant_message_id": 740627}
```

**Remaining lines** — Streaming events with placement and object:

```json
{"placement": {"turn_index": 0, "tab_index": 0, "sub_turn_index": null, "model_index": 0}, "obj": {"type": "reasoning_delta", "reasoning": "User"}}
```

`obj.type` values:

| Type | Meaning |
|------|---------|
| `reasoning_start` | Begin of chain-of-thought |
| `reasoning_delta` | Individual reasoning token |
| `message_delta` | Individual answer token (markdown) |
| `stop` | Stream end |

To reconstruct the full answer, concatenate all `message_delta.content` fields
in order.

### `generate-request.json`

Single JSON object sent to the `/_ha/generate` endpoint for fact-checking.

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | The markdown answer text to verify |
| `sources` | `array` | Source documents, each with `{text, title, source_type, link}` |
| `maxContextSegments` | `int` | Maximum context segment count |

### `generate-response.json`

Single JSON object returned by `/_ha/generate` after fact-checking.

| Field | Type | Description |
|-------|------|-------------|
| `answer_score` | `float` | Overall quality score (0–10) |
| `claims` | `array` | Extracted claims, each with `{claimString, startOffset, endOffset, segmentIds, score, rationale, skipped}` |
| `segments` | `object` | Source segment lookup by string key (`"0"`, `"1"`, ...) |

## Producing artifacts from a HAR file

1. Export a HAR from browser dev tools (Network panel → "Save all as HAR")
2. Place the `.har` in `artifacts/debug-data/`
3. Extract with a script like:

```python
import json

with open("1.har") as f:
    data = json.load(f)

for i, entry in enumerate(data["log"]["entries"]):
    url = entry["request"]["url"]
    # Extract response text from entry["response"]["content"]["text"]
    # Extract POST data from entry["request"]["postData"]["text"]
```

## See also

* [Web Service](/guides/web-service.md) — endpoint schemas and integration
* [Output Format](/architecture/output-format.md) — CheckReport structure
* [Debug Command: debug-check](/debug-prompts/debug-check.md) — CLI debug pipeline
