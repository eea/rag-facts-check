#!/usr/bin/env python3
"""Demonstrate that EdenAI's documented `reasoning_effort` values do not
disable reasoning on tensorx/qwen/qwen3.8-27b.

EdenAI API docs (https://www.edenai.co/docs/api-reference/chat/chat-completions)
state:

    reasoning_effort: enum<string> | null
        The reasoning effort level for the LLM model.
        Available options: minimal, low, medium, high, max, xhigh, disable, none

This script calls the model with each documented "off" value and checks the
actual response: if the model still returns reasoning_content / reasoning
tokens, the documented behaviour is not implemented.

No third-party dependencies (Python 3.8+ stdlib only).

Usage:
    python3 edenai-reasoning-disable-demo.py

The API key is read from the terminal (hidden input) or from the EDENAI_API_KEY
environment variable.
"""

import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://api.eu.edenai.run/v3/chat/completions"
MODEL = "tensorx/qwen/qwen3.8-27b"
TIMEOUT = 120  # seconds per request

# (label, extra request fields). "baseline" is the control.
TESTS = [
    ("baseline (no flag)", {}),
    ('reasoning_effort="disable"', {"reasoning_effort": "disable"}),
    ('reasoning_effort="none"', {"reasoning_effort": "none"}),
    ('thinking={"type": "disabled"}', {"thinking": {"type": "disabled"}}),
]


def call_edenai(api_key: str, extra: dict, probe_id: int) -> dict:
    """One chat completion call; returns the parsed response."""
    prompt = f"Say OK (probe {probe_id})"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.1,
        **extra,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    api_key = os.environ.get("EDENAI_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("EdenAI API key (sk-eden-...): ").strip()
    if not api_key:
        print("No API key provided. Aborting.")
        return 2

    print(f"Model under test: {MODEL}")
    print(f"Endpoint:         {API_URL}")
    print(
        "Claim under test:   docs say reasoning_effort 'disable'/'none' turn\n"
        "                     reasoning off (api-reference/chat/chat-completions)"
    )
    print("=" * 78)

    results = []
    for i, (label, extra) in enumerate(TESTS):
        is_off_value = label != "baseline (no flag)"
        try:
            t0 = time.time()
            data = call_edenai(api_key, extra, i)
            elapsed = time.time() - t0
            msg = data["choices"][0]["message"]
            reasoning = msg.get("reasoning_content")
            details = (
                (data.get("usage") or {}).get("completion_tokens_details") or {}
            )
            r_tokens = details.get("reasoning_tokens")
            if reasoning:
                verdict = "STILL THINKING"
            else:
                verdict = "no thinking"
            print(f"{label:34} {elapsed:5.1f}s  reasoning_tokens={r_tokens!s:>5}"
                  f"  reasoning_content={'yes' if reasoning else 'no'}  -> {verdict}")
            results.append((label, is_off_value, verdict == "no thinking"))
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            print(f"{label:34} HTTP {e.code}: {body}")
            results.append((label, is_off_value, None))
        except Exception as e:  # timeout, network, malformed response
            print(f"{label:34} ERROR: {type(e).__name__}: {e}")
            results.append((label, is_off_value, None))

    print("=" * 78)
    off_values = [(l, ok) for l, is_off, ok in results if is_off]
    failures = [l for l, ok in off_values if ok is False]
    errors = [l for l, is_off, ok in results if is_off and ok is None]

    print("RESULT")
    print("-" * 78)
    if not off_values:
        print("No disable test could be evaluated. See errors above.")
        return 2
    if failures:
        for l in failures:
            print(f"  [DOC INVALID] {l}: accepted by the API, but the model")
            print(f"                still produced reasoning (reasoning_content /")
            print(f"                reasoning_tokens present in the response).")
        print()
        print("Conclusion: the documented 'disable'/'none' values are silently")
        print("ignored for this model; reasoning cannot be turned off per request.")
        return 1
    if errors and not [l for l, ok in off_values if ok]:
        print("Only errors, no successful disable tests. Re-run to confirm.")
        return 2
    print("All documented disable values worked as documented (no thinking).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
