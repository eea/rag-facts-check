#!/usr/bin/env python3
"""
Probe Onyx ``GET /document/chunk-info`` for the documents behind a captured chat.

Answers one question: can we replace the search *blurbs* the frontend sends to the
fact-checker with the *real chunk text* Onyx read from the vector index, using only
existing Onyx endpoints?  See ``docs/architecture/onyx-evidence-contract.md``.

Requires the ``[server]`` extras (httpx) and Onyx credentials:

    export ONYX_URL=https://aihub.eea.europa.eu
    export ONYX_API_KEY=...

Usage:
    # centre chunk only (1 request per LLM-selected document)
    python scripts/probe_chunk_info.py --stream capture.jsonl

    # emulate Onyx's FULL_DOCUMENT expansion (+/-5 chunks)
    python scripts/probe_chunk_info.py --stream capture.jsonl --window 5

    # every retrieved document, not just the selected ones
    python scripts/probe_chunk_info.py --stream capture.jsonl --all-docs

    # just verify auth against one known chunk
    python scripts/probe_chunk_info.py --stream capture.jsonl --check-auth

``--stream`` is a captured Onyx v3 SSE stream (one JSON packet per line, as saved by
the browser devtools export or ``reconstitute_chatbot_response.py``).  Lines that do
not parse are skipped — captures are usually clipped at the head.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx

# Longest document_id seen in a capture (Vespa field limit).
DOC_ID_LIMIT = 793

DEFAULT_TERMS = (
    "Copernicus,European Investment Bank,Climate Awareness Bond,"
    "Forest Information System,28 headline indicators,Eurostat,ETS"
)


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file (``#`` comments, optional quotes)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_stream(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return ``(selected_docs, all_final_docs)`` keyed by ``document_id``.

    ``selected_docs`` come from ``search_tool_documents_delta`` packets — the
    sections the answer LLM actually asked for.  ``all_final_docs`` come from
    ``message_start.final_documents`` — every retrieval candidate.
    """
    selected: dict[str, dict] = {}
    final: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            packet = json.loads(line)
        except json.JSONDecodeError:
            continue
        obj = packet.get("obj") or {}
        ptype = obj.get("type")
        if ptype in ("search_tool_documents_delta", "internal_search_tool_delta"):
            for doc in obj.get("documents") or []:
                if doc.get("document_id"):
                    selected.setdefault(doc["document_id"], doc)
        elif ptype == "message_start":
            for doc in obj.get("final_documents") or []:
                if doc.get("document_id"):
                    final.setdefault(doc["document_id"], doc)
    if not selected and not final:
        sys.exit(f"No documents found in {path} (is this an Onyx stream capture?)")
    return selected, final


def fetch_chunk(
    client: httpx.Client,
    base: str,
    prefix: str,
    document_id: str,
    chunk_ind: int,
) -> tuple[int, dict | str]:
    """Fetch one chunk.  Retries once after 1s on 429 (gateway rate limit)."""
    url = (
        f"{base}{prefix}/document/chunk-info"
        f"?document_id={quote(document_id, safe='')}&chunk_id={chunk_ind}"
    )
    try:
        response = client.get(url, timeout=30)
        if response.status_code == 429:
            time.sleep(1.0)
            response = client.get(url, timeout=30)
    except httpx.HTTPError as exc:
        return -1, f"transport error: {exc}"
    if response.status_code != 200:
        return response.status_code, response.text[:200]
    return 200, response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Onyx chunk-info for the evidence behind a captured chat."
    )
    parser.add_argument("--stream", type=Path, required=True, help="captured SSE stream")
    parser.add_argument("--env-file", type=Path, help="optional KEY=VALUE file")
    parser.add_argument("--base-url", default=os.environ.get("ONYX_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("ONYX_API_KEY", ""))
    parser.add_argument("--prefix", default="/api", help="Onyx API_PREFIX")
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="also fetch chunk_ind-window..+window (2 == INCLUDE_ADJACENT_SECTIONS, "
        "5 == FULL_DOCUMENT expansion)",
    )
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help="probe every final_documents entry, not just the selected ones",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap documents probed")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel chunk requests (keep low: the aihub gateway 429s ~30 req/s)",
    )
    parser.add_argument("--check-auth", action="store_true", help="only ping the endpoint once")
    parser.add_argument(
        "--terms",
        default=DEFAULT_TERMS,
        help="comma-separated phrases: report presence in blurbs vs real chunk text",
    )
    args = parser.parse_args()

    env = read_env_file(args.env_file) if args.env_file else {}
    base = (args.base_url or env.get("ONYX_URL", "")).rstrip("/")
    api_key = args.api_key or env.get("ONYX_API_KEY", "")
    if not base or not api_key:
        sys.exit("Missing ONYX_URL / ONYX_API_KEY (env or --env-file)")

    prefix = args.prefix if args.prefix.startswith("/") else f"/{args.prefix}"
    client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        follow_redirects=True,
    )

    if args.check_auth:
        status, payload = fetch_chunk(
            client,
            base,
            prefix,
            "https://www.eea.europa.eu/en/about/policy-corner-eu-policies-we-support",
            0,
        )
        print(f"chunk-info -> HTTP {status}")
        print(json.dumps(payload, indent=2)[:600] if status == 200 else payload)
        return

    selected, final = load_stream(args.stream)
    docs = (final if args.all_docs else selected) or final
    if args.limit:
        docs = dict(list(docs.items())[: args.limit])

    print(f"Onyx: {base}{prefix}")
    print(
        f"Documents: {len(selected)} selected / {len(final)} retrieved "
        f"— probing {'ALL' if args.all_docs else 'SELECTED'} ({len(docs)})"
    )
    print(f"Window: +/-{args.window} chunks around chunk_ind\n")

    requests: list[tuple[str, int]] = []
    for document_id, doc in docs.items():
        if len(document_id) > DOC_ID_LIMIT:
            print(f"  skipping document_id longer than {DOC_ID_LIMIT} chars")
            continue
        center = int(doc.get("chunk_ind") or 0)
        for offset in range(-args.window, args.window + 1):
            chunk_ind = center + offset
            if chunk_ind < 0:
                continue
            requests.append((document_id, chunk_ind))

    started = time.monotonic()
    results: dict[tuple[str, int], tuple[int, dict | str]] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_chunk, client, base, prefix, document_id, chunk_ind): (
                document_id,
                chunk_ind,
            )
            for document_id, chunk_ind in requests
        }
        for future in cf.as_completed(futures):
            results[futures[future]] = future.result()
    elapsed = time.monotonic() - started

    ok = sum(1 for status, _ in results.values() if status == 200)
    misses: dict[int, int] = {}
    sample_failure: tuple[int, str] | None = None
    for status, payload in results.values():
        if status != 200:
            misses[status] = misses.get(status, 0) + 1
            if sample_failure is None and status != 404:
                sample_failure = (status, str(payload))

    print(f"{'document':<46} {'chunk':>5} {'blurb':>7} {'content':>8} {'tokens':>7}")
    print("-" * 78)
    total_blurb = total_content = total_tokens = 0
    for document_id, doc in docs.items():
        center = int(doc.get("chunk_ind") or 0)
        blurb = len(doc.get("blurb") or "")
        chars = tokens = 0
        for offset in range(-args.window, args.window + 1):
            payload = results.get((document_id, center + offset))
            if payload and payload[0] == 200:
                body = payload[1]
                chars += len(body.get("content") or "")
                tokens += body.get("num_tokens") or 0
        total_blurb += blurb
        total_content += chars
        total_tokens += tokens
        title = (doc.get("semantic_identifier") or "").replace("\n", " ")[:44]
        print(f"{title:<46} {center:>5} {blurb:>7} {chars:>8} {tokens:>7}")
    print("-" * 78)

    print(
        f"\n{ok}/{len(requests)} chunk requests OK in {elapsed:.1f}s"
        + (f"  failures: {misses}" if misses else "")
    )
    if sample_failure:
        print(f"  sample non-404 failure: HTTP {sample_failure[0]} {sample_failure[1]!r}")
    print(f"  sent today (blurbs):        {total_blurb:>8} chars")
    print(f"  real chunk text available:  {total_content:>8} chars  ({total_tokens} tokens)")
    if total_blurb:
        print(f"  evidence multiplier:        {total_content / total_blurb:>8.1f}x")

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    if terms:
        blurb_text = " \n ".join(
            (doc.get("blurb") or "") + " " + (doc.get("semantic_identifier") or "")
            for doc in docs.values()
        ).lower()
        chunk_text = " \n ".join(
            payload[1].get("content") or "" for payload in results.values() if payload[0] == 200
        ).lower()
        print(f"\n{'phrase':<28} {'in blurb':>9} {'in chunks':>10}")
        print("-" * 50)
        for term in terms:
            print(
                f"{term:<28} {str(term.lower() in blurb_text):>9} "
                f"{str(term.lower() in chunk_text):>10}"
            )

    print(
        "\nVerdict: "
        + (
            "chunk-info returns real text — usable as fact-checker sources."
            if total_content
            else "no content returned — check auth / prefix / chunk ids."
        )
    )


if __name__ == "__main__":
    main()
