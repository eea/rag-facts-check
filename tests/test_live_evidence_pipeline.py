"""
Live regression test for the evidence pipeline, against a real Onyx deployment.

Pins the whole chain that the false hallucination verdicts came from::

    Onyx GET /document/chunk-info
        -> real chunk text
        -> format_documents (source budgets)
        -> what the judge actually sees

It exists because that failure was silent: the pipeline kept running, the score
just became meaningless. The fixture (``tests/fixtures/onyx_evidence_capture.json``)
is the four documents the answer generator actually selected in a captured
conversation, with their blurbs — the exact inputs that produced a 7,009-character
answer checked against 2,220 characters of snippets.

Run with::

    pytest -m live

Requires ``ONYX_URL`` and ``ONYX_API_KEY`` (loaded from the environment or ``.env``);
the tests skip when they are absent. The gateway in front of Onyx rate-limits under
concurrency, so requests are sequential with a short pause.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from rag_facts_check.prompts import format_documents  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "onyx_evidence_capture.json"
CAPTURE = json.loads(FIXTURE.read_text(encoding="utf-8"))

DOCUMENTS = CAPTURE["selected_documents"]
WINDOW = CAPTURE["window"]
PHRASES = CAPTURE["phrases_absent_from_blurbs"]
MIN_EVIDENCE_CHARS = CAPTURE["expected_evidence_chars_min"]

try:  # same .env the server reads, if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover
    pass

ONYX_URL = os.environ.get("ONYX_URL", "").rstrip("/")
ONYX_API_KEY = os.environ.get("ONYX_API_KEY", "")
API_PREFIX = os.environ.get("ONYX_API_PREFIX", "/api")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (ONYX_URL and ONYX_API_KEY),
        reason="ONYX_URL / ONYX_API_KEY not configured",
    ),
]


def _fetch_chunk(client: httpx.Client, document_id: str, chunk_ind: int) -> str | None:
    """One chunk of real text, or None past the end of the document."""
    url = f"{ONYX_URL}{API_PREFIX}/document/chunk-info"
    params = {"document_id": document_id, "chunk_id": chunk_ind}
    response = client.get(url, params=params, timeout=30)
    if response.status_code == 429:  # gateway throttle; one retry clears it
        time.sleep(1.0)
        response = client.get(url, params=params, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json().get("content") or ""


@pytest.fixture(scope="module")
def evidence() -> dict[str, str]:
    """`document_id -> joined chunk text` for each selected document."""
    client = httpx.Client(
        headers={"Authorization": f"Bearer {ONYX_API_KEY}"},
        follow_redirects=True,
    )
    fetched: dict[str, str] = {}
    try:
        for doc in DOCUMENTS:
            center = int(doc["chunk_ind"] or 0)
            texts: list[str] = []
            for chunk_ind in range(max(0, center - WINDOW), center + WINDOW + 1):
                content = _fetch_chunk(client, doc["document_id"], chunk_ind)
                if content is None and chunk_ind > center:
                    break  # past the end of the document
                if content:
                    texts.append(content)
                time.sleep(0.15)  # stay under the gateway rate limit
            fetched[doc["document_id"]] = "\n".join(texts)
    finally:
        client.close()
    return fetched


def test_fixture_blurbs_lack_the_phrases():
    """Guard the fixture itself: these are the phrases blurbs could not show."""
    blurbs = " \n ".join(
        f"{doc.get('blurb') or ''} {doc.get('semantic_identifier') or ''}" for doc in DOCUMENTS
    ).lower()
    for phrase in PHRASES:
        assert phrase.lower() not in blurbs, f"{phrase!r} unexpectedly present in a blurb"


def test_every_selected_document_yields_text(evidence):
    """The endpoint contract holds for the captured documents.

    A 404 on the *centre* chunk would mean the chunk was reindexed away, which is
    a different failure from "the phrases are missing" - name it explicitly.
    """
    empty = [
        doc["semantic_identifier"] for doc in DOCUMENTS if not evidence.get(doc["document_id"])
    ]
    assert not empty, f"chunk-info returned nothing for: {empty}"


def test_chunk_info_recovers_every_phrase(evidence):
    """The real chunk text Onyx read contains what the blurbs hid."""
    combined = " \n ".join(evidence.values()).lower()
    missing = [phrase for phrase in PHRASES if phrase.lower() not in combined]
    assert not missing, f"phrases still absent from chunk text: {missing}"


def test_chunk_text_dwarfs_the_blurbs(evidence):
    """The evidence volume the answer was actually written from."""
    chunk_chars = sum(len(text) for text in evidence.values())
    blurb_chars = sum(len(doc.get("blurb") or "") for doc in DOCUMENTS)

    assert chunk_chars >= MIN_EVIDENCE_CHARS, (
        f"only {chunk_chars} chars of chunk text, expected >= {MIN_EVIDENCE_CHARS}"
    )
    assert chunk_chars > blurb_chars * 10, (
        f"evidence multiplier {(chunk_chars / blurb_chars):.1f}x, expected > 10x"
    )


def test_corpus_reaches_the_judge_untruncated(evidence):
    """The source budgets must not clip real-world evidence.

    This is the regression `format_documents` used to have: a fixed per-document
    cap cut a 12k-character source even though the whole corpus fitted the total
    budget, hiding the passage a claim would have been matched against.
    """
    sources = [
        {"text": evidence[doc["document_id"]], "title": doc.get("semantic_identifier")}
        for doc in DOCUMENTS
        if evidence.get(doc["document_id"])
    ]
    formatted = format_documents(sources)

    assert "[truncated]" not in formatted
    assert "Remaining documents truncated" not in formatted
    for source in sources:
        assert source["text"] in formatted


def test_check_reports_partial_context_for_snippets_only(evidence):
    """A snippet-only check must be labelled, not scored as if it were complete."""
    from pydantic import ValidationError

    from rag_facts_check.server import DocumentInput, _context_quality

    kinds = []
    for doc in DOCUMENTS:
        try:
            kinds.append(DocumentInput(doc_id=doc["document_id"], text=doc["blurb"]).kind)
        except ValidationError:  # pragma: no cover - fixture guard
            pytest.fail("fixture blurb rejected as a DocumentInput")
    assert _context_quality(kinds)["level"] == "unknown"

    snippet_quality = _context_quality(["snippet"] * len(DOCUMENTS))
    assert snippet_quality["level"] == "partial"
    assert "snippet" in (snippet_quality["note"] or "")

    full_quality = _context_quality(["chunk"] * len(DOCUMENTS))
    assert full_quality["level"] == "full"
    assert full_quality["note"] is None
