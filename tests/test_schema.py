"""The response contract: the refusal is a complete, JSON-safe typed dict."""
from __future__ import annotations

import json

from crew.schema import REFUSAL_TEXT, refusal

EXPECTED_KEYS = {
    "session_id", "question", "answer", "sources", "claims",
    "in_scope", "refused", "used_web_search", "confidence",
}


def test_refusal_has_all_contract_keys():
    r = refusal("sid-1", "who won the world cup?")
    assert set(r.keys()) == EXPECTED_KEYS


def test_refusal_flags_and_empty_sources():
    r = refusal("sid-1", "who won the world cup?")
    assert r["refused"] is True
    assert r["in_scope"] is False
    assert r["used_web_search"] is False
    assert r["sources"] == []
    assert r["claims"] == []
    assert r["answer"] == REFUSAL_TEXT


def test_refusal_preserves_identifiers():
    r = refusal("sid-42", "some off-topic question")
    assert r["session_id"] == "sid-42"
    assert r["question"] == "some off-topic question"


def test_refusal_is_json_serializable():
    r = refusal("sid-1", "q")
    assert json.loads(json.dumps(r)) == r
