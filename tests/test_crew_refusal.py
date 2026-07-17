"""End-to-end guardrail path through DopamineCrew.ask (no API key required).

Out-of-scope questions are refused before any LLM/synthesis call, so this runs
fully offline. The in-scope case only asserts that scope *passes* (it then hits
the API-key gate), so it needs no key either."""
from __future__ import annotations

import json

import pytest

from crew.config import settings
from crew.chat import DopamineCrew
from crew.memory import MemoryStore


@pytest.fixture
def crew(retriever, tmp_path):
    mem = MemoryStore(db_path=tmp_path / "sessions.sqlite3")
    yield DopamineCrew(retriever=retriever, memory=mem)
    mem.close()


def test_out_of_scope_is_refused_end_to_end(crew):
    resp = crew.ask("What is the best pizza in Rome?", session_id="t1")
    assert resp["refused"] is True
    assert resp["in_scope"] is False
    assert resp["sources"] == []
    assert resp["claims"] == []
    assert resp["session_id"] == "t1"
    assert resp["question"] == "What is the best pizza in Rome?"
    assert json.loads(json.dumps(resp)) == resp   # still JSON-safe


def test_refusal_is_recorded_in_memory(crew):
    crew.ask("Who won the 2022 World Cup?", session_id="t2")
    # user question + assistant refusal
    assert len(crew.memory.history("t2")) == 2


def test_in_scope_passes_scope_then_gates_on_key(crew):
    if settings.anthropic_api_key:
        pytest.skip("API key present; would perform a live synthesis call")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        crew.ask("What is dopamine's role in reward learning?", session_id="t3")
