"""Out-of-scope guardrail logic, driven with a fake retriever so the decision
boundary is deterministic and needs no index or LLM."""
from __future__ import annotations

from crew.config import settings
from crew.scope import _lexicon_hit, check_scope

FLOOR = settings.scope_threshold


class FakeRetriever:
    """Stand-in that returns a fixed similarity so scope logic is deterministic."""

    def __init__(self, similarity: float):
        self._similarity = similarity

    def max_similarity(self, query: str) -> float:
        return self._similarity


def test_lexicon_hit():
    assert _lexicon_hit("What do D2 receptors do?")
    assert _lexicon_hit("Tell me about dopamine")
    assert not _lexicon_hit("What is the capital of France?")


def test_high_similarity_is_in_scope():
    v = check_scope("How do dopamine receptors work?", FakeRetriever(FLOOR + 0.4))
    assert v.in_scope is True
    assert "high paper similarity" in v.reason


def test_far_below_floor_non_domain_is_refused():
    v = check_scope("What is the best pizza in Rome?", FakeRetriever(FLOOR - 0.2))
    assert v.in_scope is False


def test_far_below_floor_but_domain_vocab_is_rescued():
    # Low embedding score but clearly in-domain wording -> stay in scope.
    v = check_scope("What does dopamine do?", FakeRetriever(FLOOR - 0.2))
    assert v.in_scope is True
    assert "vocabulary" in v.reason


def test_gray_band_uses_llm_tiebreak(monkeypatch):
    # Force the tie-breaker path (no lexicon term, similarity inside the band).
    monkeypatch.setattr("crew.scope._llm_tiebreak", lambda q: False)
    v = check_scope("How does the brain process information?", FakeRetriever(FLOOR))
    assert v.in_scope is False

    monkeypatch.setattr("crew.scope._llm_tiebreak", lambda q: True)
    v = check_scope("How does the brain process information?", FakeRetriever(FLOOR))
    assert v.in_scope is True


def test_gray_band_lexicon_skips_llm(monkeypatch):
    # A lexicon hit resolves the gray band without spending an LLM call.
    monkeypatch.setattr(
        "crew.scope._llm_tiebreak",
        lambda q: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )
    v = check_scope("What about dopamine here?", FakeRetriever(FLOOR))
    assert v.in_scope is True
