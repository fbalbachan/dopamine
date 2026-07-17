"""The grounding guarantee: cited evidence ids resolve to real, deduped
sources, unknown ids are dropped, and used_web_search reflects ground truth."""
from __future__ import annotations

from crew.chat import DopamineCrew, _ClaimOut, _SynthOut
from crew.tools import EvidenceRegistry


def _registry_with(paper=True, web=True) -> EvidenceRegistry:
    reg = EvidenceRegistry()
    if paper:
        reg.add_paper("a.pdf", 1, "snippet A")   # -> P1
        reg.add_paper("b.pdf", 2, "snippet B")   # -> P2
    if web:
        reg.add_web("http://x", "Title X", "snippet W")  # -> W1
    return reg


def test_cited_ids_become_sources_in_first_seen_order():
    reg = _registry_with()
    synth = _SynthOut(
        answer="Claim one [P1]. Claim two [P2][W1].",
        claims=[
            _ClaimOut(statement="Claim one", evidence_ids=["P1"]),
            _ClaimOut(statement="Claim two", evidence_ids=["P2", "W1"]),
        ],
        used_web_search=False,
        confidence="high",
    )
    resp = DopamineCrew._resolve("s", "q", synth, reg)

    assert [s["type"] for s in resp["sources"]] == ["paper", "paper", "web"]
    assert resp["sources"][0]["file"] == "a.pdf" and resp["sources"][0]["page"] == 1
    assert resp["sources"][2]["url"] == "http://x"
    # claim -> source index remap
    assert resp["claims"][0]["source_indices"] == [0]
    assert resp["claims"][1]["source_indices"] == [1, 2]


def test_unknown_evidence_ids_are_dropped():
    reg = _registry_with(web=False)  # only P1, P2 exist
    synth = _SynthOut(
        answer="A claim.",
        claims=[_ClaimOut(statement="A claim", evidence_ids=["P1", "P9", "W7"])],
        used_web_search=True,
        confidence="medium",
    )
    resp = DopamineCrew._resolve("s", "q", synth, reg)
    # Only P1 is real -> one source, and the claim maps only to it.
    assert len(resp["sources"]) == 1
    assert resp["claims"][0]["source_indices"] == [0]


def test_used_web_search_is_ground_truth_not_model_claim():
    # Model says False, but a web result was actually registered -> True.
    reg = _registry_with()
    synth = _SynthOut(answer="x [W1]", claims=[], used_web_search=False, confidence="low")
    resp = DopamineCrew._resolve("s", "q", synth, reg)
    assert resp["used_web_search"] is True

    # No web evidence -> False regardless.
    reg2 = _registry_with(web=False)
    synth2 = _SynthOut(answer="x [P1]", claims=[], used_web_search=True, confidence="low")
    resp2 = DopamineCrew._resolve("s", "q", synth2, reg2)
    assert resp2["used_web_search"] is False


def test_inline_answer_ids_are_deduped_into_sources():
    reg = _registry_with(web=False)
    synth = _SynthOut(
        answer="First [P1]. Again [P1]. And [P2].",
        claims=[],
        used_web_search=False,
        confidence="high",
    )
    resp = DopamineCrew._resolve("s", "q", synth, reg)
    assert len(resp["sources"]) == 2  # P1 counted once


def test_resolved_response_is_in_scope_and_not_refused():
    reg = _registry_with()
    synth = _SynthOut(answer="ok [P1]", claims=[], used_web_search=False, confidence="high")
    resp = DopamineCrew._resolve("sid", "the question", synth, reg)
    assert resp["in_scope"] is True
    assert resp["refused"] is False
    assert resp["session_id"] == "sid"
    assert resp["question"] == "the question"
    assert resp["confidence"] == "high"
