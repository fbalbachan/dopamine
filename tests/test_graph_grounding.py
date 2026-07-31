"""The grounding guarantee for graph answers.

Same contract as ``test_resolve.py`` enforces for the Q&A crew, applied to the
graph surface: the Analyst may only cite ``[G#]`` ids the orchestrator
registered, and ``sources`` is rebuilt from the registry rather than from the
model's text. A fabricated citation therefore cannot reach the caller.
"""
from __future__ import annotations

import json

from crew.graph.chat import EvidenceRegistry, GraphCrew, _AnalystOut, _Plan
from crew.graph.schema import evidence, graph_refusal


def _resolve(synth, registry, entities=None, recommendations=None):
    return GraphCrew._resolve(
        "s1", "who should Alice meet?", _Plan(intent="recommend"), synth, registry,
        entities or [], recommendations or [],
    )


def test_cited_ids_become_sources_in_first_seen_order():
    reg = EvidenceRegistry()
    reg.add("fact one", evidence("a.pdf", 3, "snippet a"))
    reg.add("fact two", evidence("b.pdf", 7, "snippet b"))
    synth = _AnalystOut(
        answer="Second [G2] then first [G1].",
        claims=[{"statement": "Second.", "evidence_ids": ["G2"]}],
    )
    resp = _resolve(synth, reg)
    assert [s["file"] for s in resp["sources"]] == ["b.pdf", "a.pdf"]
    assert resp["claims"][0]["source_indices"] == [0]


def test_unknown_evidence_ids_are_dropped():
    """The model inventing [G9] must not produce a citation."""
    reg = EvidenceRegistry()
    reg.add("fact one", evidence("a.pdf", 1, "snippet"))
    synth = _AnalystOut(
        answer="Real [G1], invented [G9].",
        claims=[{"statement": "Both.", "evidence_ids": ["G1", "G9"]}],
    )
    resp = _resolve(synth, reg)
    assert len(resp["sources"]) == 1
    assert resp["claims"][0]["source_indices"] == [0]


def test_repeated_ids_appear_once_in_sources():
    reg = EvidenceRegistry()
    reg.add("fact", evidence("a.pdf", 1, "snippet"))
    synth = _AnalystOut(
        answer="[G1] and again [G1].",
        claims=[{"statement": "One.", "evidence_ids": ["G1", "G1"]}],
    )
    assert len(_resolve(synth, reg)["sources"]) == 1


def test_recommendations_are_passed_through_untouched():
    """Scores come from crew.graph.recommend, never from the model — so the
    resolver must not let the narration alter them."""
    reg = EvidenceRegistry()
    reg.add("fact", evidence("a.pdf", 1, "snippet"))
    recs = [{"scientist_id": "carol|c", "name": "Carol C", "score": 0.42,
             "signals": {}, "shared_topics": [], "shared_collaborators": [],
             "institutions": [], "rationale": "because", "evidence": []}]
    resp = _resolve(_AnalystOut(answer="[G1]"), reg, recommendations=recs)
    assert resp["recommendations"][0]["score"] == 0.42


def test_registry_keeps_fact_text_separate_from_provenance():
    """The fact is 'Alice is at Duke'; the provenance is the PDF page it came
    from. Conflating them puts PDF prose where the graph fact belongs."""
    reg = EvidenceRegistry()
    eid = reg.add("Alice is at Duke", evidence("a.pdf", 4, "raw pdf text"))
    assert reg.facts[eid] == "Alice is at Duke"
    assert reg.get(eid)["snippet"] == "raw pdf text"
    assert reg.get(eid)["page"] == 4


def test_answer_is_json_serialisable():
    reg = EvidenceRegistry()
    reg.add("fact", evidence("a.pdf", 1, "snippet"))
    resp = _resolve(_AnalystOut(answer="[G1]"), reg)
    assert json.loads(json.dumps(resp)) == resp


def test_refusal_contract():
    resp = graph_refusal("s1", "what is the capital of France?")
    assert resp["refused"] is True
    assert resp["in_scope"] is False
    assert resp["sources"] == []
    assert resp["claims"] == []
    assert resp["recommendations"] == []
    assert resp["question"] == "what is the capital of France?"
    assert json.loads(json.dumps(resp)) == resp


def test_evidence_snippets_are_capped():
    """Graph nodes carry evidence for thousands of people; uncapped snippets
    would balloon graph.json."""
    ev = evidence("a.pdf", 1, "x" * 900)
    assert len(ev["snippet"]) <= 301
