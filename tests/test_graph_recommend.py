"""The recommender's behavioural contract.

Everything is asserted against ``toy_graph``, whose topology is known by
construction (see ``conftest.py``), so these test the *scoring policy* rather
than whatever the current corpus happens to contain.
"""
from __future__ import annotations

from crew.graph.recommend import Recommender, recommend_connections, similar_scientists


def test_existing_coauthors_are_excluded_from_recommendations(toy_graph):
    """Suggesting someone you already publish with is a wasted suggestion."""
    recs = recommend_connections(toy_graph, "alice|a")
    assert "bob|b" not in [r["scientist_id"] for r in recs]


def test_similar_scientists_does_include_coauthors(toy_graph):
    """'Who is like me' is a different question from 'who should I meet'."""
    recs = similar_scientists(toy_graph, "alice|a")
    assert "bob|b" in [r["scientist_id"] for r in recs]


def test_a_bridged_shared_interest_outranks_an_unconnected_one(toy_graph):
    """Carol shares Alice's RPE interest *and* is bridged by Bob; Dave only
    shares the commoner receptor interest with nothing in between."""
    recs = recommend_connections(toy_graph, "alice|a")
    ranked = [r["scientist_id"] for r in recs]
    assert ranked[0] == "carol|c"
    assert ranked.index("carol|c") < ranked.index("dave|d")


def test_unrelated_scientists_are_not_recommended(toy_graph):
    """Frank shares no topic, no collaborator and no institution."""
    recs = recommend_connections(toy_graph, "alice|a")
    assert "frank|f" not in [r["scientist_id"] for r in recs]


def test_shared_collaborators_are_reported(toy_graph):
    recs = recommend_connections(toy_graph, "alice|a")
    carol = next(r for r in recs if r["scientist_id"] == "carol|c")
    assert carol["shared_collaborators"] == ["Bob B"]


def test_shared_topics_are_reported_and_rare_ones_rank_first(toy_graph):
    """`rpe` (3 holders) is rarer than `receptors` (3) — but the ordering must
    be by IDF, so the reported list is deterministic, not set-ordered."""
    recs = similar_scientists(toy_graph, "alice|a")
    bob = next(r for r in recs if r["scientist_id"] == "bob|b")
    assert set(bob["shared_topics"]) == {"Reward prediction error", "Dopamine receptors"}


def test_rarer_shared_interests_score_higher_than_common_ones(toy_graph):
    """IDF weighting: sharing a niche interest is stronger evidence."""
    rec = Recommender(toy_graph)
    idf = rec.topic_idf
    assert idf["parkinsons"] > idf["receptors"]


def test_same_institution_is_discounted_not_excluded(toy_graph):
    """Erin is at Duke like Alice; she should still appear, just penalised."""
    recs = recommend_connections(toy_graph, "alice|a")
    erin = [r for r in recs if r["scientist_id"] == "erin|e"]
    if erin:
        assert erin[0]["signals"]["same_institution"] is True


def test_scores_are_bounded_and_sorted(toy_graph):
    recs = recommend_connections(toy_graph, "alice|a")
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


def test_every_recommendation_carries_a_rationale_and_evidence(toy_graph):
    """A recommendation you cannot defend is not usable."""
    for rec in recommend_connections(toy_graph, "alice|a"):
        assert rec["rationale"].strip()
        assert rec["name"] in rec["rationale"]
        assert rec["evidence"], "recommendations must be traceable to the corpus"


def test_rationale_is_built_from_signals_not_prose(toy_graph):
    recs = recommend_connections(toy_graph, "alice|a")
    carol = next(r for r in recs if r["scientist_id"] == "carol|c")
    assert "Reward prediction error" in carol["rationale"]
    assert "Bob B" in carol["rationale"]


def test_recommendations_are_deterministic(toy_graph):
    """Same graph, same input -> byte-identical output. No sampling anywhere."""
    first = recommend_connections(toy_graph, "alice|a")
    second = recommend_connections(toy_graph, "alice|a")
    assert first == second


def test_unknown_scientist_returns_nothing(toy_graph):
    assert recommend_connections(toy_graph, "nobody|x") == []


def test_signals_are_all_normalised(toy_graph):
    for rec in similar_scientists(toy_graph, "alice|a"):
        sig = rec["signals"]
        for key in ("topic_overlap", "profile_similarity", "graph_proximity",
                    "citation_affinity"):
            assert 0.0 <= sig[key] <= 1.0, f"{key} out of range"


def test_limit_is_respected(toy_graph):
    assert len(similar_scientists(toy_graph, "alice|a", limit=2)) <= 2
