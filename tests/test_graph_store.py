"""Graph persistence, lookup and the derived co-authorship projection."""
from __future__ import annotations

import json

from crew.graph.store import ScientistGraph


def test_save_load_round_trip(toy_graph, tmp_path):
    path = toy_graph.save(tmp_path / "graph.json")
    reloaded = ScientistGraph.load(path)
    assert reloaded.scientists == toy_graph.scientists
    assert reloaded.topics == toy_graph.topics
    assert reloaded.edges == toy_graph.edges


def test_persisted_graph_is_plain_json(toy_graph, tmp_path):
    """No custom encoder needed — every node type is a TypedDict, i.e. a dict."""
    path = toy_graph.save(tmp_path / "graph.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {"stats", "scientists", "institutions", "topics",
                        "papers", "edges"}


def test_find_scientist_matches_canonical_forms(toy_graph):
    """Both a full name and its initialised form resolve to one id."""
    assert toy_graph.find_scientist("Alice A")[0][0] == "alice|a"
    assert toy_graph.find_scientist("alice a")[0][0] == "alice|a"


def test_find_scientist_handles_surname_only(toy_graph):
    assert toy_graph.find_scientist("Carol")[0][0] == "carol|c"


def test_find_scientist_returns_nothing_for_a_stranger(toy_graph):
    assert toy_graph.find_scientist("Zaphod Beeblebrox") == []


def test_find_topic_matches_label_and_alias(toy_graph):
    assert toy_graph.find_topic("Reward prediction error") == "rpe"
    assert toy_graph.find_topic("Parkinson's disease") == "parkinsons"


def test_find_institution_is_fuzzy(toy_graph):
    assert toy_graph.find_institution("Duke") == "duke"


def test_coauthor_projection_is_undirected_and_scientists_only(toy_graph):
    g = toy_graph.coauthor_graph
    assert g.has_edge("alice|a", "bob|b")
    assert g.has_edge("bob|b", "alice|a")       # undirected
    assert "rpe" not in g                       # topics excluded
    assert "duke" not in g                      # institutions excluded


def test_collaborators_excludes_indirect_links(toy_graph):
    """Carol is Bob's co-author, not Alice's."""
    assert toy_graph.collaborators("alice|a") == ["bob|b"]
    assert sorted(toy_graph.collaborators("bob|b")) == ["alice|a", "carol|c"]


def test_typed_multigraph_keeps_node_kinds(toy_graph):
    g = toy_graph.nx
    assert g.nodes["alice|a"]["kind"] == "scientist"
    assert g.nodes["rpe"]["kind"] == "topic"
    assert g.nodes["duke"]["kind"] == "institution"


def test_profile_text_describes_work_not_the_name(toy_graph):
    """Two people must look alike because of what they study, not because
    their names are spelled similarly."""
    text = toy_graph.profile_text("alice|a")
    assert "Reward prediction error" in text
    assert "Alice" not in text


def test_top_scientists_puts_corpus_authors_first(toy_graph):
    top = toy_graph.top_scientists(3)
    assert all(toy_graph.scientists[s]["corpus_author"] for s in top)


def test_summary_is_human_readable(toy_graph):
    assert "scientists" in toy_graph.summary()


def test_missing_graph_file_raises_actionable_error(tmp_path):
    try:
        ScientistGraph.load(tmp_path / "absent.json")
    except FileNotFoundError as exc:
        assert "crew.graph.build" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
