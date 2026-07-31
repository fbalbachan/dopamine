"""Shared fixtures.

Unit tests (schema, scope logic, grounding resolver, memory, config) need no
index and no API key. Integration tests need the RAG index; they skip cleanly
if it hasn't been built (`python -m crew.build_index`).
"""
from __future__ import annotations

import pytest

from crew.memory import MemoryStore


@pytest.fixture(scope="session")
def retriever():
    """The real retriever over the 9 papers; skip if the index isn't built."""
    from crew.rag import Retriever

    try:
        return Retriever.load()
    except Exception as exc:  # index not built
        pytest.skip(f"RAG index not available ({exc}); run `python -m crew.build_index`")


@pytest.fixture
def tmp_memory(tmp_path):
    """A MemoryStore backed by a throwaway SQLite file."""
    store = MemoryStore(db_path=tmp_path / "sessions.sqlite3")
    yield store
    store.close()


# --------------------------------------------------------------------------- #
# Knowledge-graph fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_graph():
    """A hand-built 6-person graph with known structure.

    Deliberately synthetic: every assertion about ranking, exclusion and
    weighting can then be checked against a topology we chose, rather than
    against whatever the current corpus extraction happens to produce.

        alice ── bob            (co-authors, both at Duke, both RPE + receptors)
        carol ── bob            (co-author of bob only; RPE + parkinsons)
        dave                    (no co-authors; receptors only, at Cambridge)
        erin                    (no co-authors; parkinsons only, at Cambridge)
        frank                   (isolated; unrelated topic)

    So alice's best *new* connection should be carol: not already a co-author,
    shares the reward-prediction-error interest, and is bridged by bob.
    """
    from crew.graph.schema import Evidence, evidence
    from crew.graph.store import ScientistGraph

    def person(pid, name, topics, institutions, papers, corpus=True, degree=0):
        return {
            "id": pid, "name": name, "aliases": [name], "institutions": institutions,
            "topics": topics, "corpus_author": corpus, "papers": papers,
            "cited_works": len(papers), "degree": degree,
            "evidence": [evidence("paper.pdf", 1, f"{name} appears here")],
        }

    scientists = {
        "alice|a": person("alice|a", "Alice A", ["rpe", "receptors"], ["duke"],
                          ["p1"], degree=1),
        "bob|b": person("bob|b", "Bob B", ["rpe", "receptors"], ["duke"],
                        ["p1", "p2"], degree=2),
        "carol|c": person("carol|c", "Carol C", ["rpe", "parkinsons"], ["mit"],
                          ["p2"], degree=1),
        "dave|d": person("dave|d", "Dave D", ["receptors"], ["cambridge"], ["p3"]),
        "erin|e": person("erin|e", "Erin E", ["parkinsons"], ["duke"], ["p4"]),
        "frank|f": person("frank|f", "Frank F", ["fasting"], [], ["p5"]),
    }
    topics = {
        # `receptors` is deliberately the most common interest, so IDF
        # weighting should make it count for less than `rpe`.
        "rpe": {"id": "rpe", "label": "Reward prediction error", "description": "RPE",
                "aliases": ["RPE"], "scientists": ["alice|a", "bob|b", "carol|c"],
                "papers": []},
        "receptors": {"id": "receptors", "label": "Dopamine receptors",
                      "description": "Receptors", "aliases": ["D2"],
                      "scientists": ["alice|a", "bob|b", "dave|d"], "papers": []},
        "parkinsons": {"id": "parkinsons", "label": "Parkinson's disease",
                       "description": "PD", "aliases": ["PD"],
                       "scientists": ["carol|c", "erin|e"], "papers": []},
        "fasting": {"id": "fasting", "label": "Dopamine fasting",
                    "description": "Fasting", "aliases": [],
                    "scientists": ["frank|f"], "papers": []},
    }
    institutions = {
        "duke": {"id": "duke", "name": "Duke University", "departments": [],
                 "country": "USA", "scientists": ["alice|a", "bob|b", "erin|e"],
                 "evidence": []},
        "mit": {"id": "mit", "name": "MIT", "departments": [], "country": "USA",
                "scientists": ["carol|c"], "evidence": []},
        "cambridge": {"id": "cambridge", "name": "University of Cambridge",
                      "departments": [], "country": "UK",
                      "scientists": ["dave|d"], "evidence": []},
    }
    papers = {
        pid: {"id": pid, "title": f"Paper {pid}", "year": 2020, "venue": "Journal",
              "doi": "", "authors": [], "topics": [], "in_corpus": pid == "p1",
              "file": "paper.pdf" if pid == "p1" else "", "mentions": 1,
              "evidence": [evidence("paper.pdf", 1, f"Paper {pid}")]}
        for pid in ("p1", "p2", "p3", "p4", "p5")
    }
    papers["p1"]["authors"] = ["alice|a", "bob|b"]
    papers["p2"]["authors"] = ["bob|b", "carol|c"]

    def coauthor(a, b):
        ev: Evidence = evidence("paper.pdf", 1, "co-authored")
        return [
            {"source": a, "target": b, "relation": "CO_AUTHORED", "weight": 1.0,
             "evidence": [ev]},
            {"source": b, "target": a, "relation": "CO_AUTHORED", "weight": 1.0,
             "evidence": [ev]},
        ]

    edges = coauthor("alice|a", "bob|b") + coauthor("bob|b", "carol|c")
    for sid, s in scientists.items():
        for tid in s["topics"]:
            edges.append({"source": sid, "target": tid, "relation": "WORKS_ON",
                          "weight": 1.0, "evidence": []})

    stats = {
        "scientists": len(scientists), "corpus_authors": len(scientists),
        "institutions": len(institutions), "topics": len(topics),
        "papers": len(papers), "corpus_papers": 1, "edges": len(edges),
        "relations": {}, "built_from": ["paper.pdf"], "corpus_hash": "test",
    }
    return ScientistGraph(scientists, institutions, topics, papers, edges, stats)


@pytest.fixture
def graph():
    """The real built graph; skip cleanly if it hasn't been built yet."""
    from crew.graph.store import ScientistGraph

    try:
        return ScientistGraph.load()
    except Exception as exc:
        pytest.skip(f"Graph not available ({exc}); run `python -m crew.graph.build`")
