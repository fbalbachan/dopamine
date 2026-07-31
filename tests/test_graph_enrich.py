"""OpenAlex enrichment: the guards that make it safe to run by default.

Enrichment is the one part of the graph that reaches the public internet, and
since it is now on by default (``settings.graph_enrich``) the interesting
assertions are about what it does when the network misbehaves and about what it
refuses to write. Nothing here makes a request: ``requests.get`` is patched in
every test that would otherwise reach OpenAlex.
"""
from __future__ import annotations

import pytest

from crew.graph import enrich as E


# --------------------------------------------------------------------------- #
# Fake OpenAlex responses
# --------------------------------------------------------------------------- #
def _record(display_name, orcid="", h_index=0, institution=None, works=0, cited=0):
    return {
        "id": f"https://openalex.org/A{abs(hash(display_name)) % 10**8}",
        "display_name": display_name,
        "orcid": orcid,
        "works_count": works,
        "cited_by_count": cited,
        "summary_stats": {"h_index": h_index},
        "last_known_institutions": (
            [{"display_name": institution, "ror": "https://ror.org/x",
              "country_code": "US"}] if institution else []
        ),
        "topics": [{"display_name": "Dopamine"}],
    }


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_search(monkeypatch, results):
    """Make every OpenAlex call return ``results``."""
    monkeypatch.setattr(
        E.requests, "get",
        lambda *a, **kw: _FakeResponse({"results": results}),
    )


def _patch_failure(monkeypatch, exc=ConnectionError("no network")):
    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(E.requests, "get", boom)


# --------------------------------------------------------------------------- #
# The surname guard
# --------------------------------------------------------------------------- #
def test_surname_must_agree_before_a_match_is_accepted(monkeypatch):
    """OpenAlex ranks by relevance, not identity — results[0] can be a stranger.

    Taking the top hit blindly would staple another researcher's h-index and
    ORCID onto a profile, which is worse than having no profile at all.
    """
    _patch_search(monkeypatch, [_record("Wolfgang Schmidt"), _record("Ann Other")])
    assert E._search_author("Wolfram Schultz") is None


def test_a_lower_ranked_candidate_wins_if_it_is_the_one_whose_surname_fits(monkeypatch):
    _patch_search(monkeypatch, [
        _record("Wolfgang Schmidt", h_index=200),
        _record("Wolfram Schultz", orcid="https://orcid.org/0000-0002-8530-4518",
                h_index=94),
    ])
    got = E._search_author("Wolfram Schultz")
    assert got is not None
    assert got["display_name"] == "Wolfram Schultz"


def test_no_results_is_not_an_error(monkeypatch):
    _patch_search(monkeypatch, [])
    assert E._search_author("Nobody At All") is None


# --------------------------------------------------------------------------- #
# Failure is always "leave the profile alone"
# --------------------------------------------------------------------------- #
def test_network_failure_degrades_rather_than_raising(monkeypatch):
    _patch_failure(monkeypatch)
    assert E._search_author("Wolfram Schultz") is None


def test_offline_enrichment_leaves_the_graph_untouched(monkeypatch, toy_graph):
    """A build with no network must produce exactly the pre-enrichment graph."""
    _patch_failure(monkeypatch)
    before = {sid: dict(s) for sid, s in toy_graph.scientists.items()}
    institutions_before = set(toy_graph.institutions)

    added = E.enrich_graph(toy_graph)

    assert added == 0
    assert set(toy_graph.institutions) == institutions_before
    for sid, s in toy_graph.scientists.items():
        assert "external" not in s
        assert s == before[sid]


# --------------------------------------------------------------------------- #
# What a successful pass writes
# --------------------------------------------------------------------------- #
def test_profile_fields_are_mapped_onto_the_graph(monkeypatch, toy_graph):
    _patch_search(monkeypatch, [
        _record("Alice A", orcid="https://orcid.org/0000-0001-0000-0001",
                h_index=42, institution="Duke University", works=120, cited=9000),
    ])
    monkeypatch.setattr(E.time, "sleep", lambda *_: None)

    added = E.enrich_graph(toy_graph, limit=1)

    assert added == 1
    ext = toy_graph.scientists["alice|a"]["external"]
    assert ext["source"] == "openalex"
    assert ext["h_index"] == 42
    assert ext["orcid"].endswith("0000-0001")
    assert ext["works_count"] == 120


def test_an_openalex_institution_joins_the_graph_proper(monkeypatch, toy_graph):
    """A ROR-backed affiliation is better evidence than a PDF header, so it is
    folded into `institutions` rather than left inside `external`."""
    monkeypatch.setattr(E.time, "sleep", lambda *_: None)
    _patch_search(monkeypatch, [
        _record("Frank F", h_index=5, institution="Karolinska Institute"),
    ])

    # frank is the only person in the toy graph with no institution at all.
    assert toy_graph.scientists["frank|f"]["institutions"] == []
    E.enrich_graph(toy_graph)

    iid = toy_graph.scientists["frank|f"]["institutions"][0]
    assert iid in toy_graph.institutions
    assert toy_graph.institutions[iid]["country"] == "US"
    assert "frank|f" in toy_graph.institutions[iid]["scientists"]


def test_enrichment_is_idempotent(monkeypatch, toy_graph):
    """Re-running must not duplicate rosters or institution entries."""
    monkeypatch.setattr(E.time, "sleep", lambda *_: None)
    _patch_search(monkeypatch, [
        _record("Alice A", h_index=42, institution="Duke University"),
    ])

    E.enrich_graph(toy_graph, refresh=True)
    institutions = {k: list(v["scientists"]) for k, v in toy_graph.institutions.items()}
    memberships = list(toy_graph.scientists["alice|a"]["institutions"])

    E.enrich_graph(toy_graph, refresh=True)

    assert {k: v["scientists"] for k, v in toy_graph.institutions.items()} == institutions
    assert toy_graph.scientists["alice|a"]["institutions"] == memberships


# --------------------------------------------------------------------------- #
# Resumability, and the on-demand trigger
# --------------------------------------------------------------------------- #
def test_a_second_pass_skips_people_who_already_have_a_profile(monkeypatch, toy_graph):
    """This is what makes the pass resumable — and cheap to re-run."""
    monkeypatch.setattr(E.time, "sleep", lambda *_: None)
    queried = []

    def spy(name, timeout=15):
        queried.append(name)
        return _record(name, h_index=1)

    monkeypatch.setattr(E, "_search_author", spy)

    first = E.enrich_graph(toy_graph)
    assert first > 0
    queried.clear()

    assert E.enrich_graph(toy_graph) == 0
    assert queried == []


def test_refresh_re_fetches_everyone(monkeypatch, toy_graph):
    monkeypatch.setattr(E.time, "sleep", lambda *_: None)
    calls = []
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: (calls.append(name), _record(name, h_index=7))[1],
    )

    E.enrich_graph(toy_graph)
    before = len(calls)
    E.enrich_graph(toy_graph, refresh=True)

    assert len(calls) == before * 2


def test_enrich_scientist_targets_exactly_one_person(monkeypatch, toy_graph):
    """The unit behind `--enrich <name>`: one name, one request."""
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: _record("Carol C", h_index=11),
    )

    assert E.enrich_scientist(toy_graph, "carol|c") is True
    assert toy_graph.scientists["carol|c"]["external"]["h_index"] == 11
    assert "external" not in toy_graph.scientists["alice|a"]


def test_enrich_scientist_declines_a_repeat_unless_asked_to_refresh(monkeypatch,
                                                                   toy_graph):
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: _record("Carol C", h_index=11),
    )
    assert E.enrich_scientist(toy_graph, "carol|c") is True
    assert E.enrich_scientist(toy_graph, "carol|c") is False
    assert E.enrich_scientist(toy_graph, "carol|c", refresh=True) is True


def test_enrich_scientist_ignores_an_unknown_id(toy_graph):
    assert E.enrich_scientist(toy_graph, "nobody|z") is False


# --------------------------------------------------------------------------- #
# Node lists and the edge list must not diverge
# --------------------------------------------------------------------------- #
def test_an_added_institution_also_gets_an_affiliation_edge(monkeypatch, toy_graph):
    """Otherwise the institution is a node nothing points at: isolated in
    `graph.nx`, and an unconnected hub in the viewer, which draws its nodes
    from `scientist["institutions"]` but its lines from `edges`."""
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: _record("Frank F", h_index=5,
                                         institution="Karolinska Institute"),
    )
    E.enrich_scientist(toy_graph, "frank|f")

    iid = toy_graph.scientists["frank|f"]["institutions"][0]
    edge = [e for e in toy_graph.edges
            if e["source"] == "frank|f" and e["target"] == iid]
    assert len(edge) == 1
    assert edge[0]["relation"] == "AFFILIATED_WITH"
    assert iid in toy_graph.nx
    assert toy_graph.nx.degree(iid) > 0


def test_the_affiliation_edge_carries_openalex_provenance(monkeypatch, toy_graph):
    """A fact from the web must not look like a fact read off a PDF page."""
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: _record(
            "Frank F", orcid="https://orcid.org/0000-0002-1111-2222",
            institution="Karolinska Institute"),
    )
    E.enrich_scientist(toy_graph, "frank|f")

    ev = [e for e in toy_graph.edges
          if e["relation"] == "AFFILIATED_WITH" and e["source"] == "frank|f"
          ][0]["evidence"][0]
    assert ev["file"] == "openalex"
    assert ev["page"] == 0
    assert "0000-0002-1111-2222" in ev["snippet"]


def test_affiliation_edges_are_not_duplicated_on_refresh(monkeypatch, toy_graph):
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: _record("Frank F", institution="Karolinska Institute"),
    )
    E.enrich_scientist(toy_graph, "frank|f")
    count = sum(1 for e in toy_graph.edges if e["relation"] == "AFFILIATED_WITH")

    E.enrich_scientist(toy_graph, "frank|f", refresh=True)

    assert sum(1 for e in toy_graph.edges
               if e["relation"] == "AFFILIATED_WITH") == count


def test_recount_stats_reflects_what_enrichment_added(monkeypatch, toy_graph):
    """`stats` is what the CLI, the viewer header and summary() all read, so a
    stale count is a wrong number on every surface."""
    monkeypatch.setattr(E.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: _record(name, h_index=3,
                                         institution="Karolinska Institute"),
    )
    edges_before = toy_graph.stats["edges"]

    E.enrich_graph(toy_graph)
    toy_graph.recount_stats()

    assert toy_graph.stats["edges"] == len(toy_graph.edges) > edges_before
    assert toy_graph.stats["institutions"] == len(toy_graph.institutions)
    assert toy_graph.stats["relations"]["AFFILIATED_WITH"] == sum(
        1 for e in toy_graph.edges if e["relation"] == "AFFILIATED_WITH"
    )
    assert toy_graph.stats["enriched"] == sum(
        1 for s in toy_graph.scientists.values() if s.get("external")
    )


def test_recount_stats_drops_caches_built_from_the_old_graph(toy_graph):
    """A cached projection built before enrichment would miss the new edges."""
    toy_graph.nx                       # force the caches to populate
    toy_graph.coauthor_graph
    assert toy_graph._nx is not None

    toy_graph.recount_stats()

    assert toy_graph._nx is None
    assert toy_graph._coauthor is None
    assert toy_graph._profile_matrix is None


def test_on_demand_trigger_persists_and_updates_stats(monkeypatch, toy_graph,
                                                      tmp_path, capsys):
    """`--enrich <name>` must write the graph back, not just mutate memory."""
    from crew.graph import interface as I
    from crew.graph.store import ScientistGraph

    target = tmp_path / "graph.json"
    monkeypatch.setattr(ScientistGraph, "save",
                        lambda self, path=None: _save_to(self, target))
    monkeypatch.setattr(
        E, "_search_author",
        lambda name, timeout=15: _record("Carol C", h_index=11,
                                         institution="Karolinska Institute"),
    )

    assert I.run_enrichment(toy_graph, "Carol C") is True

    reloaded = ScientistGraph.load(target)
    assert reloaded.scientists["carol|c"]["external"]["h_index"] == 11
    assert reloaded.stats["enriched"] == 1
    assert reloaded.stats["institutions"] == len(reloaded.institutions)


def test_on_demand_trigger_reports_a_miss_without_touching_the_graph(monkeypatch,
                                                                    toy_graph):
    from crew.graph import interface as I

    monkeypatch.setattr(E, "_search_author", lambda name, timeout=15: None)
    before = dict(toy_graph.scientists["carol|c"])

    assert I.run_enrichment(toy_graph, "Carol C") is False
    assert toy_graph.scientists["carol|c"] == before


def _save_to(graph, path):
    """ScientistGraph.save, redirected to a fixed path."""
    import json

    path.write_text(
        json.dumps({
            "stats": graph.stats,
            "scientists": list(graph.scientists.values()),
            "institutions": list(graph.institutions.values()),
            "topics": list(graph.topics.values()),
            "papers": list(graph.papers.values()),
            "edges": graph.edges,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return path


def test_corpus_only_is_the_default_scope(monkeypatch, toy_graph):
    """Every person in the toy graph is a corpus author except one we demote."""
    monkeypatch.setattr(E.time, "sleep", lambda *_: None)
    toy_graph.scientists["frank|f"]["corpus_author"] = False
    seen = []

    def spy(name, timeout=15):
        seen.append(name)
        return None

    monkeypatch.setattr(E, "_search_author", spy)
    E.enrich_graph(toy_graph, corpus_only=True)

    assert "Frank F" not in seen
    assert "Alice A" in seen


def test_external_summary_renders_only_the_fields_present():
    assert E.external_summary({}) == ""
    line = E.external_summary({"external": {
        "works_count": 330, "cited_by_count": 48141, "h_index": 94,
        "orcid": "https://orcid.org/0000-0002-8530-4518",
    }})
    assert "330 works" in line
    assert "48,141 citations" in line
    assert "h-index 94" in line
    assert "ORCID 0000-0002-8530-4518" in line


# --------------------------------------------------------------------------- #
# Build-time wiring
# --------------------------------------------------------------------------- #
def test_enrichment_is_on_by_default():
    """The whole point of this change: no flag needed."""
    from crew.config import settings

    assert settings.graph_enrich is True
    assert settings.graph_enrich_all is False


def test_stats_record_what_the_pass_covered(monkeypatch, toy_graph):
    from crew.graph.build import _apply_enrichment

    monkeypatch.setattr(E.time, "sleep", lambda *_: None)
    _patch_search(monkeypatch, [_record("Alice A", h_index=42)])

    _apply_enrichment(toy_graph, enrich_all=False)

    assert toy_graph.stats["enrich_scope"] == "corpus"
    assert toy_graph.stats["enriched"] >= 1
    # The count must reflect reality, not the pre-enrichment snapshot.
    assert toy_graph.stats["institutions"] == len(toy_graph.institutions)
