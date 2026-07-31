"""The exported viewer.

The viewer is one big template string, so a broken brace or an unbalanced tag
fails silently as a blank page rather than as an exception. These assert the
structural properties that would otherwise only be caught by opening it.
"""
from __future__ import annotations

import json
import re

from crew.graph.viz import NODE_KINDS, build_view, export_html, render_fragment


def _payload(html: str) -> dict:
    match = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert match, "viewer payload not found"
    return json.loads(match.group(1))


def test_view_contains_every_scientist_and_a_deterministic_layout(toy_graph):
    view = build_view(toy_graph, max_nodes=50)
    kinds = {n["kind"] for n in view["nodes"]}
    assert kinds <= set(NODE_KINDS)
    assert view["total_scientists"] == len(toy_graph.scientists)
    for node in view["nodes"]:
        assert 0.0 <= node["x"] <= 1.0
        assert 0.0 <= node["y"] <= 1.0


def test_layout_is_reproducible(toy_graph):
    """A fixed seed means reopening the file does not reshuffle the map."""
    first = {n["id"]: (n["x"], n["y"]) for n in build_view(toy_graph)["nodes"]}
    second = {n["id"]: (n["x"], n["y"]) for n in build_view(toy_graph)["nodes"]}
    assert first == second


def test_symmetric_coauthorship_is_drawn_once(toy_graph):
    """CO_AUTHORED is stored both ways; drawing both doubles every line."""
    view = build_view(toy_graph)
    pairs = [frozenset((e["s"], e["t"])) for e in view["edges"]
             if e["r"] == "CO_AUTHORED"]
    assert len(pairs) == len(set(pairs))


def test_recommendations_ship_with_their_signal_breakdown(toy_graph):
    """The card renders four small multiples; without `sig` they read as zero."""
    view = build_view(toy_graph)
    recs = [r for n in view["nodes"] for r in n.get("recs", [])]
    assert recs, "expected some recommendations in the toy graph"
    for rec in recs:
        assert set(rec["sig"]) == {"topics", "profile", "network", "citation"}
        assert all(0.0 <= v <= 1.0 for v in rec["sig"].values())


def test_max_nodes_is_respected(toy_graph):
    scientists = [n for n in build_view(toy_graph, max_nodes=2)["nodes"]
                  if n["kind"] in ("scientist", "corpus_author")]
    assert len(scientists) <= 2


def test_fragment_is_embeddable_and_balanced(toy_graph):
    """Published as an Artifact, the fragment is wrapped in a page skeleton —
    so it must not bring its own document tags."""
    html = render_fragment(build_view(toy_graph))
    lowered = html.lower()
    for tag in ("<!doctype", "<html", "<head>", "<body"):
        assert tag not in lowered
    assert html.count("<script>") == html.count("</script>") == 1
    assert html.count("<style>") == html.count("</style>") == 1


def test_both_theme_overrides_are_present(toy_graph):
    """The viewer's toggle has to beat prefers-color-scheme in both directions."""
    html = render_fragment(build_view(toy_graph))
    assert ':root[data-theme="dark"]' in html
    assert ':root[data-theme="light"]' in html
    assert "@media (prefers-color-scheme: dark)" in html


def test_script_terminator_in_data_cannot_break_out(toy_graph):
    """A paper title containing </script> would otherwise close the tag early."""
    toy_graph.papers["p1"]["title"] = "Attack </script><script>alert(1)</script>"
    html = render_fragment(build_view(toy_graph))
    # Exactly one real script element: the payload's angle brackets are escaped
    # to \u003c, which is inert inside a JSON string literal.
    assert html.count("<script>") == 1
    assert html.count("</script>") == 1
    assert "\\u003c/script" in html


def test_export_writes_a_standalone_document(toy_graph, tmp_path):
    path = export_html(toy_graph, path=tmp_path / "graph.html")
    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert _payload(html)["nodes"]
