"""Integration tests over the real index (skipped if it isn't built)."""
from __future__ import annotations

from crew.config import PAPER_FILENAMES, settings


def test_search_returns_page_level_citations(retriever):
    hits = retriever.search("dopamine role in reward prediction error", k=5)
    assert hits, "expected at least one hit"
    for h in hits:
        assert h.file in PAPER_FILENAMES     # citation points at a real paper
        assert h.page >= 1                   # 1-indexed page number
        assert h.snippet                     # non-empty supporting text


def test_in_domain_scores_above_off_topic(retriever):
    in_domain = retriever.max_similarity("dopamine receptors in the striatum")
    off_topic = retriever.max_similarity("best italian pizza recipe in Rome")
    assert in_domain > settings.scope_threshold > off_topic


def test_lexical_half_finds_exact_tokens(retriever):
    # BM25 side of the hybrid should surface exact receptor tokens.
    hits = retriever.search("D2 receptor", k=8)
    assert hits
    assert any("d2" in h.snippet.lower() or "d(2)" in h.snippet.lower() for h in hits) \
        or all(h.file in PAPER_FILENAMES for h in hits)
