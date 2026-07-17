"""The crew's tools + the evidence registry that makes citations trustworthy.

These are executed by the orchestrator on behalf of the Retriever / WebSearcher
agents (rather than through CrewAI's ReAct tool-executor, whose assistant
prefill is rejected by Claude 4.8). Driving them in Python has a second benefit:
every result is written into the shared :class:`EvidenceRegistry` under a stable
id (``P1``, ``P2`` … for paper pages; ``W1``, ``W2`` … for web hits), and the
Synthesizer may only cite those ids. Because ``file``/``page``/``url`` come from
the registry — never from the model's free text — a citation can't point at a
page or URL that retrieval didn't actually return.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import requests

from .config import settings
from .schema import PaperSource, WebSource


class EvidenceRegistry:
    """Collects the ground-truth evidence returned to the agents this turn."""

    def __init__(self) -> None:
        self.papers: Dict[str, PaperSource] = {}
        self.web: Dict[str, WebSource] = {}

    def add_paper(self, file: str, page: int, snippet: str) -> str:
        eid = f"P{len(self.papers) + 1}"
        self.papers[eid] = PaperSource(type="paper", file=file, page=page, snippet=snippet)
        return eid

    def add_web(self, url: str, title: str, snippet: str) -> str:
        eid = f"W{len(self.web) + 1}"
        self.web[eid] = WebSource(type="web", url=url, title=title, snippet=snippet)
        return eid

    def get(self, eid: str) -> Optional[dict]:
        return self.papers.get(eid) or self.web.get(eid)

    @property
    def any_web(self) -> bool:
        return bool(self.web)


def search_papers(retriever, registry: EvidenceRegistry, queries: List[str],
                  k: Optional[int] = None) -> str:
    """Hybrid (dense + BM25) search over the nine papers for one or more queries.

    Results are deduped by (file, page), registered as [P#], and returned as a
    text block for the Synthesizer's context.
    """
    seen = set()
    lines: List[str] = []
    for query in queries:
        for h in retriever.search(query, k=k):
            key = (h.file, h.page)
            if key in seen:
                continue
            seen.add(key)
            eid = registry.add_paper(h.file, h.page, h.snippet)
            lines.append(f"[{eid}] file={h.file} page={h.page}\n{h.snippet}")
    return "\n\n".join(lines) if lines else "No relevant passages found in the papers."


def search_web(registry: EvidenceRegistry, query: str, max_results: int = 5) -> str:
    """Live web search via the Brave Search API. Registers hits as [W#].

    Degrades gracefully: a missing key or an API error becomes a clean status
    string rather than an exception, so the crew still answers from the papers.
    """
    if not settings.brave_api_key:
        return "WEB_SEARCH_UNAVAILABLE: no Brave API key configured."
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": settings.brave_api_key,
            },
            params={"q": query, "count": max_results},
            timeout=20,
        )
        resp.raise_for_status()
        results = (resp.json().get("web", {}) or {}).get("results", []) or []
    except Exception as exc:  # network / auth / quota
        return f"WEB_SEARCH_ERROR: {exc}"

    if not results:
        return "No web results found."
    lines: List[str] = []
    for r in results[:max_results]:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("description", "") or ""
        eid = registry.add_web(url, title, snippet)
        lines.append(f"[{eid}] {title} — {url}\n{snippet}")
    return "\n\n".join(lines)
