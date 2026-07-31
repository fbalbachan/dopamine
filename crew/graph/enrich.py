"""Optional enrichment of graph profiles from an external scholarly source.

Why not Google Scholar
----------------------
Google Scholar has **no public API**, and its terms of service prohibit
automated access; it defends that with CAPTCHAs and IP blocks after a handful of
programmatic requests. Libraries like ``scholarly`` work by scraping and need
rotating proxies to survive, and paid scrapers (SerpApi and friends) are simply
someone else absorbing that risk. Building the feature on it would mean a
pipeline that is legally grey, silently rate-limited, and broken by the next
markup change.

**OpenAlex** is used instead. It is the open successor to Microsoft Academic
Graph, covers ~250M works, requires no API key, allows 100k requests/day, and
publishes its data as CC0. For this feature it is also a strictly better source
than Scholar: it exposes exactly the fields the graph wants — canonical author
ids, **ORCID**, last-known institution with a ROR id, concept/topic tags, works
and citation counts, and an h-index.

ORCID is the part that matters most. :mod:`crew.graph.canon` resolves people by
surname + first initial, which cannot distinguish two ``Wang D.``; an ORCID is
an authoritative person identifier, so enrichment is also the upgrade path for
the graph's biggest known weakness.

This pass is **on by default** (``settings.graph_enrich``); disable it with
``build --no-enrich`` or ``GRAPH_ENRICH=0``. Turning it on by default is safe
precisely because it is additive and cannot fail loudly: every failure — no
network, rate limit, ambiguous name, no match — degrades to "leave this profile
as it was", never to an exception, matching how ``crew.tools.search_web``
handles a missing Brave key. A graph built with no network is exactly the graph
that was built before enrichment existed.

Two guards keep the additions honest:

* **Surname agreement.** OpenAlex's ``search`` is happy to return a confident
  near-miss, so :func:`_search_author` rejects every candidate whose surname
  disagrees with the query rather than taking ``results[0]``.
* **Scope.** Corpus authors only by default — the ~36 people the PDFs print
  affiliations for. ``--enrich-all`` covers cited-only researchers too, which
  is the documented fix for their missing institutions, at the cost of
  thousands of requests.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import requests

from .canon import canonical_institution, parse_name

OPENALEX_AUTHORS = "https://api.openalex.org/authors"

# OpenAlex asks callers to identify themselves; doing so moves you into the
# faster, more reliable "polite pool".
_MAILTO = "rootstrap-dopamine-graph@example.com"

# The API allows ~10 requests/second. Stay well under it.
_DELAY_SECONDS = 0.15


def _search_author(name: str, timeout: int = 15) -> Optional[dict]:
    """Best OpenAlex match for a name, or None."""
    try:
        resp = requests.get(
            OPENALEX_AUTHORS,
            params={"search": name, "per-page": 5, "mailto": _MAILTO},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", []) or []
    except Exception:
        return None
    if not results:
        return None

    # Guard against confident-but-wrong matches: require the surname to agree.
    target = parse_name(name)
    if target:
        surname = target.surname.lower()
        for candidate in results:
            got = parse_name(candidate.get("display_name", ""))
            if got and got.surname.lower() == surname:
                return candidate
        return None
    return results[0]


def _profile_from(record: dict) -> dict:
    """Map an OpenAlex author record onto the fields the graph stores."""
    institutions = record.get("last_known_institutions") or []
    if not institutions and record.get("last_known_institution"):
        institutions = [record["last_known_institution"]]
    stats = record.get("summary_stats") or {}
    topics = [
        t.get("display_name", "")
        for t in (record.get("topics") or record.get("x_concepts") or [])[:8]
        if t.get("display_name")
    ]
    return {
        "source": "openalex",
        "openalex_id": record.get("id", ""),
        "orcid": record.get("orcid") or "",
        "display_name": record.get("display_name", ""),
        "works_count": int(record.get("works_count") or 0),
        "cited_by_count": int(record.get("cited_by_count") or 0),
        "h_index": int(stats.get("h_index") or 0),
        "institutions": [
            {
                "name": i.get("display_name", ""),
                "ror": i.get("ror", ""),
                "country": i.get("country_code", "") or "",
            }
            for i in institutions
            if i.get("display_name")
        ],
        "topics": topics,
    }


def enrich_scientist(graph, sid: str, refresh: bool = False) -> bool:
    """Fetch and attach one scientist's OpenAlex profile. Returns True if written.

    The single-person unit exists so enrichment can be **triggered on demand** —
    ``graph.interface --enrich "Wolfram Schultz"`` or ``/enrich <name>`` in the
    REPL — rather than only as a build step. One name is one request, so a user
    who spots a stale or missing profile can fix that profile alone.
    """
    scientist = graph.scientists.get(sid)
    if scientist is None:
        return False
    if scientist.get("external") and not refresh:
        return False

    record = _search_author(scientist["name"])
    if not record:
        return False

    profile = _profile_from(record)
    scientist["external"] = profile

    # An ORCID-backed institution is better evidence than a PDF header, so fold
    # it into the graph proper rather than leaving it in `external`.
    for inst in profile["institutions"][:1]:
        iid = canonical_institution(inst["name"])
        if not iid:
            continue
        ev = _openalex_evidence(scientist["name"], inst["name"], profile)
        if iid not in graph.institutions:
            graph.institutions[iid] = {
                "id": iid, "name": inst["name"], "departments": [],
                "country": inst.get("country", ""), "scientists": [],
                "evidence": [ev],
            }
        if sid not in graph.institutions[iid]["scientists"]:
            graph.institutions[iid]["scientists"].append(sid)
        if iid not in scientist["institutions"]:
            scientist["institutions"].append(iid)
        # The node lists and the edge list must agree. Without this the
        # institution exists but nothing points at it: `graph.nx` shows an
        # isolated node and the viewer draws an unconnected hub, because it
        # takes nodes from `scientist["institutions"]` and lines from `edges`.
        _ensure_edge(graph, sid, iid, "AFFILIATED_WITH", ev)
    return True


def _openalex_evidence(person: str, institution: str, profile: dict) -> dict:
    """Provenance for a fact that came from OpenAlex rather than a PDF page.

    ``file`` names the source instead of a filename and ``page`` is 0 — the
    convention :mod:`crew.graph.chat` already uses for graph-derived evidence,
    so a consumer can tell a corpus observation from an external one without a
    new field.
    """
    orcid = profile.get("orcid") or ""
    return {
        "file": "openalex",
        "page": 0,
        "snippet": (
            f"OpenAlex lists {person} at {institution}"
            f"{' (ORCID ' + orcid.replace('https://orcid.org/', '') + ')' if orcid else ''}."
        ),
    }


def _ensure_edge(graph, source: str, target: str, relation: str, ev: dict) -> None:
    """Append an edge unless an identical (source, target, relation) exists.

    The seen-set is built once and cached on the graph: a linear scan per call
    is fine for 36 corpus authors but quadratic over the 3,600 that
    ``--enrich-all`` covers, against an edge list of ~44k.
    """
    index = graph._edge_index
    if index is None:
        index = {(e["source"], e["target"], e["relation"]) for e in graph.edges}
        graph._edge_index = index
    key = (source, target, relation)
    if key in index:
        return
    index.add(key)
    graph.edges.append({
        "source": source, "target": target, "relation": relation,
        "weight": 1.0, "evidence": [ev],
    })


def enrich_graph(graph, corpus_only: bool = True, limit: Optional[int] = None,
                 refresh: bool = False,
                 log: Optional[Callable[[str], None]] = None) -> int:
    """Attach OpenAlex profiles to scientists in ``graph``. Returns the count added.

    Defaults to corpus authors only: they are the ~40 people with real profiles,
    and querying every cited author would mean thousands of requests to decorate
    nodes that exist mainly as network structure.

    Resumable by default: anyone who already carries an ``external`` profile is
    skipped, so an interrupted pass can simply be run again and only pays for
    what is still missing. ``refresh=True`` re-fetches everyone instead — that
    is the "the data has moved on" path, and the only one that costs a full
    round of requests.
    """
    def say(msg: str) -> None:
        if log:
            log(msg)

    targets: List[str] = [
        sid for sid, s in graph.scientists.items()
        if s["corpus_author"] or not corpus_only
    ]
    targets.sort(key=lambda sid: not graph.scientists[sid]["corpus_author"])
    if not refresh:
        targets = [sid for sid in targets if not graph.scientists[sid].get("external")]
    if limit:
        targets = targets[:limit]

    if not targets:
        say("  every profile in scope is already enriched (use refresh to re-fetch)")
        return 0

    added = 0
    for n, sid in enumerate(targets, start=1):
        if enrich_scientist(graph, sid, refresh=refresh):
            added += 1
        time.sleep(_DELAY_SECONDS)
        if n % 10 == 0:
            say(f"  enriched {n}/{len(targets)}…")

    say(f"  {added}/{len(targets)} profiles matched on OpenAlex")
    return added


def external_summary(scientist: dict) -> str:
    """One-line rendering of an enriched profile, for CLI and viewer use."""
    ext: Dict = scientist.get("external") or {}
    if not ext:
        return ""
    bits = []
    if ext.get("works_count"):
        bits.append(f"{ext['works_count']} works")
    if ext.get("cited_by_count"):
        bits.append(f"{ext['cited_by_count']:,} citations")
    if ext.get("h_index"):
        bits.append(f"h-index {ext['h_index']}")
    if ext.get("orcid"):
        bits.append(ext["orcid"].replace("https://orcid.org/", "ORCID "))
    return " · ".join(bits)
