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

Everything here is **optional and off by default** (``build --enrich``). The
graph is fully functional offline; enrichment only adds fields. Every failure —
no network, rate limit, ambiguous name — degrades to "leave this profile as it
was", never to an exception, matching how ``crew.tools.search_web`` handles a
missing Brave key.
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


def enrich_graph(graph, corpus_only: bool = True, limit: Optional[int] = None,
                 log: Optional[Callable[[str], None]] = None) -> int:
    """Attach OpenAlex profiles to scientists in ``graph``. Returns the count added.

    Defaults to corpus authors only: they are the ~40 people with real profiles,
    and querying every cited author would mean thousands of requests to decorate
    nodes that exist mainly as network structure.
    """
    def say(msg: str) -> None:
        if log:
            log(msg)

    targets: List[str] = [
        sid for sid, s in graph.scientists.items()
        if s["corpus_author"] or not corpus_only
    ]
    targets.sort(key=lambda sid: not graph.scientists[sid]["corpus_author"])
    if limit:
        targets = targets[:limit]

    added = 0
    for n, sid in enumerate(targets, start=1):
        scientist = graph.scientists[sid]
        record = _search_author(scientist["name"])
        time.sleep(_DELAY_SECONDS)
        if not record:
            continue

        profile = _profile_from(record)
        scientist["external"] = profile
        added += 1

        # An ORCID-backed institution is better evidence than a PDF header, so
        # fold it into the graph proper rather than leaving it in `external`.
        for inst in profile["institutions"][:1]:
            iid = canonical_institution(inst["name"])
            if not iid:
                continue
            if iid not in graph.institutions:
                graph.institutions[iid] = {
                    "id": iid, "name": inst["name"], "departments": [],
                    "country": inst.get("country", ""), "scientists": [],
                    "evidence": [],
                }
            if sid not in graph.institutions[iid]["scientists"]:
                graph.institutions[iid]["scientists"].append(sid)
            if iid not in scientist["institutions"]:
                scientist["institutions"].append(iid)

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
