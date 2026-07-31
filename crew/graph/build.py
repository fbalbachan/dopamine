"""Build the scientist knowledge graph from the nine PDFs.

    python -m crew.graph.build               # full build (front matter + references)
    python -m crew.graph.build --no-refs     # front matter only — fast and free
    python -m crew.graph.build --force       # ignore the corpus hash and rebuild
    python -m crew.graph.build --no-enrich   # skip OpenAlex — fully offline
    python -m crew.graph.build --enrich-all  # enrich cited researchers too (slow)

Pipeline
--------
1. **Extract** front matter and reference lists per paper (:mod:`.extract`).
2. **Resolve** every surface form to a canonical id (:mod:`.canon`).
3. **Assign** research interests from a closed taxonomy (:mod:`.topics`).
4. **Enrich** profiles from OpenAlex — ORCID, h-index, citation counts,
   ROR-backed affiliations (:mod:`.enrich`). On by default; no API key needed.
5. **Assemble** typed nodes and evidenced edges, then persist to
   ``crew/.graph/graph.json`` and render ``graph.html``.

Like ``crew.build_index``, the build is idempotent: a ``corpus_hash`` over the
PDF names and sizes plus the extraction settings short-circuits a rebuild when
nothing has changed, so it is safe to run on every deploy.

Enrichment sits *outside* that hash. It decorates nodes rather than creating
them, so turning it on must not cost a full re-extraction: when the graph is
already current but unenriched, the builder runs the OpenAlex pass over the
loaded graph and re-saves. That also means an enriched graph is never
re-enriched on a no-op build — use ``--force`` to refresh the profiles.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import PAPER_FILENAMES, settings
from .canon import (
    best_display,
    canonical_institution,
    parse_name,
    slug,
    split_author_list,
)
from .embedding import default_embedder
from .extract import extract_front_matter, extract_references, load_pages
from .schema import (
    Edge,
    Evidence,
    GraphStats,
    Institution,
    Paper,
    Scientist,
    Topic,
    evidence,
)
from .store import GRAPH_FILENAME, ScientistGraph
from .topics import Taxonomy, derive_taxonomy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _log(msg: str) -> None:
    print(msg, flush=True)


def _corpus_hash(paths: List[Path], with_refs: bool) -> str:
    h = sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    h.update(settings.graph_extract_model.encode())
    h.update(settings.embed_model.encode())
    h.update(f"refs={with_refs}|min={settings.graph_min_mentions}".encode())
    return h.hexdigest()


def _work_id(authors: List[str], title: str, year: Optional[int]) -> str:
    """Stable id for a cited work: first author + year + title stem.

    Titles are the discriminating field, but plenty of extracted references have
    none, so the key degrades to author+year rather than collapsing every
    untitled work by one author into a single node.
    """
    first = ""
    for raw in authors:
        parsed = parse_name(raw)
        if parsed:
            first = parsed.key.split("|")[0]
            break
    stem = slug(" ".join((title or "").split()[:6]))
    if stem:
        return f"w:{first}-{year or 'na'}-{stem}"[:120]
    tail = slug("-".join(sorted(parse_name(a).key for a in authors if parse_name(a))[:3]))
    return f"w:{first}-{year or 'na'}-{tail or 'x'}"[:120]


class _Builder:
    """Accumulates nodes and edges, then freezes them into a ScientistGraph."""

    def __init__(self, taxonomy: Taxonomy, embed):
        self.taxonomy = taxonomy
        self.embed = embed
        self.scientists: Dict[str, Scientist] = {}
        self.institutions: Dict[str, Institution] = {}
        self.papers: Dict[str, Paper] = {}
        self._name_forms: Dict[str, List[str]] = defaultdict(list)
        # Richer text to topic a paper by than its title alone (abstracts for
        # corpus papers). Titles are often too short to discriminate:
        # "Dopamine in Parkinson's disease" embeds close to every "Dopamine in
        # X" topic description.
        self.topic_text: Dict[str, str] = {}
        self._edges: Dict[Tuple[str, str, str], Edge] = {}
        self._topic_scores: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    # -- node helpers --------------------------------------------------------
    def scientist(self, raw_name: str, ev: Evidence, corpus_author: bool = False
                  ) -> Optional[str]:
        parsed = parse_name(raw_name)
        if not parsed:
            return None
        sid = parsed.key
        if not sid or sid.startswith("|"):
            return None
        self._name_forms[sid].append(raw_name)

        node = self.scientists.get(sid)
        if node is None:
            node = Scientist(
                id=sid, name=parsed.display, aliases=[], institutions=[], topics=[],
                corpus_author=corpus_author, papers=[], cited_works=0, degree=0,
                evidence=[],
            )
            self.scientists[sid] = node
        if corpus_author:
            node["corpus_author"] = True
        if len(node["evidence"]) < 5 and ev not in node["evidence"]:
            node["evidence"].append(ev)
        return sid

    def institution(self, organization: str, department: str, country: str,
                    ev: Evidence) -> Optional[str]:
        iid = canonical_institution(organization)
        if not iid:
            return None
        node = self.institutions.get(iid)
        if node is None:
            node = Institution(
                id=iid, name=" ".join(organization.split()), departments=[],
                country=country or "", scientists=[], evidence=[],
            )
            self.institutions[iid] = node
        if department and department not in node["departments"]:
            node["departments"].append(department)
        if country and not node["country"]:
            node["country"] = country
        if len(node["evidence"]) < 3:
            node["evidence"].append(ev)
        return iid

    def paper(self, pid: str, title: str, year, venue: str, doi: str,
              in_corpus: bool, file: str, ev: Evidence) -> str:
        node = self.papers.get(pid)
        if node is None:
            node = Paper(
                id=pid, title=" ".join((title or "").split()), year=year,
                venue=venue or "", doi=doi or "", authors=[], topics=[],
                in_corpus=in_corpus, file=file or "", mentions=0, evidence=[],
            )
            self.papers[pid] = node
        node["mentions"] += 1
        if title and len(title) > len(node["title"]):
            node["title"] = " ".join(title.split())
        if year and not node["year"]:
            node["year"] = year
        if len(node["evidence"]) < 3:
            node["evidence"].append(ev)
        return pid

    # -- edge helper ---------------------------------------------------------
    def edge(self, source: str, target: str, relation: str, ev: Optional[Evidence],
             weight: float = 1.0) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, relation)
        existing = self._edges.get(key)
        if existing is None:
            self._edges[key] = Edge(
                source=source, target=target, relation=relation,
                weight=weight, evidence=[ev] if ev else [],
            )
            return
        existing["weight"] += weight
        if ev and len(existing["evidence"]) < 3:
            existing["evidence"].append(ev)

    # -- topic accumulation --------------------------------------------------
    def note_topics(self, pid: str, assignments: List[Tuple[str, float]]) -> None:
        paper = self.papers.get(pid)
        if not paper:
            return
        paper["topics"] = [t for t, _ in assignments]
        for tid, score in assignments:
            self.edge(pid, tid, "COVERS", None, weight=score)
            for sid in paper["authors"]:
                self._topic_scores[sid][tid] += score

    # -- freeze --------------------------------------------------------------
    def finish(self, taxonomy: Taxonomy, built_from: List[str], corpus_hash: str
               ) -> ScientistGraph:
        # Best display name + alias list per scientist.
        for sid, forms in self._name_forms.items():
            node = self.scientists[sid]
            node["name"] = best_display(forms) or node["name"]
            node["aliases"] = sorted({" ".join(f.split()) for f in forms})[:12]

        # Scientist -> topic edges, strongest interests first.
        for sid, scores in self._topic_scores.items():
            if sid not in self.scientists:
                continue
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:6]
            self.scientists[sid]["topics"] = [t for t, _ in ranked]
            for tid, score in ranked:
                self.edge(sid, tid, "WORKS_ON", None, weight=round(score, 4))

        # Topic nodes, populated from what actually landed on them.
        topics: Dict[str, Topic] = {}
        for t in taxonomy.topics:
            topics[t["id"]] = Topic(
                id=t["id"], label=t["label"], description=t["description"],
                aliases=t["aliases"], scientists=[], papers=[],
            )
        for sid, s in self.scientists.items():
            for tid in s["topics"]:
                if tid in topics:
                    topics[tid]["scientists"].append(sid)
        for pid, p in self.papers.items():
            for tid in p["topics"]:
                if tid in topics:
                    topics[tid]["papers"].append(pid)

        # Institution rosters.
        for sid, s in self.scientists.items():
            for iid in s["institutions"]:
                if iid in self.institutions and sid not in self.institutions[iid]["scientists"]:
                    self.institutions[iid]["scientists"].append(sid)

        # Co-authorship degree.
        degree: Dict[str, set] = defaultdict(set)
        for (src, tgt, rel) in self._edges:
            if rel == "CO_AUTHORED":
                degree[src].add(tgt)
        for sid, s in self.scientists.items():
            s["degree"] = len(degree.get(sid, ()))

        edges = list(self._edges.values())
        relations: Dict[str, int] = defaultdict(int)
        for e in edges:
            relations[e["relation"]] += 1

        stats = GraphStats(
            scientists=len(self.scientists),
            corpus_authors=sum(1 for s in self.scientists.values() if s["corpus_author"]),
            institutions=len(self.institutions),
            topics=len(topics),
            papers=len(self.papers),
            corpus_papers=sum(1 for p in self.papers.values() if p["in_corpus"]),
            edges=len(edges),
            relations=dict(relations),
            built_from=built_from,
            corpus_hash=corpus_hash,
        )
        return ScientistGraph(
            self.scientists, self.institutions, topics, self.papers, edges, stats
        )


def _apply_enrichment(graph: ScientistGraph, enrich_all: bool,
                      refresh: bool = False) -> int:
    """Run the OpenAlex pass and record what it covered in ``stats``."""
    from .enrich import enrich_graph

    scope = "every scientist" if enrich_all else "corpus authors"
    _log(f"Enriching from OpenAlex ({scope})…")
    added = enrich_graph(graph, corpus_only=not enrich_all, refresh=refresh,
                         log=_log)
    # Enrichment folds in ROR-backed institutions the PDFs never printed, plus
    # the AFFILIATED_WITH edges that reach them, so every derived count moves.
    graph.recount_stats()
    graph.stats["enrich_scope"] = "all" if enrich_all else "corpus"
    return added


def build_graph(force: bool = False, with_refs: bool = True,
                enrich: Optional[bool] = None, enrich_all: Optional[bool] = None,
                refresh_enrich: bool = False,
                export_html: bool = True) -> ScientistGraph:
    """Run the full pipeline and persist the graph.

    ``enrich``/``enrich_all`` default to ``settings.graph_enrich`` /
    ``settings.graph_enrich_all`` (env-configurable) when left as ``None``.
    ``refresh_enrich`` re-fetches profiles that are already present — the
    "OpenAlex has newer numbers" path, which the idempotency check would
    otherwise skip.
    """
    from ..rag import download_papers

    enrich = settings.graph_enrich if enrich is None else enrich
    enrich_all = settings.graph_enrich_all if enrich_all is None else enrich_all

    settings.ensure_dirs()
    paths = download_papers()
    graph_path = settings.graph_dir / GRAPH_FILENAME
    corpus_hash = _corpus_hash(paths, with_refs)

    if not force and graph_path.exists():
        current = False
        try:
            existing = json.loads(graph_path.read_text(encoding="utf-8"))
            current = existing.get("stats", {}).get("corpus_hash") == corpus_hash
        except Exception:
            current = False
        if current:
            graph = ScientistGraph.load(graph_path)
            # Structure is unchanged, but enrichment is not part of the hash —
            # so an enrichment that has never run still has work to do, and an
            # explicit --refresh-enrich always does.
            if enrich and (refresh_enrich or not graph.stats.get("enriched")):
                _log(f"Graph structure is current ({graph_path}); "
                     f"{'refreshing' if refresh_enrich else 'adding'} "
                     f"OpenAlex profiles.")
                _apply_enrichment(graph, enrich_all, refresh=refresh_enrich)
                graph.save()
                _log(f"  {graph.summary()}")
                if export_html:
                    _export_viewer(graph)
            else:
                _log(f"Graph is current ({graph_path}). Use --force to rebuild.")
                # The viewer is a derived artifact: an enrichment pass on an
                # earlier run updates graph.json and leaves graph.html behind.
                # Regenerate when it is missing or older than its source.
                if export_html and _viewer_is_stale(graph_path):
                    _log("  viewer is out of date — re-rendering.")
                    _export_viewer(graph)
            return graph

    if not settings.anthropic_api_key:
        _log("! No ANTHROPIC_API_KEY — falling back to heuristic extraction.")
        _log("  Author networks still build; titles and affiliations will be sparse.")

    _log(f"Loading {len(paths)} PDFs…")
    pages_by_file: Dict[str, List[str]] = {}
    for path in paths:
        pages_by_file[path.name] = load_pages(path)

    # 1. Front matter — the rich profiles.
    _log("Extracting front matter…")
    metas = {}
    for path in paths:
        _log(f"  {path.name}")
        metas[path.name] = extract_front_matter(
            pages_by_file[path.name], path.name, warn=_log
        )

    # 2. Taxonomy — derived once from what the corpus is about, then cached.
    # Topic ids are node ids: re-deriving on every build would silently
    # re-partition everyone's research interests, so an existing taxonomy is
    # reused unless --force explicitly asks for a fresh one.
    taxonomy_path = settings.graph_dir / "taxonomy.json"
    if taxonomy_path.exists() and not force:
        taxonomy = Taxonomy.load(taxonomy_path)
        _log(f"Reusing cached taxonomy ({len(taxonomy)} topics)")
    else:
        _log("Deriving topic taxonomy…")
        summaries = [
            f"{m.title}\n{m.abstract}" if m.title
            else " ".join(pages_by_file[f][0].split())[:1200]
            for f, m in metas.items()
        ]
        taxonomy = derive_taxonomy(summaries)
        taxonomy.save(taxonomy_path)
        _log(f"  {len(taxonomy)} topics")

    embed = default_embedder()
    builder = _Builder(taxonomy, embed)

    # 3. Corpus papers, their authors and institutions.
    _log("Building corpus author profiles…")
    for path in paths:
        filename = path.name
        meta = metas[filename]
        pages = pages_by_file[filename]
        ev = evidence(filename, 1, meta.title or (pages[0] if pages else ""))
        pid = f"c:{filename}"
        builder.paper(
            pid, meta.title or filename, meta.year, meta.venue, meta.doi,
            in_corpus=True, file=filename, ev=ev,
        )
        builder.topic_text[pid] = " ".join(
            f"{meta.title}. {meta.abstract}".split()
        )[:1500] or filename
        for author in meta.authors:
            sid = builder.scientist(author.name, ev, corpus_author=True)
            if not sid:
                continue
            node = builder.scientists[sid]
            if pid not in node["papers"]:
                node["papers"].append(pid)
            # Two printed forms of one person ("J. Patel" and "Jitendra Patel")
            # resolve to the same id; the author list must not list them twice.
            if sid not in builder.papers[pid]["authors"]:
                builder.papers[pid]["authors"].append(sid)
            builder.edge(sid, pid, "AUTHORED", ev)
            if author.organization:
                iid = builder.institution(
                    author.organization, author.department, author.country, ev
                )
                if iid:
                    if iid not in node["institutions"]:
                        node["institutions"].append(iid)
                    builder.edge(sid, iid, "AFFILIATED_WITH", ev)
        _link_coauthors(builder, builder.papers[pid]["authors"], ev)

    # A corpus paper with no authors means front-matter extraction failed, and
    # the graph will build anyway — minus that paper's whole author team, their
    # affiliations and their interests. Say so rather than letting it pass.
    empty = [p["file"] for p in builder.papers.values()
             if p["in_corpus"] and not p["authors"]]
    if empty:
        _log(f"! {len(empty)} paper(s) yielded no authors — profiles will be "
             f"missing for them:")
        for name in empty:
            _log(f"    {name}")
        _log("  Re-run with --force to retry the extraction.")

    # 4. Reference lists — the network.
    if with_refs:
        _log("Extracting reference lists…")
        for path in paths:
            filename = path.name
            _log(f"  {filename}")
            works = extract_references(pages_by_file[filename], filename, progress=_log)
            _log(f"    {len(works)} references")
            citing_pid = f"c:{filename}"
            for work in works:
                _add_cited_work(builder, work, filename, citing_pid)

    # 5. Prune one-off cited works if configured to.
    if settings.graph_min_mentions > 1:
        _prune(builder, settings.graph_min_mentions)

    # 6. Topics for every paper, inherited by its authors.
    _log("Assigning research interests…")
    _assign_topics(builder, taxonomy, embed)

    graph = builder.finish(taxonomy, [p.name for p in paths], corpus_hash)

    # 7. External profiles. Runs before the save so `graph.json` is written
    # once, complete — and before the viewer, so the HTML carries the profiles.
    if enrich:
        _log("")
        _apply_enrichment(graph, enrich_all, refresh=refresh_enrich)

    out = graph.save()
    _log(f"\nGraph written to {out}")
    _log(f"  {graph.summary()}")

    if export_html:
        _export_viewer(graph, embed=embed)

    return graph


def _export_viewer(graph: ScientistGraph, embed=None) -> None:
    from .viz import export_html as _export

    html = _export(graph, embed=embed or default_embedder())
    _log(f"  viewer: {html}")


def _viewer_is_stale(graph_path: Path) -> bool:
    html = graph_path.with_suffix(".html")
    if not html.exists():
        return True
    return html.stat().st_mtime < graph_path.stat().st_mtime


def _link_coauthors(builder: _Builder, author_ids: List[str], ev: Evidence) -> None:
    """CO_AUTHORED is symmetric, so store it in both directions."""
    unique = [a for i, a in enumerate(author_ids) if a and a not in author_ids[:i]]
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            builder.edge(a, b, "CO_AUTHORED", ev)
            builder.edge(b, a, "CO_AUTHORED", ev)


def _add_cited_work(builder: _Builder, work, filename: str, citing_pid: str) -> None:
    names = [n for n in work.authors if n and len(n) > 1]
    if not names:
        return
    # The extractor sometimes returns one packed string instead of a list.
    if len(names) == 1 and ("," in names[0] or " and " in names[0].lower()):
        names = split_author_list(names[0]) or names

    ev = evidence(filename, work.page or 1, work.title or "; ".join(names[:4]))
    pid = _work_id(names, work.title, work.year)
    builder.paper(pid, work.title, work.year, work.venue, "", False, "", ev)

    author_ids: List[str] = []
    for raw in names[:25]:
        sid = builder.scientist(raw, ev)
        if not sid:
            continue
        author_ids.append(sid)
        node = builder.scientists[sid]
        node["cited_works"] += 1
        if pid not in node["papers"]:
            node["papers"].append(pid)
        builder.edge(sid, pid, "AUTHORED", ev)

    paper_authors = builder.papers[pid]["authors"]
    for sid in author_ids:
        if sid not in paper_authors:
            paper_authors.append(sid)
    _link_coauthors(builder, author_ids, ev)
    builder.edge(citing_pid, pid, "CITES", ev)


def _prune(builder: _Builder, min_mentions: int) -> None:
    """Drop cited works seen fewer than ``min_mentions`` times, and any
    scientist left with no remaining work."""
    doomed = {
        pid for pid, p in builder.papers.items()
        if not p["in_corpus"] and p["mentions"] < min_mentions
    }
    if not doomed:
        return
    for pid in doomed:
        del builder.papers[pid]
    for s in builder.scientists.values():
        s["papers"] = [p for p in s["papers"] if p not in doomed]
    orphans = {
        sid for sid, s in builder.scientists.items()
        if not s["papers"] and not s["corpus_author"]
    }
    for sid in orphans:
        del builder.scientists[sid]
    builder._edges = {
        k: v for k, v in builder._edges.items()
        if k[0] not in doomed and k[1] not in doomed
        and k[0] not in orphans and k[1] not in orphans
    }


def _assign_topics(builder: _Builder, taxonomy: Taxonomy, embed) -> None:
    """Topic every paper by its title, then let authors inherit.

    Papers whose title never survived extraction inherit the topics of the
    corpus paper that cited them, at half weight — a weak signal, but it beats
    leaving a cited author with no research interests at all.
    """
    # Corpus papers are topiced from title + abstract and may carry several
    # interests; a one-line reference entry gets fewer, from its title alone.
    corpus_pids = [p for p in builder.papers if builder.papers[p]["in_corpus"]]
    cited_pids = [p for p in builder.papers if not builder.papers[p]["in_corpus"]]

    if corpus_pids:
        texts = [builder.topic_text.get(p) or builder.papers[p]["title"]
                 for p in corpus_pids]
        for pid, assigned in zip(corpus_pids, taxonomy.assign(texts, embed,
                                                              max_topics=5)):
            builder.note_topics(pid, assigned)

    if cited_pids:
        texts = [
            f"{builder.papers[p]['title']} {builder.papers[p]['venue']}".strip()
            or "dopamine"
            for p in cited_pids
        ]
        for pid, assigned in zip(cited_pids, taxonomy.assign(texts, embed,
                                                             max_topics=3)):
            builder.note_topics(pid, assigned)

    # Inheritance pass for untitled works.
    cites: Dict[str, List[str]] = defaultdict(list)
    for (src, tgt, rel) in builder._edges:
        if rel == "CITES":
            cites[tgt].append(src)
    for pid, paper in builder.papers.items():
        if paper["topics"] or paper["in_corpus"]:
            continue
        inherited: Dict[str, float] = defaultdict(float)
        for parent in cites.get(pid, []):
            for tid in builder.papers.get(parent, {}).get("topics", []):
                inherited[tid] += 0.5
        if inherited:
            ranked = sorted(inherited.items(), key=lambda kv: kv[1], reverse=True)[:2]
            builder.note_topics(pid, ranked)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the scientist knowledge graph.")
    parser.add_argument("--force", action="store_true", help="rebuild even if current")
    parser.add_argument("--no-refs", action="store_true",
                        help="skip reference lists (front matter only — fast)")
    parser.add_argument("--enrich", action="store_true",
                        help="force OpenAlex enrichment on (the default; "
                             "overrides GRAPH_ENRICH=0)")
    parser.add_argument("--no-enrich", action="store_true",
                        help="skip OpenAlex — build with no network calls")
    parser.add_argument("--enrich-all", action="store_true",
                        help="enrich cited researchers too, not just corpus "
                             "authors (thousands of requests — slow)")
    parser.add_argument("--refresh-enrich", action="store_true",
                        help="re-fetch OpenAlex profiles that are already "
                             "present (citation counts move; the graph doesn't)")
    parser.add_argument("--no-html", action="store_true", help="skip the HTML viewer")
    args = parser.parse_args()

    # --no-enrich wins over --enrich; both override the env default.
    if args.no_enrich:
        enrich = False
    elif args.enrich or args.enrich_all or args.refresh_enrich:
        enrich = True
    else:
        enrich = None

    build_graph(
        force=args.force,
        with_refs=not args.no_refs,
        enrich=enrich,
        enrich_all=args.enrich_all or None,
        refresh_enrich=args.refresh_enrich,
        export_html=not args.no_html,
    )


if __name__ == "__main__":
    main()
