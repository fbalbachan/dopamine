"""The persisted scientist graph, and every read path over it.

Storage is a single human-readable JSON file (``crew/.graph/graph.json``) in the
same spirit as ``chunks.json`` in the retrieval layer: small enough to open in an
editor, diffable, and loadable without any database. NetworkX is built on top at
load time rather than persisted, so the on-disk format never depends on a
library version.

The graph is a ``MultiDiGraph`` — nodes are typed by ``kind`` and edges by
``relation`` — plus one derived structure that earns its keep: the undirected
**co-authorship projection**, which is what the link-prediction half of the
recommender runs on.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import settings
from .canon import parse_name
from .schema import (
    Edge,
    Evidence,
    GraphStats,
    Institution,
    Paper,
    Scientist,
    Topic,
)

GRAPH_FILENAME = "graph.json"


class ScientistGraph:
    """Loaded graph + queries. Construct via :meth:`load` or the build pipeline."""

    def __init__(
        self,
        scientists: Dict[str, Scientist],
        institutions: Dict[str, Institution],
        topics: Dict[str, Topic],
        papers: Dict[str, Paper],
        edges: List[Edge],
        stats: GraphStats,
    ):
        self.scientists = scientists
        self.institutions = institutions
        self.topics = topics
        self.papers = papers
        self.edges = edges
        self.stats = stats
        self._nx = None
        self._coauthor = None
        self._profile_matrix: Optional[np.ndarray] = None
        self._profile_ids: List[str] = []
        # (source, target, relation) set, built lazily by enrichment so an
        # append can check for a duplicate without rescanning ~44k edges.
        self._edge_index: Optional[set] = None

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ScientistGraph":
        path = path or (settings.graph_dir / GRAPH_FILENAME)
        if not path.exists():
            raise FileNotFoundError(
                f"No graph found at {path}. Run `python -m crew.graph.build` first."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            scientists={s["id"]: s for s in raw.get("scientists", [])},
            institutions={i["id"]: i for i in raw.get("institutions", [])},
            topics={t["id"]: t for t in raw.get("topics", [])},
            papers={p["id"]: p for p in raw.get("papers", [])},
            edges=raw.get("edges", []),
            stats=raw.get("stats", {}),
        )

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (settings.graph_dir / GRAPH_FILENAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False: author names carry accents and CJK characters that
        # would otherwise be escaped into unreadable soup.
        path.write_text(
            json.dumps(
                {
                    "stats": self.stats,
                    "scientists": list(self.scientists.values()),
                    "institutions": list(self.institutions.values()),
                    "topics": list(self.topics.values()),
                    "papers": list(self.papers.values()),
                    "edges": self.edges,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        return path

    # -- post-build mutation -------------------------------------------------
    def recount_stats(self) -> GraphStats:
        """Recompute derived counts after nodes or edges changed post-build.

        :func:`crew.graph.enrich.enrich_graph` mutates a graph that was already
        frozen by ``build._Builder.finish``, adding institutions and
        ``AFFILIATED_WITH`` edges. Without this the persisted ``stats`` keep
        describing the graph as it was before the pass — and ``stats`` is what
        the CLI, the viewer header and ``summary()`` all read.

        Also drops the derived caches (``nx``, the co-authorship projection,
        profile vectors), since every one of them was built from the old
        edge list or the old affiliations.
        """
        relations: Dict[str, int] = {}
        for e in self.edges:
            relations[e["relation"]] = relations.get(e["relation"], 0) + 1
        self.stats.update({
            "scientists": len(self.scientists),
            "corpus_authors": sum(1 for s in self.scientists.values()
                                  if s["corpus_author"]),
            "institutions": len(self.institutions),
            "topics": len(self.topics),
            "papers": len(self.papers),
            "corpus_papers": sum(1 for p in self.papers.values() if p["in_corpus"]),
            "edges": len(self.edges),
            "relations": relations,
            "enriched": sum(1 for s in self.scientists.values()
                            if s.get("external")),
        })
        self._nx = None
        self._coauthor = None
        self._profile_matrix = None
        self._profile_ids = []
        self._edge_index = None
        return self.stats

    # -- networkx views ------------------------------------------------------
    @property
    def nx(self):
        """The full typed multigraph."""
        if self._nx is None:
            import networkx as nx

            g = nx.MultiDiGraph()
            for sid, s in self.scientists.items():
                g.add_node(sid, kind="scientist", label=s["name"])
            for iid, i in self.institutions.items():
                g.add_node(iid, kind="institution", label=i["name"])
            for tid, t in self.topics.items():
                g.add_node(tid, kind="topic", label=t["label"])
            for pid, p in self.papers.items():
                g.add_node(pid, kind="paper", label=p["title"] or pid)
            for e in self.edges:
                if e["source"] in g and e["target"] in g:
                    g.add_edge(e["source"], e["target"],
                               relation=e["relation"], weight=e.get("weight", 1.0))
            self._nx = g
        return self._nx

    @property
    def coauthor_graph(self):
        """Undirected, weighted co-authorship projection over scientists only.

        This is the structure link prediction needs: Adamic-Adar and common-
        neighbour counts are undefined on a mixed-kind multigraph, and running
        them over topic or institution nodes would make everyone at one
        university look like a collaborator.
        """
        if self._coauthor is None:
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(self.scientists.keys())
            for e in self.edges:
                if e["relation"] != "CO_AUTHORED":
                    continue
                a, b = e["source"], e["target"]
                if a not in self.scientists or b not in self.scientists:
                    continue
                w = e.get("weight", 1.0)
                if g.has_edge(a, b):
                    g[a][b]["weight"] = max(g[a][b]["weight"], w)
                else:
                    g.add_edge(a, b, weight=w)
            self._coauthor = g
        return self._coauthor

    # -- lookups -------------------------------------------------------------
    def find_scientist(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Resolve a free-text name to scientist ids, best match first.

        Tries the canonical key first (so "W. Schultz" and "Wolfram Schultz"
        both hit exactly), then falls back to fuzzy matching over names and
        aliases — which is what makes a typo or a surname-only query work.
        """
        q = (query or "").strip()
        if not q:
            return []

        parsed = parse_name(q)
        if parsed and parsed.key in self.scientists:
            return [(parsed.key, 1.0)]

        ql = q.lower()
        scored: Dict[str, float] = {}
        for sid, s in self.scientists.items():
            forms = [s["name"], *s.get("aliases", [])]
            best = 0.0
            for form in forms:
                fl = form.lower()
                if fl == ql:
                    best = 1.0
                    break
                # A surname-only query should match strongly.
                if ql in fl.split() or fl.split()[-1:] == [ql]:
                    best = max(best, 0.92)
                best = max(best, SequenceMatcher(None, ql, fl).ratio())
            if best >= 0.62:
                scored[sid] = best
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        # Corpus authors outrank cited-only people at equal string similarity —
        # they are far more likely to be who the user meant.
        ranked.sort(key=lambda kv: (round(kv[1], 3),
                                    self.scientists[kv[0]]["corpus_author"]),
                    reverse=True)
        return ranked[:limit]

    def find_topic(self, query: str) -> Optional[str]:
        ql = (query or "").strip().lower()
        if not ql:
            return None
        for tid, t in self.topics.items():
            if tid == ql or t["label"].lower() == ql:
                return tid
        best, best_score = None, 0.0
        for tid, t in self.topics.items():
            hay = " ".join([t["label"], *t.get("aliases", [])]).lower()
            score = SequenceMatcher(None, ql, t["label"].lower()).ratio()
            if ql in hay:
                score = max(score, 0.9)
            if score > best_score:
                best, best_score = tid, score
        return best if best_score >= 0.6 else None

    def find_institution(self, query: str) -> Optional[str]:
        ql = (query or "").strip().lower()
        if not ql:
            return None
        best, best_score = None, 0.0
        for iid, inst in self.institutions.items():
            score = SequenceMatcher(None, ql, inst["name"].lower()).ratio()
            if ql in inst["name"].lower():
                score = max(score, 0.9)
            if score > best_score:
                best, best_score = iid, score
        return best if best_score >= 0.6 else None

    # -- neighbourhoods ------------------------------------------------------
    def collaborators(self, sid: str) -> List[str]:
        g = self.coauthor_graph
        return sorted(g.neighbors(sid)) if sid in g else []

    def topics_of(self, sid: str) -> List[str]:
        s = self.scientists.get(sid)
        return list(s["topics"]) if s else []

    def scientists_by_topic(self, topic_id: str) -> List[str]:
        t = self.topics.get(topic_id)
        return list(t["scientists"]) if t else []

    def scientists_by_institution(self, inst_id: str) -> List[str]:
        inst = self.institutions.get(inst_id)
        return list(inst["scientists"]) if inst else []

    def papers_of(self, sid: str) -> List[Paper]:
        s = self.scientists.get(sid)
        if not s:
            return []
        return [self.papers[p] for p in s["papers"] if p in self.papers]

    def institutions_of(self, sid: str) -> List[Institution]:
        s = self.scientists.get(sid)
        if not s:
            return []
        return [self.institutions[i] for i in s["institutions"] if i in self.institutions]

    def evidence_for(self, sid: str, limit: int = 3) -> List[Evidence]:
        s = self.scientists.get(sid)
        return list(s.get("evidence", []))[:limit] if s else []

    # -- profile embeddings --------------------------------------------------
    def profile_text(self, sid: str) -> str:
        """The text that represents a scientist for similarity purposes.

        Topic labels and paper titles — not the person's name. Two researchers
        should look alike because they study the same things, not because their
        surnames are spelled similarly.
        """
        s = self.scientists.get(sid)
        if not s:
            return ""
        parts: List[str] = []
        parts.extend(self.topics[t]["label"] for t in s["topics"] if t in self.topics)
        for pid in s["papers"][:8]:
            paper = self.papers.get(pid)
            if paper and paper["title"]:
                parts.append(paper["title"])
        parts.extend(
            self.institutions[i]["name"] for i in s["institutions"][:3]
            if i in self.institutions
        )
        return ". ".join(p for p in parts if p) or s["name"]

    def profile_vectors(self, embed: Callable[[List[str]], np.ndarray]
                        ) -> Tuple[List[str], np.ndarray]:
        """Embed every scientist's profile once; cached on the instance."""
        if self._profile_matrix is None:
            ids = sorted(self.scientists.keys())
            texts = [self.profile_text(i) for i in ids]
            self._profile_ids = ids
            self._profile_matrix = embed(texts)
        return self._profile_ids, self._profile_matrix

    # -- summaries -----------------------------------------------------------
    def top_scientists(self, limit: int = 20) -> List[str]:
        """Most connected people, corpus authors first — the graph's 'front page'."""
        ranked = sorted(
            self.scientists.values(),
            key=lambda s: (s["corpus_author"], s["degree"], s["cited_works"]),
            reverse=True,
        )
        return [s["id"] for s in ranked[:limit]]

    def summary(self) -> str:
        st = self.stats or {}
        return (
            f"{st.get('scientists', len(self.scientists))} scientists "
            f"({st.get('corpus_authors', 0)} corpus authors), "
            f"{st.get('institutions', len(self.institutions))} institutions, "
            f"{st.get('topics', len(self.topics))} topics, "
            f"{st.get('papers', len(self.papers))} papers, "
            f"{st.get('edges', len(self.edges))} edges"
        )
