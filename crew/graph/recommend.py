"""Connection recommendation over the scientist graph.

Two different questions get two different functions, because they have two
different right answers:

* :func:`similar_scientists` — "who works on what I work on?" Existing
  collaborators are legitimate answers.
* :func:`recommend_connections` — "who *should* I meet?" A person you have
  already published with is a useless suggestion, so co-authors are excluded and
  same-institution candidates are discounted. The value of an introduction is
  highest across a structural hole, not within a team you are already in.

Scoring blends four normalised signals (weights live in :mod:`crew.config`):

======================  ==========================================================
``topic_overlap``       IDF-weighted Jaccard over research interests. Inverse
                        document frequency matters: nearly everyone in this
                        corpus touches "dopamine receptors", so sharing it says
                        little, while sharing "dopamine fasting" says a lot.
``profile_similarity``  Cosine between profile-text embeddings — catches related
                        work that never shares an exact topic label.
``graph_proximity``     Adamic-Adar over the co-authorship projection: shared
                        collaborators predict future collaboration, weighted so
                        a mutual friend with 3 co-authors counts for more than
                        one with 300.
``citation_affinity``   Co-citation — being cited by the same corpus papers.
======================  ==========================================================

Every number, and every word of the ``rationale``, is computed here in Python
from graph state. An LLM may later *narrate* a recommendation
(:mod:`crew.graph.chat`), but it never produces one — the same "agents decide,
the orchestrator guarantees" split the Q&A crew uses for citations.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from ..config import settings
from .schema import Evidence, Recommendation, Signals


class Recommender:
    """Scores candidate connections for a scientist. Construct once per graph."""

    def __init__(self, graph, embed: Optional[Callable[[List[str]], np.ndarray]] = None):
        self.graph = graph
        self._embed = embed
        self._idf: Optional[Dict[str, float]] = None
        self._citing: Optional[Dict[str, Set[str]]] = None

    # -- cached corpus statistics -------------------------------------------
    @property
    def topic_idf(self) -> Dict[str, float]:
        """log(N / df) per topic, so common interests carry less weight."""
        if self._idf is None:
            total = max(len(self.graph.scientists), 1)
            df: Dict[str, int] = {}
            for s in self.graph.scientists.values():
                for t in set(s["topics"]):
                    df[t] = df.get(t, 0) + 1
            self._idf = {
                t: math.log(1.0 + total / max(count, 1)) for t, count in df.items()
            }
        return self._idf

    @property
    def citing_sets(self) -> Dict[str, Set[str]]:
        """scientist id -> ids of the corpus papers that *cite* their work.

        Derived from ``CITES`` edges, not from where a name happens to appear.
        The distinction matters: a paper's own authors appear on its front page
        without being cited by it, so keying on appearance made every co-author
        of a corpus paper look co-cited with every other — including people who
        share nothing else.

        Coarse (there are only nine citing papers) but real: someone cited by
        both the ADHD meta-analysis and the Parkinson's review keeps different
        intellectual company from someone cited by one alone.
        """
        if self._citing is None:
            cited_by: Dict[str, Set[str]] = defaultdict(set)
            for e in self.graph.edges:
                if e["relation"] == "CITES":
                    cited_by[e["target"]].add(e["source"])

            out: Dict[str, Set[str]] = defaultdict(set)
            for sid, s in self.graph.scientists.items():
                for pid in s["papers"]:
                    out[sid] |= cited_by.get(pid, set())
            self._citing = dict(out)
        return self._citing

    # -- individual signals --------------------------------------------------
    def _topic_overlap(self, a: str, b: str) -> tuple[float, List[str]]:
        ta, tb = set(self.graph.topics_of(a)), set(self.graph.topics_of(b))
        if not ta or not tb:
            return 0.0, []
        shared = ta & tb
        if not shared:
            return 0.0, []
        idf = self.topic_idf
        num = sum(idf.get(t, 1.0) for t in shared)
        den = sum(idf.get(t, 1.0) for t in ta | tb)
        labels = sorted(
            shared, key=lambda t: idf.get(t, 1.0), reverse=True
        )
        return (num / den if den else 0.0), [
            self.graph.topics[t]["label"] for t in labels if t in self.graph.topics
        ]

    def _profile_similarity(self, a: str, candidates: Sequence[str]) -> Dict[str, float]:
        if self._embed is None:
            return {c: 0.0 for c in candidates}
        ids, matrix = self.graph.profile_vectors(self._embed)
        pos = {sid: i for i, sid in enumerate(ids)}
        if a not in pos:
            return {c: 0.0 for c in candidates}
        row = matrix[pos[a]]
        out = {}
        for c in candidates:
            if c in pos:
                # Cosine on normalised vectors; clamp the small negative tail
                # so the signal stays on the same [0, 1] scale as the others.
                out[c] = max(0.0, float(row @ matrix[pos[c]]))
            else:
                out[c] = 0.0
        return out

    def _adamic_adar(self, a: str, candidates: Sequence[str]) -> Dict[str, float]:
        g = self.graph.coauthor_graph
        if a not in g:
            return {c: 0.0 for c in candidates}
        neighbours_a = set(g.neighbors(a))
        raw: Dict[str, float] = {}
        for c in candidates:
            if c not in g:
                raw[c] = 0.0
                continue
            shared = neighbours_a & set(g.neighbors(c))
            raw[c] = sum(
                1.0 / math.log(max(g.degree(z), 2)) for z in shared
            )
        peak = max(raw.values(), default=0.0)
        return {c: (v / peak if peak > 0 else 0.0) for c, v in raw.items()}

    def _citation_affinity(self, a: str, b: str) -> float:
        sets = self.citing_sets
        sa, sb = sets.get(a, set()), sets.get(b, set())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # -- public API ----------------------------------------------------------
    def candidates(self, sid: str, exclude_collaborators: bool) -> List[str]:
        pool = set(self.graph.scientists) - {sid}
        if exclude_collaborators:
            pool -= set(self.graph.collaborators(sid))
        return sorted(pool)

    def score_all(self, sid: str, exclude_collaborators: bool = True,
                  limit: int = 10, pool: Optional[List[str]] = None
                  ) -> List[Recommendation]:
        """Score every candidate and return the top ``limit``."""
        if sid not in self.graph.scientists:
            return []
        cands = pool if pool is not None else self.candidates(sid, exclude_collaborators)
        if not cands:
            return []

        profile_sims = self._profile_similarity(sid, cands)
        proximity = self._adamic_adar(sid, cands)
        my_institutions = set(self.graph.scientists[sid]["institutions"])
        my_collaborators = set(self.graph.collaborators(sid))
        g = self.graph.coauthor_graph

        scored: List[Recommendation] = []
        for cid in cands:
            other = self.graph.scientists[cid]
            topic_score, shared_topics = self._topic_overlap(sid, cid)
            profile_score = profile_sims.get(cid, 0.0)
            graph_score = proximity.get(cid, 0.0)
            cite_score = self._citation_affinity(sid, cid)

            same_inst = bool(my_institutions & set(other["institutions"]))

            total = (
                settings.rec_w_topic * topic_score
                + settings.rec_w_profile * profile_score
                + settings.rec_w_graph * graph_score
                + settings.rec_w_citation * cite_score
            )
            if same_inst and exclude_collaborators:
                total *= (1.0 - settings.rec_same_institution_penalty)
            if total <= 0.0:
                continue

            bridges = sorted(
                my_collaborators & set(g.neighbors(cid)) if cid in g else set()
            )
            bridge_names = [
                self.graph.scientists[b]["name"] for b in bridges[:5]
                if b in self.graph.scientists
            ]
            institutions = [
                self.graph.institutions[i]["name"] for i in other["institutions"]
                if i in self.graph.institutions
            ]

            signals = Signals(
                topic_overlap=round(topic_score, 4),
                profile_similarity=round(profile_score, 4),
                graph_proximity=round(graph_score, 4),
                citation_affinity=round(cite_score, 4),
                same_institution=same_inst,
            )
            scored.append(
                Recommendation(
                    scientist_id=cid,
                    name=other["name"],
                    score=round(min(total, 1.0), 4),
                    signals=signals,
                    shared_topics=shared_topics[:5],
                    shared_collaborators=bridge_names,
                    institutions=institutions[:3],
                    rationale=_rationale(
                        self.graph.scientists[sid]["name"], other["name"],
                        signals, shared_topics[:5], bridge_names, institutions[:1],
                    ),
                    evidence=_pair_evidence(self.graph, sid, cid),
                )
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]


def recommend_connections(graph, sid: str, embed=None, limit: int = 10
                          ) -> List[Recommendation]:
    """Who this scientist should meet: excludes existing co-authors."""
    return Recommender(graph, embed).score_all(
        sid, exclude_collaborators=True, limit=limit
    )


def similar_scientists(graph, sid: str, embed=None, limit: int = 10
                       ) -> List[Recommendation]:
    """Who most resembles this scientist: co-authors are allowed."""
    return Recommender(graph, embed).score_all(
        sid, exclude_collaborators=False, limit=limit
    )


def _rationale(a_name: str, b_name: str, signals: Signals,
               shared_topics: List[str], bridges: List[str],
               institutions: List[str]) -> str:
    """Plain-language justification, assembled from the computed signals only."""
    bits: List[str] = []
    if shared_topics:
        topics = ", ".join(shared_topics[:3])
        bits.append(f"both work on {topics}")
    if bridges:
        names = ", ".join(bridges[:2])
        more = f" and {len(bridges) - 2} other" + ("s" if len(bridges) - 2 > 1 else "") \
            if len(bridges) > 2 else ""
        bits.append(f"connected through {names}{more}")
    if signals["citation_affinity"] > 0:
        bits.append("cited together in the same papers")
    if not bits and signals["profile_similarity"] > 0:
        bits.append("closely related research profiles")
    if institutions:
        bits.append(f"{b_name} is at {institutions[0]}")

    if not bits:
        return f"{b_name} is a weak match for {a_name}."
    body = "; ".join(bits)
    return f"{a_name} → {b_name}: {body}."


def _pair_evidence(graph, a: str, b: str, limit: int = 4) -> List[Evidence]:
    """Where in the corpus each side of the pair was observed."""
    out: List[Evidence] = []
    seen = set()
    for sid in (a, b):
        for ev in graph.evidence_for(sid, limit=2):
            key = (ev["file"], ev["page"])
            if key not in seen:
                seen.add(key)
                out.append(ev)
    return out[:limit]
