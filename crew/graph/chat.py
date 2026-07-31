"""GraphCrew — the orchestrator for graph-grounded questions.

Per question:

    resolve intent + entities (Navigator agent)
      -> run the matching graph query IN PYTHON      <- facts enter here
      -> register each fact as [G#] evidence
      -> Analyst agent narrates, citing [G#] ids only
      -> _resolve() rebuilds sources from the registry
      -> typed GraphAnswer dict

This is the retrieval half of GraphRAG: instead of embedding-matched passages,
the retrieved context is a **subgraph** — a person's institutions, interests,
collaborators and scored recommendations — serialised into evidence lines.

The grounding guarantee is the same one :mod:`crew.chat` enforces, and it is
enforced the same way. The Analyst is handed ``[G1] … [G7]`` and may cite only
those ids; ``sources`` is rebuilt from the registry afterwards, so an invented
``[G9]`` is silently dropped rather than becoming a fake citation. Scores and
rationales in ``recommendations`` come straight from
:mod:`crew.graph.recommend`, never from the model — an LLM that talks up a
match cannot change the number attached to it.

Queries without an API key still work: every intent returns a deterministic
summary built by :func:`_fallback_answer`, so the graph is explorable offline.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from ..config import settings
from ..memory import MemoryStore
from .recommend import Recommender
from .schema import Evidence, GraphAnswer, Recommendation, graph_refusal
from .store import ScientistGraph

_EVIDENCE_RE = re.compile(r"\b(G\d+)\b")

Intent = Literal["profile", "connections", "recommend", "topic", "institution",
                 "overview", "compare"]


class _Plan(BaseModel):
    """What the Navigator decided to look up."""
    intent: Intent = "overview"
    scientists: List[str] = Field(default_factory=list)
    topic: str = ""
    institution: str = ""


class _AnalystOut(BaseModel):
    answer: str
    claims: List[dict] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class EvidenceRegistry:
    """Ground truth for one turn: [G#] -> the graph fact it names."""

    def __init__(self) -> None:
        self.items: Dict[str, Evidence] = {}
        # The graph fact behind each id, kept separately from its provenance:
        # ``items[eid]`` points at the PDF page a fact was extracted from, which
        # is not the same string as the fact itself.
        self.facts: Dict[str, str] = {}
        self._lines: List[str] = []

    def add(self, text: str, ev: Optional[Evidence] = None) -> str:
        eid = f"G{len(self.items) + 1}"
        self.items[eid] = ev or Evidence(file="graph", page=0, snippet=text)
        self.facts[eid] = text
        self._lines.append(f"[{eid}] {text}")
        return eid

    def get(self, eid: str) -> Optional[Evidence]:
        return self.items.get(eid)

    def block(self) -> str:
        return "\n".join(self._lines) if self._lines else "The graph has no data for this."


class GraphCrew:
    """Ask the scientist graph questions in natural language."""

    def __init__(self, graph: Optional[ScientistGraph] = None,
                 memory: Optional[MemoryStore] = None, embed=None):
        self.graph = graph or ScientistGraph.load()
        self.memory = memory or MemoryStore()
        self._embed = embed
        self._agents = None
        self._recommender: Optional[Recommender] = None

    # -- public --------------------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> GraphAnswer:
        plan = self._plan(question)
        registry = EvidenceRegistry()
        entities, recommendations = self._retrieve(plan, registry)

        if not registry.items:
            resp = graph_refusal(session_id, question)
            self.memory.record_exchange(session_id, question, resp["answer"])
            return resp

        answer = self._narrate(question, plan, registry)
        resp = self._resolve(session_id, question, plan, answer, registry,
                             entities, recommendations)
        self.memory.record_exchange(session_id, question, resp["answer"])
        return resp

    def recommender(self) -> Recommender:
        """Scoring is lazy, but never embedding-free.

        Falling back to ``embed=None`` would zero the profile-similarity signal
        for every candidate — which reads as "nobody is similar" rather than
        "this quarter of the score is switched off". The model loads on first
        use, so a question that needs no recommendations never pays for it.
        """
        if self._recommender is None:
            from .embedding import default_embedder

            self._recommender = Recommender(
                self.graph, self._embed or default_embedder()
            )
        return self._recommender

    # -- planning ------------------------------------------------------------
    def _plan(self, question: str) -> _Plan:
        """Decide intent + entities. Falls back to keyword rules with no key."""
        if not settings.anthropic_api_key:
            return _heuristic_plan(question)
        try:
            from crewai import Crew, Process, Task

            agents = self._ensure_agents()
            names = ", ".join(
                self.graph.scientists[s]["name"] for s in self.graph.top_scientists(25)
            )
            topics = ", ".join(t["label"] for t in self.graph.topics.values())
            task = Task(
                description=(
                    f"Question: {question}\n\n"
                    f"Known scientists (sample): {names}\n"
                    f"Known research topics: {topics}\n\n"
                    "Classify the intent and extract the entities:\n"
                    "- profile: what do we know about person X\n"
                    "- connections: who has X worked with\n"
                    "- recommend: who should X connect with / who is similar to X\n"
                    "- topic: who works on topic T\n"
                    "- institution: who is at institution I\n"
                    "- compare: how do X and Y relate\n"
                    "- overview: general questions about the network\n\n"
                    "Return names exactly as the user wrote them. Leave a field "
                    "empty when the question does not mention it."
                ),
                expected_output=(
                    "JSON: intent, scientists (list of names), topic, institution."
                ),
                agent=agents["navigator"],
                output_pydantic=_Plan,
            )
            crew = Crew(agents=[agents["navigator"]], tasks=[task],
                        process=Process.sequential, memory=False, verbose=False)
            plan = _coerce(crew.kickoff(), _Plan)
            return plan or _heuristic_plan(question)
        except Exception:
            return _heuristic_plan(question)

    def _ensure_agents(self):
        if self._agents is None:
            from .agents import build_agents, build_llm

            self._agents = build_agents(build_llm())
        return self._agents

    # -- retrieval (this is where facts enter) -------------------------------
    def _retrieve(self, plan: _Plan, registry: EvidenceRegistry
                  ) -> Tuple[List[str], List[Recommendation]]:
        g = self.graph
        entities: List[str] = []
        recommendations: List[Recommendation] = []

        resolved: List[str] = []
        for name in plan.scientists[:3]:
            matches = g.find_scientist(name)
            if matches:
                resolved.append(matches[0][0])
        entities.extend(resolved)

        if plan.intent in ("profile", "connections", "recommend", "compare") and resolved:
            for sid in resolved:
                self._describe_scientist(sid, registry)
            if plan.intent in ("recommend",) or (plan.intent == "profile" and len(resolved) == 1):
                recommendations = self.recommender().score_all(
                    resolved[0], exclude_collaborators=True, limit=6
                )
                for rec in recommendations:
                    registry.add(
                        f"RECOMMENDATION: {g.scientists[resolved[0]]['name']} → "
                        f"{rec['name']} (score {rec['score']:.2f}). {rec['rationale']} "
                        f"Signals: topics={rec['signals']['topic_overlap']:.2f}, "
                        f"profile={rec['signals']['profile_similarity']:.2f}, "
                        f"network={rec['signals']['graph_proximity']:.2f}.",
                        rec["evidence"][0] if rec["evidence"] else None,
                    )
            if plan.intent == "compare" and len(resolved) >= 2:
                self._compare(resolved[0], resolved[1], registry)

        if plan.intent == "topic" or plan.topic:
            tid = g.find_topic(plan.topic or "")
            if tid:
                entities.append(tid)
                topic = g.topics[tid]
                # Corpus authors first: they are the people with affiliations,
                # so a "who works on X and where" question can be answered.
                ranked = sorted(
                    (s for s in topic["scientists"] if s in g.scientists),
                    key=lambda s: (g.scientists[s]["corpus_author"],
                                   g.scientists[s]["degree"]),
                    reverse=True,
                )
                registry.add(
                    f"TOPIC {topic['label']}: {topic['description']} "
                    f"{len(topic['scientists'])} researchers work on it, including "
                    f"{', '.join(g.scientists[s]['name'] for s in ranked[:15])}."
                )
                # One line per notable person, so affiliations are actually in
                # evidence rather than merely present somewhere in the graph.
                for sid in ranked[:8]:
                    self._describe_scientist(sid, registry, brief=True)

        if plan.intent == "institution" or plan.institution:
            iid = g.find_institution(plan.institution or "")
            if iid:
                entities.append(iid)
                inst = g.institutions[iid]
                members = [s for s in inst["scientists"] if s in g.scientists]
                registry.add(
                    f"INSTITUTION {inst['name']}"
                    f"{' (' + inst['country'] + ')' if inst['country'] else ''}"
                    f"{': ' + '; '.join(inst['departments'][:3]) if inst['departments'] else ''}. "
                    f"{len(members)} people in the graph — "
                    f"{', '.join(g.scientists[s]['name'] for s in members[:15])}.",
                    inst["evidence"][0] if inst["evidence"] else None,
                )
                for sid in members[:8]:
                    self._describe_scientist(sid, registry, brief=True)

        if not registry.items and plan.intent == "overview":
            self._overview(registry)
        return entities, recommendations

    def _describe_scientist(self, sid: str, registry: EvidenceRegistry,
                            brief: bool = False) -> None:
        """Register what the graph knows about one person.

        ``brief`` emits only the profile line — used when summarising a topic or
        institution roster, where a full paper-and-collaborator dump per person
        would bury the answer.
        """
        g = self.graph
        s = g.scientists[sid]
        institutions = [i["name"] for i in g.institutions_of(sid)]
        topics = [g.topics[t]["label"] for t in s["topics"] if t in g.topics]
        ev = s["evidence"][0] if s["evidence"] else None

        registry.add(
            f"PROFILE {s['name']}: "
            f"{'author of a corpus paper' if s['corpus_author'] else 'cited researcher'}; "
            f"institutions: {', '.join(institutions) or 'not recorded in the graph'}; "
            f"research interests: {', '.join(topics) or 'unknown'}; "
            f"{s['degree']} co-authors; appears on {s['cited_works']} referenced works.",
            ev,
        )
        if brief:
            return
        external = s.get("external") or {}
        if external.get("works_count"):
            registry.add(
                f"EXTERNAL PROFILE (OpenAlex) for {s['name']}: "
                f"{external['works_count']} works, "
                f"{external.get('cited_by_count', 0)} citations, "
                f"h-index {external.get('h_index', 0)}"
                f"{', ORCID ' + external['orcid'] if external.get('orcid') else ''}."
            )
        for paper in g.papers_of(sid)[:5]:
            registry.add(
                f"PAPER by {s['name']}: \"{paper['title']}\""
                f"{' (' + str(paper['year']) + ')' if paper['year'] else ''}"
                f"{' [in the nine-paper corpus]' if paper['in_corpus'] else ''}.",
                paper["evidence"][0] if paper["evidence"] else None,
            )
        collaborators = [
            g.scientists[c]["name"] for c in g.collaborators(sid)[:12]
            if c in g.scientists
        ]
        if collaborators:
            registry.add(
                f"COLLABORATORS of {s['name']} ({len(g.collaborators(sid))} total): "
                f"{', '.join(collaborators)}.", ev
            )

    def _compare(self, a: str, b: str, registry: EvidenceRegistry) -> None:
        g = self.graph
        shared_topics = set(g.topics_of(a)) & set(g.topics_of(b))
        shared_people = set(g.collaborators(a)) & set(g.collaborators(b))
        direct = b in g.collaborators(a)
        registry.add(
            f"RELATIONSHIP {g.scientists[a]['name']} ↔ {g.scientists[b]['name']}: "
            f"{'they have co-authored together' if direct else 'no direct co-authorship'}; "
            f"shared research interests: "
            f"{', '.join(g.topics[t]['label'] for t in shared_topics if t in g.topics) or 'none'}; "
            f"shared collaborators: "
            f"{', '.join(g.scientists[p]['name'] for p in list(shared_people)[:8]) or 'none'}."
        )

    def _overview(self, registry: EvidenceRegistry) -> None:
        g = self.graph
        registry.add(f"GRAPH SUMMARY: {g.summary()}.")
        top = [
            f"{g.scientists[s]['name']} ({g.scientists[s]['degree']} co-authors)"
            for s in g.top_scientists(10)
        ]
        registry.add(f"MOST CONNECTED: {', '.join(top)}.")
        busiest = sorted(g.topics.values(), key=lambda t: len(t["scientists"]),
                         reverse=True)[:8]
        registry.add(
            "LARGEST RESEARCH TOPICS: "
            + ", ".join(f"{t['label']} ({len(t['scientists'])})" for t in busiest) + "."
        )

    # -- narration -----------------------------------------------------------
    def _narrate(self, question: str, plan: _Plan,
                 registry: EvidenceRegistry) -> _AnalystOut:
        if not settings.anthropic_api_key:
            return _fallback_answer(registry)
        try:
            from crewai import Crew, Process, Task

            agents = self._ensure_agents()
            task = Task(
                description=(
                    f"Question: {question}\n"
                    f"Detected intent: {plan.intent}\n\n"
                    f"GRAPH EVIDENCE (cite these ids):\n{registry.block()}\n\n"
                    "Answer the question using ONLY the evidence above. Rules:\n"
                    "- Cite ids inline like [G2]. Never state a fact without one.\n"
                    "- Report recommendation scores exactly as given; do not "
                    "re-rank or re-score them.\n"
                    "- Distinguish co-authorship from topical similarity.\n"
                    "- In `claims`, list each factual sentence as "
                    '{"statement": str, "evidence_ids": [str]}.\n'
                    "- If the evidence does not answer the question, say so."
                ),
                expected_output=(
                    "JSON: answer, claims (statement + evidence_ids), confidence."
                ),
                agent=agents["analyst"],
                output_pydantic=_AnalystOut,
            )
            crew = Crew(agents=[agents["analyst"]], tasks=[task],
                        process=Process.sequential, memory=False, verbose=False)
            return _coerce(crew.kickoff(), _AnalystOut) or _fallback_answer(registry)
        except Exception:
            return _fallback_answer(registry)

    # -- grounding resolution ------------------------------------------------
    @staticmethod
    def _resolve(session_id: str, question: str, plan: _Plan, synth: _AnalystOut,
                 registry: EvidenceRegistry, entities: List[str],
                 recommendations: List[Recommendation]) -> GraphAnswer:
        ordered: List[str] = []

        def see(eid: str) -> None:
            if eid not in ordered and registry.get(eid) is not None:
                ordered.append(eid)

        for claim in synth.claims:
            for eid in (claim.get("evidence_ids") or []):
                see(str(eid))
        for eid in _EVIDENCE_RE.findall(synth.answer or ""):
            see(eid)

        sources = [registry.get(eid) for eid in ordered]
        index_of = {eid: i for i, eid in enumerate(ordered)}
        claims = [
            {
                "statement": c.get("statement", ""),
                "source_indices": [
                    index_of[str(e)] for e in (c.get("evidence_ids") or [])
                    if str(e) in index_of
                ],
            }
            for c in synth.claims
        ]

        return GraphAnswer(
            session_id=session_id,
            question=question,
            answer=(synth.answer or "").strip(),
            intent=plan.intent,
            entities=entities,
            sources=sources,          # type: ignore[arg-type]
            claims=claims,
            recommendations=recommendations,   # computed in Python, never by the model
            in_scope=True,
            refused=False,
            confidence=synth.confidence
            if synth.confidence in ("high", "medium", "low") else "medium",
        )


# --- helpers --------------------------------------------------------------- #
def _coerce(result, model_cls):
    """Extract a pydantic instance from a CrewAI result (three strategies)."""
    cand = getattr(result, "pydantic", None)
    if isinstance(cand, model_cls):
        return cand
    tasks = getattr(result, "tasks_output", None)
    if tasks:
        cand = getattr(tasks[-1], "pydantic", None)
        if isinstance(cand, model_cls):
            return cand
    raw = getattr(result, "raw", None) or str(result)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        return model_cls(**json.loads(raw[start : end + 1]))
    except Exception:
        return None


_RECOMMEND_WORDS = ("recommend", "should meet", "should connect", "introduce",
                    "similar to", "who is like", "collaborat")
_CONNECTION_WORDS = ("worked with", "co-author", "coauthor", "collaborators of")


def _heuristic_plan(question: str) -> _Plan:
    """Keyword intent detection, used when Claude is unavailable.

    Capitalised runs are treated as candidate names — crude, but it keeps the
    whole CLI usable with no API key, and :meth:`ScientistGraph.find_scientist`
    fuzzy-matches whatever it produces.
    """
    q = question.lower()
    if any(w in q for w in _RECOMMEND_WORDS):
        intent: Intent = "recommend"
    elif any(w in q for w in _CONNECTION_WORDS):
        intent = "connections"
    elif "who works on" in q or "topic" in q or "research on" in q:
        intent = "topic"
    elif "university" in q or "institute" in q or "institution" in q or " at " in q:
        intent = "institution"
    elif "overview" in q or "summar" in q or "how many" in q:
        intent = "overview"
    else:
        intent = "profile"

    names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z']+)*\b", question)
    stop = {"Who", "What", "Which", "Where", "How", "Does", "Is", "Are", "The"}
    names = [n for n in names if n.split()[0] not in stop]
    return _Plan(intent=intent, scientists=names[:2],
                 topic=question if intent == "topic" else "",
                 institution=question if intent == "institution" else "")


def _fallback_answer(registry: EvidenceRegistry) -> _AnalystOut:
    """Deterministic answer assembled from evidence when no LLM is available."""
    ids = list(registry.facts)[:12]
    body = "\n".join(f"- {registry.facts[eid]} [{eid}]" for eid in ids)
    return _AnalystOut(
        answer=f"From the graph:\n{body}",
        claims=[{"statement": registry.facts[eid], "evidence_ids": [eid]}
                for eid in ids],
        confidence="medium",
    )
