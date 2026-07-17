"""DopamineCrew — the orchestrator.

Per question the pipeline is:

    load memory ─▶ MemoryKeeper rewrites follow-up to standalone
                 ─▶ scope guardrail (refuse if out-of-domain)
                 ─▶ Retriever plans queries ─▶ [RAG search in Python]
                 ─▶ WebSearcher decides + queries ─▶ [Brave search in Python]
                 ─▶ Synthesizer writes a grounded answer + claims
                 ─▶ resolve cited evidence ids to real sources
                 ─▶ record exchange in memory
                 ─▶ return a typed CrewResponse dict

The agents contribute judgment; the orchestrator executes the RAG/Brave tools
and owns the correctness invariants. Grounding is enforced in :meth:`_resolve`:
the final ``sources`` come from the :class:`~crew.tools.EvidenceRegistry` the
tools populated this turn, so no citation can reference a page or URL that
retrieval didn't actually return.
"""
from __future__ import annotations

import json
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .config import settings
from .memory import MemoryStore
from .rag import Retriever
from .schema import Claim, CrewResponse, Source, refusal
from .scope import check_scope

_EVIDENCE_RE = re.compile(r"\b([PW]\d+)\b")

# Statuses from search_web that mean "no usable web evidence".
_NO_WEB_PREFIXES = ("WEB_SEARCH_UNAVAILABLE", "WEB_SEARCH_ERROR", "No web results")


# --- Structured agent outputs ---------------------------------------------- #
class _RetrieverPlan(BaseModel):
    queries: List[str] = Field(default_factory=list)


class _WebDecision(BaseModel):
    need_web: bool = False
    query: str = ""


class _ClaimOut(BaseModel):
    statement: str = Field(..., description="One factual sentence from the answer.")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence ids ([P#]/[W#]) that support this statement.",
    )


class _SynthOut(BaseModel):
    answer: str
    claims: List[_ClaimOut] = Field(default_factory=list)
    used_web_search: bool = False
    confidence: Literal["high", "medium", "low"] = "medium"


class DopamineCrew:
    """Public entry point. Construct once, call :meth:`ask` per question."""

    def __init__(self, retriever: Optional[Retriever] = None,
                 memory: Optional[MemoryStore] = None):
        # Retriever is LLM-free, so it loads even without an API key (handy for
        # tests). Synthesis in `ask` does require ANTHROPIC_API_KEY.
        self.retriever = retriever or Retriever.load()
        self.memory = memory or MemoryStore()
        self._agents = None  # built lazily (needs the API key)

    # -- public --------------------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> CrewResponse:
        history = self.memory.context_block(session_id)
        standalone = self._rewrite(question, history)

        verdict = check_scope(standalone, self.retriever)
        if not verdict.in_scope:
            resp = refusal(session_id, question)
            self.memory.record_exchange(session_id, question, resp["answer"])
            return resp

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. The retrieval/scope layers work "
                "without it, but synthesis needs Claude. Add the key to crew/.env."
            )

        synth, registry = self._run_pipeline(question, standalone)
        resp = self._resolve(session_id, question, synth, registry)
        self.memory.record_exchange(session_id, question, resp["answer"])
        return resp

    # -- agent plumbing ------------------------------------------------------
    def _ensure_agents(self):
        if self._agents is None:
            from .agents import build_agents, build_llm

            self._agents = build_agents(build_llm())
        return self._agents

    @staticmethod
    def _run_agent(agent, description: str, expected_output: str, output_pydantic=None):
        """Run a single agent as a one-task crew and return the CrewAI output."""
        from crewai import Crew, Process, Task

        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            output_pydantic=output_pydantic,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential,
                    memory=False, verbose=False)
        return crew.kickoff()

    @staticmethod
    def _coerce(result, model_cls):
        """Extract a pydantic instance of ``model_cls`` from a CrewAI result."""
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

    # -- pipeline steps ------------------------------------------------------
    def _rewrite(self, question: str, history: str) -> str:
        """MemoryKeeper: resolve a follow-up into a standalone question."""
        if not history or not settings.anthropic_api_key:
            return question
        try:
            agents = self._ensure_agents()
            out = self._run_agent(
                agents["memory_keeper"],
                description=(
                    f"Conversation so far:\n{history}\n\n"
                    f"Latest user question: {question}\n\n"
                    "Rewrite the latest question so it is fully self-contained, "
                    "resolving pronouns/omissions from the conversation. If it is "
                    "already standalone, return it verbatim. Return ONLY the "
                    "rewritten question, with no preamble."
                ),
                expected_output="A single self-contained question.",
            )
            text = (getattr(out, "raw", None) or str(out)).strip()
            return text or question
        except Exception:
            return question

    def _run_pipeline(self, question: str, standalone: str):
        """Retriever ▶ (RAG) ▶ WebSearcher ▶ (Brave) ▶ Synthesizer."""
        from .tools import EvidenceRegistry, search_papers, search_web

        agents = self._ensure_agents()
        registry = EvidenceRegistry()

        # 1. Retriever agent plans focused queries; Python runs the RAG search.
        queries = [standalone]
        plan = self._coerce(
            self._run_agent(
                agents["retriever"],
                description=(
                    f"User question: {standalone}\n\n"
                    "Produce 1-3 focused search queries that will retrieve the "
                    "most relevant passages from the nine dopamine papers. "
                    "Include exact terminology (receptor names, brain regions) "
                    "where useful."
                ),
                expected_output="JSON with a 'queries' list of strings.",
                output_pydantic=_RetrieverPlan,
            ),
            _RetrieverPlan,
        )
        if plan and plan.queries:
            queries = plan.queries[:3]
        paper_evidence = search_papers(self.retriever, registry, queries)

        # 2. WebSearcher agent decides whether to search; Python runs Brave.
        web_evidence = ""
        decision = self._coerce(
            self._run_agent(
                agents["web_searcher"],
                description=(
                    f"User question: {standalone}\n\n"
                    f"Evidence already gathered from the papers:\n{paper_evidence}\n\n"
                    "Decide whether a live web search is needed to answer well — "
                    "a genuine gap in the papers, or a recent/current fact to "
                    "verify. If yes, provide one focused web query."
                ),
                expected_output="JSON: need_web (bool) and query (string).",
                output_pydantic=_WebDecision,
            ),
            _WebDecision,
        )
        if decision and decision.need_web and decision.query.strip():
            web_evidence = search_web(registry, decision.query.strip())

        # 3. Synthesizer agent composes the grounded answer.
        evidence_block = paper_evidence
        if web_evidence and not web_evidence.startswith(_NO_WEB_PREFIXES):
            evidence_block += "\n\n" + web_evidence

        synth = self._coerce(
            self._run_agent(
                agents["synthesizer"],
                description=(
                    f"Original question: {question}\n"
                    f"Standalone question: {standalone}\n\n"
                    f"EVIDENCE (cite these ids):\n{evidence_block}\n\n"
                    "Using ONLY the [P#] and [W#] evidence above, write a clear, "
                    "accurate answer about dopamine. Rules:\n"
                    "- Ground every factual sentence in evidence; cite ids inline "
                    "like [P2] or [W1].\n"
                    "- In `claims`, list each factual sentence with the "
                    "evidence_ids that support it. Omit claims you cannot support.\n"
                    "- Set used_web_search true only if you used [W#] items.\n"
                    "- Set confidence from how well the evidence covers the "
                    "question.\n"
                    "- If the evidence does not cover the question, say so plainly "
                    "rather than inventing facts."
                ),
                expected_output=(
                    "A JSON object: answer, claims (statement + evidence_ids), "
                    "used_web_search, confidence."
                ),
                output_pydantic=_SynthOut,
            ),
            _SynthOut,
        )
        if synth is None:
            synth = _SynthOut(
                answer="I could not produce a grounded answer from the available evidence.",
                claims=[], used_web_search=registry.any_web, confidence="low",
            )
        return synth, registry

    # -- grounding resolution ------------------------------------------------
    @staticmethod
    def _resolve(session_id: str, question: str, synth: _SynthOut, registry) -> CrewResponse:
        """Turn cited evidence ids into a real, deduped ``sources`` list."""
        ordered_ids: List[str] = []

        def _see(eid: str) -> None:
            if eid not in ordered_ids and registry.get(eid) is not None:
                ordered_ids.append(eid)

        for c in synth.claims:
            for eid in c.evidence_ids:
                _see(eid)
        for eid in _EVIDENCE_RE.findall(synth.answer or ""):
            _see(eid)

        sources: List[Source] = [registry.get(eid) for eid in ordered_ids]  # type: ignore[misc]
        id_to_index = {eid: i for i, eid in enumerate(ordered_ids)}

        claims: List[Claim] = []
        for c in synth.claims:
            idxs = [id_to_index[e] for e in c.evidence_ids if e in id_to_index]
            claims.append(Claim(statement=c.statement, source_indices=idxs))

        confidence = synth.confidence if synth.confidence in ("high", "medium", "low") else "medium"

        return CrewResponse(
            session_id=session_id,
            question=question,
            answer=synth.answer.strip(),
            sources=sources,
            claims=claims,
            in_scope=True,
            refused=False,
            used_web_search=registry.any_web,   # ground truth, not the model's word
            confidence=confidence,
        )
