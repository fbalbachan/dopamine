"""The graph crew — CrewAI agents backed by Claude.

Same shape as :mod:`crew.agents`, and for the same reason: these agents are
**tool-free**. CrewAI's ReAct tool executor prefills the assistant turn, which
Claude 4.6+ rejects outright, so the orchestrator runs every graph query in
Python and feeds results back as task context.

That constraint turns out to be the right architecture here anyway. The graph
holds the facts — who is at which institution, who co-authored what, which
recommendation scored highest. If an agent could query it directly, the numbers
in the final answer would be whatever the model transcribed. Because the
orchestrator queries it, the model only ever *narrates* results it was handed,
and :meth:`crew.graph.chat.GraphCrew._resolve` checks every citation against the
evidence registry before it reaches the caller.

Two agents, two judgments:

* **GraphNavigator** reads the question and decides *what to look up* — which
  intent, which names. It never answers.
* **GraphAnalyst** writes the answer from retrieved subgraph evidence, citing
  ``[G#]`` ids. It never decides what is true.
"""
from __future__ import annotations

from typing import Dict

from crewai import LLM, Agent

from ..config import settings


def build_llm() -> LLM:
    """Claude LLM shared by both graph agents.

    No ``temperature``: sampling parameters were removed on Opus 4.7/4.8 and
    return a 400 if sent.
    """
    return LLM(model=settings.litellm_model, api_key=settings.anthropic_api_key)


def build_agents(llm: LLM) -> Dict[str, Agent]:
    navigator = Agent(
        role="Graph Navigator",
        goal=(
            "Read a question about researchers and work out exactly what to "
            "look up in the knowledge graph: the intent behind the question, "
            "and the scientist, topic or institution names it refers to."
        ),
        backstory=(
            "You are a research-network analyst who knows that a vague lookup "
            "returns a vague answer. You extract names exactly as the user "
            "wrote them, and you never guess at an entity the question does "
            "not mention."
        ),
        llm=llm, verbose=False, allow_delegation=False,
    )

    analyst = Agent(
        role="Graph Analyst",
        goal=(
            "Explain what the graph shows about researchers and their "
            "connections, grounding every statement in the [G#] evidence "
            "provided and never asserting a link the graph does not contain."
        ),
        backstory=(
            "You brief people on collaboration networks. You are precise about "
            "the difference between 'these two have published together' and "
            "'these two work on similar problems', because conflating them "
            "produces a useless introduction. When the graph is silent, you "
            "say so rather than filling the gap with plausible-sounding "
            "detail, and you always explain *why* a suggested connection "
            "scored well."
        ),
        llm=llm, verbose=False, allow_delegation=False,
    )

    return {"navigator": navigator, "analyst": analyst}
