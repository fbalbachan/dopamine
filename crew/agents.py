"""The four crew members — CrewAI ``Agent`` objects backed by Claude.

They are tool-free: each contributes *judgment* (query planning, the decision
to search the web, coreference resolution, grounded synthesis), while the
orchestrator executes the actual RAG / Brave tools in Python and feeds the
evidence back as task context. This split is deliberate — CrewAI's ReAct
tool-executor prefills the assistant turn, which Claude 4.8 rejects; keeping the
agents tool-free avoids that and lets the orchestrator guarantee grounding.
"""
from __future__ import annotations

from typing import Dict

from crewai import LLM, Agent

from .config import settings


def build_llm() -> LLM:
    """Claude LLM shared by every agent (via CrewAI's litellm wrapper).

    No ``temperature`` — sampling parameters are removed on Opus 4.7/4.8 (and
    Sonnet 5) and return a 400 if sent.
    """
    return LLM(model=settings.litellm_model, api_key=settings.anthropic_api_key)


def build_agents(llm: LLM) -> Dict[str, Agent]:
    memory_keeper = Agent(
        role="Memory Keeper",
        goal=(
            "Turn a possibly-elliptical follow-up question into a fully "
            "self-contained question using the conversation so far, so the rest "
            "of the crew never has to guess what 'it' or 'that' refers to."
        ),
        backstory=(
            "You are meticulous about conversational context. You resolve "
            "pronouns and elisions against earlier turns, but never invent "
            "intent that isn't there — an already-standalone question is "
            "returned unchanged."
        ),
        llm=llm, verbose=False, allow_delegation=False,
    )

    retriever = Agent(
        role="Retriever",
        goal=(
            "Plan the best searches over a corpus of nine dopamine papers: turn "
            "the user's question into a small set of focused retrieval queries "
            "that will surface the most relevant passages."
        ),
        backstory=(
            "You are a domain librarian for a dopamine research corpus. You know "
            "that precise, well-phrased queries — including exact receptor names "
            "and terminology — retrieve better than a raw question."
        ),
        llm=llm, verbose=False, allow_delegation=False,
    )

    web_searcher = Agent(
        role="Web Searcher",
        goal=(
            "Judge whether the papers already answer the question. If they do, "
            "skip the web. If there's a gap, or the question concerns recent / "
            "current information, craft one focused web query to fill it."
        ),
        backstory=(
            "You are frugal and skeptical. You only reach for the live web when "
            "the papers fall short or a claim needs up-to-date verification."
        ),
        llm=llm, verbose=False, allow_delegation=False,
    )

    synthesizer = Agent(
        role="Synthesizer",
        goal=(
            "Write a correct, well-scoped answer in which every factual claim is "
            "tied to the [P#]/[W#] evidence ids that support it. Never assert "
            "anything you cannot ground in the provided evidence."
        ),
        backstory=(
            "You are a careful science writer. You would rather say 'the "
            "provided sources don't cover this' than fabricate a citation, and "
            "you map each claim to the exact evidence ids behind it."
        ),
        llm=llm, verbose=False, allow_delegation=False,
    )

    return {
        "memory_keeper": memory_keeper,
        "retriever": retriever,
        "web_searcher": web_searcher,
        "synthesizer": synthesizer,
    }
