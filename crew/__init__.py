"""Dopamine-domain Q&A crew.

A modular CrewAI crew that answers questions about dopamine by combining
RAG over nine papers, live Brave web search, source-aware synthesis, and
per-session conversational memory.

Public entry point:

    from crew import DopamineCrew
    crew = DopamineCrew()
    response = crew.ask("What is dopamine's role in reward learning?", session_id="s1")
"""
from .schema import CrewResponse, Source, PaperSource, WebSource, Claim
from .chat import DopamineCrew

__all__ = [
    "DopamineCrew",
    "CrewResponse",
    "Source",
    "PaperSource",
    "WebSource",
    "Claim",
]
