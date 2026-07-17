"""Fully-typed response contract for the crew.

The crew always returns a ``CrewResponse`` — a plain ``dict`` at runtime that
is ``json.dumps``-able and validates against these ``TypedDict`` definitions.

Traceability model
-------------------
Every factual statement in ``answer`` is also emitted as a ``Claim`` whose
``source_indices`` point into the ``sources`` list. A claim with an empty
``source_indices`` is, by contract, not allowed for in-scope answers — the
synthesizer must ground each claim in at least one paper page or web URL.
"""
from __future__ import annotations

from typing import List, Literal, TypedDict, Union


class PaperSource(TypedDict):
    """A citation into one of the nine papers, resolved to a page."""
    type: Literal["paper"]
    file: str          # e.g. "da_reward_learning_wise.pdf"
    page: int          # 1-indexed page number within that PDF
    snippet: str       # the supporting text pulled from that page


class WebSource(TypedDict):
    """A citation to a live web result (Brave search / fetched page)."""
    type: Literal["web"]
    url: str
    title: str
    snippet: str


# A source is either a paper page or a web URL — discriminated on ``type``.
Source = Union[PaperSource, WebSource]


class Claim(TypedDict):
    """One factual statement mapped to the sources that support it."""
    statement: str
    source_indices: List[int]   # indices into CrewResponse["sources"]


class CrewResponse(TypedDict):
    """The single dict the crew returns for every question."""
    session_id: str
    question: str
    answer: str
    sources: List[Source]
    claims: List[Claim]
    in_scope: bool
    refused: bool
    used_web_search: bool
    confidence: Literal["high", "medium", "low"]


# --- Constructors / helpers (keep call sites honest and consistent) ---

REFUSAL_TEXT = (
    "I'm scoped to answer questions about dopamine — its biology, receptors, "
    "signaling, and roles in reward, movement, and disorders such as "
    "Parkinson's and ADHD — grounded in the challenge's nine papers and the "
    "web. That question falls outside this scope, so I can't answer it here."
)


def refusal(session_id: str, question: str) -> CrewResponse:
    """Build the standard out-of-scope response: flagged, empty sources."""
    return CrewResponse(
        session_id=session_id,
        question=question,
        answer=REFUSAL_TEXT,
        sources=[],
        claims=[],
        in_scope=False,
        refused=True,
        used_web_search=False,
        confidence="high",
    )
