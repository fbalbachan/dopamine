"""Typed contract for the scientist knowledge graph.

Everything here is a ``TypedDict``, so instances are plain dicts at runtime and
``json.dumps`` works with no custom encoder — the same choice the Q&A crew makes
in :mod:`crew.schema`, and what lets the graph persist to a human-readable JSON
file and ship straight into the HTML viewer.

Provenance model
----------------
Every node and every edge carries at least one :class:`Evidence` entry naming
the PDF and 1-indexed page it was extracted from. Nothing enters the graph
without one. That is what lets a recommendation be defended ("these two share
*reward prediction error*, asserted on p.3 of Berke and p.2 of Schultz") rather
than merely asserted, and it mirrors the ``file + page`` guarantee the retrieval
layer already provides.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

# Node kinds present in the graph.
NodeKind = Literal["scientist", "institution", "topic", "paper"]

# Edge relations. Directed unless noted.
#   AUTHORED         scientist -> paper
#   AFFILIATED_WITH  scientist -> institution
#   CO_AUTHORED      scientist -> scientist   (symmetric; stored both ways)
#   WORKS_ON         scientist -> topic
#   COVERS           paper     -> topic
#   CITES            paper     -> paper
Relation = Literal[
    "AUTHORED", "AFFILIATED_WITH", "CO_AUTHORED", "WORKS_ON", "COVERS", "CITES"
]


class Evidence(TypedDict):
    """Where in the corpus a node or edge was observed."""
    file: str          # local PDF filename, e.g. "nihms-987662.pdf"
    page: int          # 1-indexed page within that PDF
    snippet: str       # the supporting text, capped at 300 chars


class ExternalProfile(TypedDict, total=False):
    """Optional enrichment from an external scholarly source (OpenAlex).

    Absent unless the graph was built with ``--enrich``; see
    :mod:`crew.graph.enrich` for why OpenAlex rather than Google Scholar.
    """
    source: str
    openalex_id: str
    orcid: str
    display_name: str
    works_count: int
    cited_by_count: int
    h_index: int
    institutions: List[dict]
    topics: List[str]


class _ScientistBase(TypedDict):
    """A researcher: either an author of one of the nine papers, or someone
    cited by them."""
    id: str                      # canonical key, e.g. "schultz|w"
    name: str                    # best display form observed
    aliases: List[str]           # every surface form seen ("Schultz W", "W. Schultz")
    institutions: List[str]      # institution ids (corpus authors only, usually)
    topics: List[str]            # topic ids, ordered by strength
    corpus_author: bool          # True if they authored one of the nine papers
    papers: List[str]            # paper ids they authored
    cited_works: int             # how many referenced works they appear on
    degree: int                  # number of distinct co-authors
    evidence: List[Evidence]


class Scientist(_ScientistBase, total=False):
    """A researcher, optionally decorated with an external profile."""
    external: ExternalProfile


class Institution(TypedDict):
    """A university, hospital, institute or company."""
    id: str
    name: str                    # canonical organisation name
    departments: List[str]
    country: str                 # "" when unknown
    scientists: List[str]        # scientist ids
    evidence: List[Evidence]


class Topic(TypedDict):
    """A research interest, drawn from an LLM-derived taxonomy over the corpus."""
    id: str
    label: str
    description: str             # one line; also the text embedded for matching
    aliases: List[str]           # exact tokens ("D2", "L-DOPA") for lexical matching
    scientists: List[str]
    papers: List[str]


class Paper(TypedDict):
    """A publication: one of the nine, or a work cited by them."""
    id: str
    title: str
    year: Optional[int]
    venue: str
    doi: str
    authors: List[str]           # scientist ids, in author order
    topics: List[str]
    in_corpus: bool              # True for the nine indexed PDFs
    file: str                    # local PDF filename when in_corpus, else ""
    mentions: int                # times cited across the corpus
    evidence: List[Evidence]


class Edge(TypedDict):
    """A typed, evidenced relation between two nodes."""
    source: str
    target: str
    relation: Relation
    weight: float
    evidence: List[Evidence]


class GraphStats(TypedDict):
    """Summary counts, written at build time and shown by the CLI/viewer."""
    scientists: int
    corpus_authors: int
    institutions: int
    topics: int
    papers: int
    corpus_papers: int
    edges: int
    relations: Dict[str, int]
    built_from: List[str]        # the PDF filenames the graph was built over
    corpus_hash: str


class Signals(TypedDict):
    """The per-signal breakdown behind a recommendation score.

    Each value is already normalised to [0, 1] *before* weighting, so the
    numbers are directly comparable to one another.
    """
    topic_overlap: float         # weighted Jaccard over research interests
    profile_similarity: float    # cosine between profile-text embeddings
    graph_proximity: float       # Adamic-Adar over the co-authorship network
    citation_affinity: float     # do their citation neighbourhoods overlap
    same_institution: bool       # discounted, not excluded


class Recommendation(TypedDict):
    """One suggested connection, with the evidence needed to justify it."""
    scientist_id: str
    name: str
    score: float                 # blended, [0, 1]
    signals: Signals
    shared_topics: List[str]     # topic labels, strongest first
    shared_collaborators: List[str]   # scientist names bridging the two
    institutions: List[str]
    rationale: str               # plain-language, generated from the fields above
    evidence: List[Evidence]


class Profile(TypedDict):
    """Everything the graph knows about one scientist."""
    scientist: Scientist
    institutions: List[Institution]
    topics: List[Topic]
    papers: List[Paper]
    collaborators: List[str]     # names
    recommendations: List[Recommendation]


class GraphAnswer(TypedDict):
    """The dict :meth:`crew.graph.chat.GraphCrew.ask` always returns.

    Deliberately shaped like :class:`crew.schema.CrewResponse` — same
    ``sources``/``claims`` traceability contract — so the two surfaces can be
    consumed by the same client code.
    """
    session_id: str
    question: str
    answer: str
    intent: str                  # profile | connections | recommend | topic | institution | overview
    entities: List[str]          # graph node ids the question resolved to
    sources: List[Evidence]
    claims: List[dict]           # {"statement": str, "source_indices": List[int]}
    recommendations: List[Recommendation]
    in_scope: bool
    refused: bool
    confidence: Literal["high", "medium", "low"]


# --- Constructors ---------------------------------------------------------- #

REFUSAL_TEXT = (
    "I answer questions about the researchers behind the nine dopamine papers — "
    "their institutions, research interests, collaborators, and who they might "
    "usefully connect with. That question falls outside this graph, so I can't "
    "answer it here."
)


def graph_refusal(session_id: str, question: str) -> GraphAnswer:
    """The standard out-of-scope answer: flagged, with nothing asserted."""
    return GraphAnswer(
        session_id=session_id,
        question=question,
        answer=REFUSAL_TEXT,
        intent="refused",
        entities=[],
        sources=[],
        claims=[],
        recommendations=[],
        in_scope=False,
        refused=True,
        confidence="high",
    )


def evidence(file: str, page: int, snippet: str) -> Evidence:
    """Build an Evidence entry with the snippet capped to keep the graph small."""
    snippet = " ".join((snippet or "").split())
    return Evidence(
        file=file,
        page=int(page),
        snippet=snippet if len(snippet) <= 300 else snippet[:300] + "…",
    )
