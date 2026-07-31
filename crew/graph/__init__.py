"""GraphRAG over the people behind the nine dopamine papers.

Where :mod:`crew` answers *what the papers say*, this package answers *who wrote
them, what they work on, and who should meet whom*. It extracts scientists,
institutions, research interests and citation links out of the same PDFs, builds
a typed knowledge graph, and recommends connections from it.

    from crew.graph import ScientistGraph, recommend_connections

    graph = ScientistGraph.load()
    for rec in recommend_connections(graph, "schultz|w"):
        print(rec["name"], rec["score"], rec["rationale"])

Entry points:

    python -m crew.graph.build       # build the graph + HTML viewer
    python -m crew.graph.interface   # explore it (CLI / REPL)
"""
from .recommend import Recommender, recommend_connections, similar_scientists
from .schema import (
    Edge,
    Evidence,
    GraphAnswer,
    Institution,
    Paper,
    Profile,
    Recommendation,
    Scientist,
    Topic,
)
from .store import ScientistGraph

__all__ = [
    "ScientistGraph",
    "Recommender",
    "recommend_connections",
    "similar_scientists",
    "Scientist",
    "Institution",
    "Topic",
    "Paper",
    "Edge",
    "Evidence",
    "Recommendation",
    "Profile",
    "GraphAnswer",
]
