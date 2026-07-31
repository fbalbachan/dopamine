"""The shared local embedder.

One module so every consumer — the build, the recommender, the CLI and the
GraphRAG chat — uses the *same* model. That matters twice over: profile vectors
are only comparable when produced by one model, and the several-hundred-megabyte
sentence-transformers download should happen once per machine and load once per
process.

It also removes a quiet failure mode. :class:`crew.graph.recommend.Recommender`
accepts ``embed=None`` and degrades by scoring profile similarity as zero for
everyone — which looks like "nobody is similar" rather than "this signal is
switched off". Callers that want scoring should take the embedder from here
rather than passing nothing.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..config import settings

_EMBEDDER: Optional[Callable[[List[str]], np.ndarray]] = None


def default_embedder() -> Callable[[List[str]], np.ndarray]:
    """Return the process-wide embedder, loading the model on first use.

    The returned callable takes a list of strings and returns L2-normalised
    float32 row vectors, so an inner product is a cosine similarity.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embed_model)

        def embed(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, model.get_sentence_embedding_dimension()),
                                dtype="float32")
            return model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=64,
            ).astype("float32")

        _EMBEDDER = embed
    return _EMBEDDER
