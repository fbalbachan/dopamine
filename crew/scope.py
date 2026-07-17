"""Out-of-scope guardrail.

Two cheap layers decide whether a question belongs to the dopamine domain
*before* we spend agent tokens or a web search:

1. **Retrieval floor** — the best cosine similarity between the question and
   any paper chunk. Far below the floor -> out; comfortably above -> in.
2. **LLM tie-breaker** — only for the narrow "gray band" around the floor, a
   cheap Claude Haiku call makes the call. This keeps the common cases free
   while still catching phrasings the embedder misjudges.

An out-of-scope question yields the standard refusal (flagged, empty sources)
in :mod:`crew.schema`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import settings

# Cheap model for the tie-breaker only (not the main agent model).
GUARD_MODEL = "anthropic/claude-haiku-4-5"

# Half-width of the gray band around the similarity floor.
GRAY_BAND = 0.06

# Obvious in-domain vocabulary. A strong lexical hit rescues questions the
# embedder underscores (e.g. a terse "What does D2 do?").
DOPAMINE_LEXICON = {
    "dopamine", "dopaminergic", "d1", "d2", "d3", "d4", "d5",
    "receptor", "reward", "reinforcement", "substantia nigra", "striatum",
    "parkinson", "parkinson's", "adhd", "l-dopa", "levodopa", "vta",
    "ventral tegmental", "nucleus accumbens", "prediction error", "tonic",
    "phasic", "neurotransmitter", "mesolimbic", "nigrostriatal", "catecholamine",
}


@dataclass
class ScopeVerdict:
    in_scope: bool
    similarity: float
    reason: str


def _lexicon_hit(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in DOPAMINE_LEXICON)


def _llm_tiebreak(question: str) -> bool:
    """Ask a cheap model whether the question is about dopamine.

    Returns True on any failure (fail-open to the retrieval signal is handled
    by the caller; here we simply default to in-scope so we don't wrongly
    refuse a borderline-but-valid question because of a transient error).
    """
    if not settings.anthropic_api_key:
        return True
    try:
        from litellm import completion

        resp = completion(
            model=GUARD_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scope classifier for a dopamine-neuroscience "
                        "Q&A assistant. Answer with exactly one word: YES if the "
                        "question is about dopamine (its biology, receptors, "
                        "signaling, reward/movement roles, or disorders like "
                        "Parkinson's or ADHD), otherwise NO."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=4,
            temperature=0,
        )
        answer = resp["choices"][0]["message"]["content"].strip().upper()
        return answer.startswith("Y")
    except Exception:
        return True


def check_scope(question: str, retriever) -> ScopeVerdict:
    """Decide whether ``question`` is in the dopamine domain.

    ``retriever`` is a :class:`crew.rag.Retriever`; we only use its
    ``max_similarity`` method so this module stays decoupled from indexing.
    """
    sim = retriever.max_similarity(question)
    floor = settings.scope_threshold

    if sim >= floor + GRAY_BAND:
        return ScopeVerdict(True, sim, "high paper similarity")

    if sim < floor - GRAY_BAND:
        # Rescue terse but clearly in-domain questions on lexicon alone.
        if _lexicon_hit(question):
            return ScopeVerdict(True, sim, "low similarity but in-domain vocabulary")
        return ScopeVerdict(False, sim, "far below retrieval floor")

    # Gray band: spend one cheap Haiku call.
    if _lexicon_hit(question) or _llm_tiebreak(question):
        return ScopeVerdict(True, sim, "gray band resolved in-scope")
    return ScopeVerdict(False, sim, "gray band resolved out-of-scope")
