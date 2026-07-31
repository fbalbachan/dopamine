"""Research-interest taxonomy and topic assignment.

A scientist's "interests" are the topics attached to the work they authored or
were cited for. That needs a *closed* vocabulary — free-form topic strings from
an LLM would give ``reward prediction error``, ``RPE``, ``prediction errors in
reward`` as three unrelated nodes, and no two scientists would ever share one.

So the labour splits the same way it does everywhere else in this project:

* the **LLM derives the taxonomy once** at build time, reading the nine papers
  (:func:`derive_taxonomy`) — that is a judgment call about what this corpus is
  actually about;
* **Python assigns topics** to every paper and cited work
  (:func:`assign_topics`) — deterministic, free, and identical on every rebuild.

Assignment is hybrid, mirroring the retrieval layer's dense+lexical split: an
alias hit (``D2``, ``L-DOPA``, ``6-OHDA``) is exact evidence and scores 1.0,
while everything else falls back to cosine similarity between the title and the
topic description. Exact receptor and drug tokens are precisely where embeddings
are weakest, which is why the lexical half exists here too.

``SEED_TAXONOMY`` is the offline fallback so ``python -m crew.graph.build`` runs
with no API key at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import settings
from .canon import canonical_topic

# Curated fallback taxonomy: used when no API key is configured, and as the
# few-shot anchor for the LLM-derived version. Aliases are matched case-
# insensitively against word boundaries, so they must be unambiguous tokens.
SEED_TAXONOMY: List[dict] = [
    {
        "label": "Reward prediction error",
        "description": "Phasic dopamine signalling of the difference between "
                       "predicted and received reward; temporal-difference learning.",
        "aliases": ["reward prediction error", "prediction error", "RPE",
                    "temporal difference", "reward learning"],
    },
    {
        "label": "Motivation and incentive salience",
        "description": "Dopamine's role in motivation, effort, wanting, "
                       "incentive salience and vigour of behaviour.",
        "aliases": ["incentive salience", "motivation", "wanting", "effort",
                    "vigour", "vigor"],
    },
    {
        "label": "Dopamine receptors",
        "description": "D1-like and D2-like receptor structure, subtypes, "
                       "pharmacology and signal transduction.",
        "aliases": ["D1", "D2", "D3", "D4", "D5", "dopamine receptor",
                    "D1-like", "D2-like", "receptor subtype"],
    },
    {
        "label": "Parkinson's disease",
        "description": "Degeneration of nigrostriatal dopamine neurons, motor "
                       "symptoms, and the pathophysiology of Parkinson's disease.",
        "aliases": ["Parkinson", "Parkinson's", "PD", "nigrostriatal degeneration",
                    "MPTP", "6-OHDA", "alpha-synuclein"],
    },
    {
        "label": "Levodopa and PD therapeutics",
        "description": "L-DOPA, dopamine agonists, deep brain stimulation and "
                       "other treatments for dopamine-related movement disorders.",
        "aliases": ["L-DOPA", "levodopa", "dopamine agonist", "deep brain stimulation",
                    "DBS", "dyskinesia"],
    },
    {
        "label": "ADHD",
        "description": "Attention-deficit hyperactivity disorder, its dopaminergic "
                       "basis, candidate genes and stimulant treatment.",
        "aliases": ["ADHD", "attention deficit", "hyperactivity", "methylphenidate"],
    },
    {
        "label": "Addiction and drugs of abuse",
        "description": "Cocaine, amphetamine, nicotine and opioid actions on "
                       "mesolimbic dopamine; self-administration and relapse.",
        "aliases": ["addiction", "cocaine", "amphetamine", "self-administration",
                    "drug abuse", "nicotine", "relapse", "substance use"],
    },
    {
        "label": "Schizophrenia and antipsychotics",
        "description": "The dopamine hypothesis of schizophrenia, neuroleptics "
                       "and antipsychotic receptor blockade.",
        "aliases": ["schizophrenia", "antipsychotic", "neuroleptic", "haloperidol",
                    "psychosis", "clozapine"],
    },
    {
        "label": "Dopamine synthesis and metabolism",
        "description": "Tyrosine hydroxylase, DOPA decarboxylase, MAO/COMT "
                       "catabolism, storage, and oxidative dopamine metabolites.",
        "aliases": ["tyrosine hydroxylase", "MAO", "COMT", "metabolism",
                    "biosynthesis", "oxidative stress", "dopamine quinone"],
    },
    {
        "label": "Dopamine transporter and release",
        "description": "DAT and VMAT2 transport, vesicular storage, reuptake, "
                       "and measurement of dopamine release.",
        "aliases": ["dopamine transporter", "DAT", "VMAT", "reuptake",
                    "voltammetry", "microdialysis", "dopamine release"],
    },
    {
        "label": "Striatal circuits and basal ganglia",
        "description": "Striatum, nucleus accumbens, direct and indirect "
                       "pathways, and basal ganglia circuit function.",
        "aliases": ["striatum", "striatal", "nucleus accumbens", "basal ganglia",
                    "medium spiny neuron", "globus pallidus", "putamen"],
    },
    {
        "label": "Midbrain dopamine neurons",
        "description": "Substantia nigra pars compacta and ventral tegmental "
                       "area dopamine cell physiology and firing patterns.",
        "aliases": ["substantia nigra", "ventral tegmental", "VTA", "SNc",
                    "midbrain dopamine", "burst firing", "tonic", "phasic"],
    },
    {
        "label": "Molecular genetics of dopamine",
        "description": "Gene structure, polymorphisms and knockout models of "
                       "dopamine receptors and transporters.",
        "aliases": ["polymorphism", "knockout", "allele", "genotype", "DRD2",
                    "DRD4", "SLC6A3", "gene structure"],
    },
    {
        "label": "Behavioural neuroscience methods",
        "description": "Electrophysiology, optogenetics, lesion studies and "
                       "behavioural task design in dopamine research.",
        "aliases": ["electrophysiology", "optogenetic", "lesion", "single-unit",
                    "conditioning", "operant"],
    },
    {
        "label": "Well-being and dopamine fasting",
        "description": "Lifestyle, holistic well-being, dopamine fasting and "
                       "behavioural-addiction framings of dopamine.",
        "aliases": ["dopamine fasting", "well-being", "wellbeing", "mindfulness",
                    "lifestyle", "screen time"],
    },
]


class Taxonomy:
    """The closed topic vocabulary plus the machinery to assign topics to text."""

    def __init__(self, topics: List[dict]):
        self.topics: List[dict] = []
        for t in topics:
            label = (t.get("label") or "").strip()
            if not label:
                continue
            self.topics.append(
                {
                    "id": canonical_topic(label),
                    "label": label,
                    "description": (t.get("description") or label).strip(),
                    "aliases": [a for a in (t.get("aliases") or []) if a.strip()],
                }
            )
        self._matrix: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.topics)

    @property
    def ids(self) -> List[str]:
        return [t["id"] for t in self.topics]

    def label(self, topic_id: str) -> str:
        for t in self.topics:
            if t["id"] == topic_id:
                return t["label"]
        return topic_id

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Taxonomy":
        path = path or (settings.graph_dir / "taxonomy.json")
        if path.exists():
            return cls(json.loads(path.read_text(encoding="utf-8")))
        return cls(SEED_TAXONOMY)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or (settings.graph_dir / "taxonomy.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.topics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- assignment ----------------------------------------------------------
    def _embed_matrix(self, embed) -> np.ndarray:
        """Embed every topic once; cached for the life of the object."""
        if self._matrix is None:
            texts = [f"{t['label']}. {t['description']}" for t in self.topics]
            self._matrix = embed(texts)
        return self._matrix

    def assign(self, texts: Sequence[str], embed, threshold: Optional[float] = None,
               max_topics: int = 3) -> List[List[Tuple[str, float]]]:
        """Assign topics to each text. Returns per-text ``[(topic_id, score)]``.

        ``embed`` is a callable ``List[str] -> np.ndarray`` of L2-normalised
        row vectors — normally the shared sentence-transformers model, so the
        graph and the RAG layer use one embedder and one download.

        Selection is **relative, not absolute**. Every topic in this taxonomy is
        a kind of dopamine research, so every topic description embeds close to
        every dopamine paper and a fixed cosine cutoff admits all of them — an
        early build tagged the Parkinson's review as "Dopamine in
        Schizophrenia" for exactly this reason. Standardising each text's
        similarities into z-scores subtracts that shared "this is about
        dopamine" component and leaves only what actually discriminates.
        An absolute floor is still applied so a wholly unrelated title cannot
        be dragged in just by being the least-bad match.
        """
        floor = settings.graph_topic_threshold if threshold is None else threshold
        if not texts or not self.topics:
            return [[] for _ in texts]

        matrix = self._embed_matrix(embed)
        vectors = embed(list(texts))
        sims = vectors @ matrix.T          # cosine: both sides are normalised

        out: List[List[Tuple[str, float]]] = []
        for row, text in zip(sims, texts):
            mean = float(row.mean())
            std = float(row.std()) or 1e-6
            scored: Dict[str, float] = {}
            lowered = f" {text.lower()} "
            for j, topic in enumerate(self.topics):
                cosine = float(row[j])
                # An exact alias hit is direct evidence, not a similarity guess.
                if any(_alias_hit(lowered, a) for a in topic["aliases"]):
                    scored[topic["id"]] = 1.0
                    continue
                z = (cosine - mean) / std
                if z >= settings.graph_topic_z and cosine >= floor:
                    # Map the z-score into (0, 1) so it stays comparable with
                    # the 1.0 an alias hit scores.
                    scored[topic["id"]] = round(min(0.5 + z / 8.0, 0.99), 4)
            ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
            out.append(ranked[:max_topics])
        return out


def _alias_hit(haystack_lowered: str, alias: str) -> bool:
    """Word-boundary containment test.

    Padding both sides with spaces and testing on non-alphanumeric boundaries
    stops ``D2`` matching inside ``D20`` and ``PD`` matching inside ``PDF``.
    """
    needle = alias.lower().strip()
    if not needle:
        return False
    start = haystack_lowered.find(needle)
    while start != -1:
        before = haystack_lowered[start - 1] if start else " "
        after_i = start + len(needle)
        after = haystack_lowered[after_i] if after_i < len(haystack_lowered) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        start = haystack_lowered.find(needle, start + 1)
    return False


def derive_taxonomy(paper_summaries: List[str]) -> Taxonomy:
    """Ask the extraction model to build the topic vocabulary for this corpus.

    Falls back to :data:`SEED_TAXONOMY` when there is no API key or the call
    fails, so the build never hard-depends on Claude for this step.
    """
    if not settings.anthropic_api_key:
        return Taxonomy(SEED_TAXONOMY)
    try:
        from litellm import completion

        joined = "\n\n---\n\n".join(s[:1500] for s in paper_summaries)
        resp = completion(
            model=settings.litellm_graph_extract_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You build controlled vocabularies for bibliometric "
                        "knowledge graphs. Return ONLY a JSON array."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Below are excerpts from nine dopamine research papers.\n\n"
                        f"{joined}\n\n"
                        "Produce 12-20 research topics that together cover this "
                        "corpus and would work as *research interests* on a "
                        "scientist's profile. Each entry: "
                        '{"label": str, "description": one sentence, '
                        '"aliases": [exact terms/acronyms that reliably indicate '
                        "the topic]}. Topics must be mutually distinguishable and "
                        "neither too broad ('neuroscience') nor too narrow "
                        "('figure 3 methodology'). Return ONLY the JSON array."
                    ),
                },
            ],
            max_tokens=4000,
            temperature=0,
        )
        raw = resp["choices"][0]["message"]["content"]
        start, end = raw.find("["), raw.rfind("]")
        parsed = json.loads(raw[start : end + 1])
        taxonomy = Taxonomy(parsed)
        # A degenerate result is worse than the curated seed.
        return taxonomy if len(taxonomy) >= 8 else Taxonomy(SEED_TAXONOMY)
    except Exception:
        return Taxonomy(SEED_TAXONOMY)
