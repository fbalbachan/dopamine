"""Topic assignment: the closed vocabulary and how texts are matched to it."""
from __future__ import annotations

import numpy as np
import pytest

from crew.graph.topics import SEED_TAXONOMY, Taxonomy, _alias_hit


@pytest.fixture
def taxonomy():
    return Taxonomy(SEED_TAXONOMY)


def _fake_embed(vectors_by_text):
    """Embedder returning fixed, pre-normalised vectors — no model download."""
    def embed(texts):
        out = []
        for t in texts:
            v = np.array(vectors_by_text.get(t, [1.0, 0.0, 0.0]), dtype="float32")
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.vstack(out).astype("float32")
    return embed


def test_alias_hit_respects_word_boundaries():
    """'D2' must not match inside 'D20', and 'PD' not inside 'PDF'."""
    assert _alias_hit(" the d2 receptor ", "D2")
    assert not _alias_hit(" the d20 receptor ", "D2")
    assert _alias_hit(" studies of pd patients ", "PD")
    assert not _alias_hit(" open the pdf file ", "PD")


def test_alias_hit_matches_hyphenated_tokens():
    assert _alias_hit(" treated with l-dopa daily ", "L-DOPA")


def test_topic_ids_are_slugs(taxonomy):
    assert "reward-prediction-error" in taxonomy.ids
    assert all(" " not in tid for tid in taxonomy.ids)


def test_taxonomy_round_trips_to_disk(taxonomy, tmp_path):
    path = tmp_path / "taxonomy.json"
    taxonomy.save(path)
    reloaded = Taxonomy.load(path)
    assert reloaded.ids == taxonomy.ids


def test_taxonomy_load_falls_back_to_the_seed(tmp_path):
    """No API key and no cached file must still yield a usable vocabulary."""
    assert len(Taxonomy.load(tmp_path / "missing.json")) == len(SEED_TAXONOMY)


def test_an_alias_hit_scores_maximum(taxonomy):
    """Exact terminology is evidence, not a similarity guess."""
    embed = _fake_embed({})
    assigned = taxonomy.assign(["A study of L-DOPA in patients"], embed)
    scores = dict(assigned[0])
    assert scores.get("levodopa-and-pd-therapeutics") == 1.0


def test_uniformly_similar_topics_are_not_all_assigned(taxonomy):
    """Every topic here is 'about dopamine', so a flat similarity profile must
    select nothing rather than everything — the bug that once filed the
    Parkinson's review under schizophrenia."""
    flat = _fake_embed({})    # identical vector for text and every topic
    assigned = taxonomy.assign(["something with no exact terms in it"], flat)
    assert assigned[0] == []


def test_max_topics_is_respected(taxonomy):
    text = "Parkinson's disease, L-DOPA, ADHD, cocaine, schizophrenia and D2"
    assigned = taxonomy.assign([text], _fake_embed({}), max_topics=2)
    assert len(assigned[0]) <= 2


def test_empty_input_is_handled(taxonomy):
    assert taxonomy.assign([], _fake_embed({})) == []


def test_seed_taxonomy_aliases_are_unambiguous():
    """A one- or two-character alias would match half the corpus."""
    for topic in SEED_TAXONOMY:
        for alias in topic["aliases"]:
            assert len(alias) >= 2, f"{topic['label']}: alias {alias!r} too short"
