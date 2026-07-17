"""Configuration invariants."""
from __future__ import annotations

import os

from crew.config import PAPER_FILENAMES, settings


def test_nine_papers_pinned():
    assert len(PAPER_FILENAMES) == 9
    assert len(set(PAPER_FILENAMES)) == 9
    assert all(name.endswith(".pdf") for name in PAPER_FILENAMES)


def test_double_space_filename_is_exact():
    # The Marsden paper's repo filename has a double space before "the".
    assert (
        "British J Pharmacology - 2009 - Marsden - Dopamine  the rewarding years.pdf"
        in PAPER_FILENAMES
    )


def test_litellm_model_prefix():
    assert settings.litellm_model == f"anthropic/{settings.agent_model}"


def test_scope_threshold_is_float():
    assert isinstance(settings.scope_threshold, float)
    assert 0.0 < settings.scope_threshold < 1.0


def test_implicit_hf_token_disabled():
    # Importing crew.config sets this so a stale stored HF token can't 401
    # public model downloads.
    assert os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1"
