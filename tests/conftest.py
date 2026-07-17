"""Shared fixtures.

Unit tests (schema, scope logic, grounding resolver, memory, config) need no
index and no API key. Integration tests need the RAG index; they skip cleanly
if it hasn't been built (`python -m crew.build_index`).
"""
from __future__ import annotations

import pytest

from crew.memory import MemoryStore


@pytest.fixture(scope="session")
def retriever():
    """The real retriever over the 9 papers; skip if the index isn't built."""
    from crew.rag import Retriever

    try:
        return Retriever.load()
    except Exception as exc:  # index not built
        pytest.skip(f"RAG index not available ({exc}); run `python -m crew.build_index`")


@pytest.fixture
def tmp_memory(tmp_path):
    """A MemoryStore backed by a throwaway SQLite file."""
    store = MemoryStore(db_path=tmp_path / "sessions.sqlite3")
    yield store
    store.close()
