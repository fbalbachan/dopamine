"""Conversational memory: ordering, session isolation, and the bounded window."""
from __future__ import annotations

from crew.memory import MemoryStore


def test_record_exchange_ordering(tmp_memory: MemoryStore):
    tmp_memory.record_exchange("s1", "q1", "a1")
    tmp_memory.record_exchange("s1", "q2", "a2")
    turns = tmp_memory.history("s1")
    assert [(t.role, t.content) for t in turns] == [
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q2"), ("assistant", "a2"),
    ]


def test_context_block_renders_turns(tmp_memory: MemoryStore):
    tmp_memory.record_exchange("s1", "what is dopamine?", "a neurotransmitter")
    block = tmp_memory.context_block("s1")
    assert "User: what is dopamine?" in block
    assert "Assistant: a neurotransmitter" in block


def test_sessions_are_isolated(tmp_memory: MemoryStore):
    tmp_memory.record_exchange("s1", "q-a", "ans-a")
    tmp_memory.record_exchange("s2", "q-b", "ans-b")
    assert len(tmp_memory.history("s1")) == 2
    assert all("q-b" not in t.content for t in tmp_memory.history("s1"))


def test_history_window_is_bounded(tmp_path):
    store = MemoryStore(max_turns=4, db_path=tmp_path / "m.sqlite3")
    for i in range(5):  # 10 messages total
        store.record_exchange("s", f"q{i}", f"a{i}")
    turns = store.history("s")
    assert len(turns) == 4                      # only the most recent window
    assert turns[-1].content == "a4"            # newest last
    store.close()


def test_clear_removes_session(tmp_memory: MemoryStore):
    tmp_memory.record_exchange("s1", "q", "a")
    tmp_memory.clear("s1")
    assert tmp_memory.history("s1") == []


def test_empty_context_block_is_empty_string(tmp_memory: MemoryStore):
    assert tmp_memory.context_block("never-seen") == ""
