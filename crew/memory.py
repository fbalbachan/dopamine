"""Conversational memory keyed by ``session_id``.

Backed by SQLite so a session survives across process restarts. Two things
are stored per turn: the user question and the crew's answer. On a follow-up,
:meth:`MemoryStore.context_block` renders the recent turns into a compact text
block that is injected into the Retriever/Synthesizer prompts so pronouns and
elisions ("what about in Parkinson's?", "does it also do that?") resolve
against earlier turns.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import settings


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    content: str
    ts: float


class MemoryStore:
    def __init__(self, max_turns: int = 8, db_path: Optional[Path] = None):
        # max_turns = how many recent messages to surface for follow-up context.
        # db_path lets tests point at a throwaway database.
        self.max_turns = max_turns
        if db_path is None:
            settings.ensure_dirs()
            db_path = settings.memory_db
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                ts         REAL    NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON turns(session_id, id)"
        )
        self._conn.commit()

    # -- writes --------------------------------------------------------------
    def add(self, session_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO turns (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        self._conn.commit()

    def record_exchange(self, session_id: str, question: str, answer: str) -> None:
        self.add(session_id, "user", question)
        self.add(session_id, "assistant", answer)

    # -- reads ---------------------------------------------------------------
    def history(self, session_id: str, limit: int | None = None) -> List[Turn]:
        limit = limit or self.max_turns
        rows = self._conn.execute(
            "SELECT role, content, ts FROM turns WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [Turn(role=r, content=c, ts=t) for r, c, t in reversed(rows)]

    def context_block(self, session_id: str) -> str:
        """Render recent turns for prompt injection (empty if no history)."""
        turns = self.history(session_id)
        if not turns:
            return ""
        lines = []
        for t in turns:
            who = "User" if t.role == "user" else "Assistant"
            snippet = t.content if len(t.content) <= 500 else t.content[:500] + "…"
            lines.append(f"{who}: {snippet}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
