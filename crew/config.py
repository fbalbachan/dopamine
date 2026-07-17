"""Central configuration for the dopamine-Q&A crew.

Everything tunable lives here and is read from ``crew/.env`` (git-ignored).
Import ``settings`` anywhere; it is a module-level singleton.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# We only ever download *public* Hugging Face models (the embedder). If the
# environment has a stale/expired stored HF token, the hub sends it and public
# downloads 401. Disabling the implicit token forces anonymous access. A user
# who genuinely needs a token can still set one explicitly via HF_TOKEN.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# CrewAI ships an interactive "view execution traces?" prompt and background
# telemetry. Both hang or add noise in non-interactive/CI runs — disable them.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# crew/ package root, e.g. C:\RootStrap\crew
PACKAGE_ROOT = Path(__file__).resolve().parent

# Load crew/.env explicitly so the app works regardless of CWD.
load_dotenv(PACKAGE_ROOT / ".env")

# The nine papers that define the knowledge base (dopamine domain).
# Source repo: https://github.com/rootstrap/ai-agents-challenge-papers (branch: main)
PAPERS_REPO_RAW = (
    "https://raw.githubusercontent.com/rootstrap/ai-agents-challenge-papers/main"
)
PAPER_FILENAMES = [
    "20240725-319105-k9hkuf.pdf",
    "Behavioral_dopamine_signals.pdf",
    # NB: exact repo filename has a double space before "the".
    "British J Pharmacology - 2009 - Marsden - Dopamine  the rewarding years.pdf",
    "Dopamine_in_Parkinsons_disease.pdf",
    "Role_of_Dopamine_Receptors_in_ADHD_A_Systematic_Me.pdf",
    "da_reward_learning_wise.pdf",
    "missale-et-al-1998-dopamine-receptors-from-structure-to-function.pdf",
    "nihms-987662.pdf",
    "s40035-023-00378-6.pdf",
]


@dataclass(frozen=True)
class Settings:
    # --- Credentials ---
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    brave_api_key: str = os.getenv("BRAVE_SEARCH_API_KEY", "")

    # --- Models ---
    # Bare Claude id used in prompts/logging; litellm form is `anthropic/<id>`.
    agent_model: str = os.getenv("AGENT_MODEL", "claude-opus-4-8")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Retrieval ---
    chunk_chars: int = 1200            # target characters per chunk
    chunk_overlap: int = 200           # overlap between adjacent chunks
    top_k: int = 6                     # chunks returned by hybrid search
    rrf_k: int = 60                    # reciprocal-rank-fusion constant

    # --- Scope guardrail ---
    # A question whose best paper-similarity is below this cosine score is
    # treated as out-of-scope (before we even spend a web search).
    scope_threshold: float = float(os.getenv("SCOPE_THRESHOLD", "0.28"))

    # --- Paths (all git-ignored) ---
    papers_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / ".papers")
    index_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / ".index")
    memory_db: Path = field(default_factory=lambda: PACKAGE_ROOT / ".memory" / "sessions.sqlite3")

    @property
    def litellm_model(self) -> str:
        """Model string CrewAI/litellm expects for Anthropic."""
        return f"anthropic/{self.agent_model}"

    def ensure_dirs(self) -> None:
        for p in (self.papers_dir, self.index_dir, self.memory_db.parent):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
