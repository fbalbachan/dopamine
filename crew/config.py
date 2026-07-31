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

    # --- Knowledge graph (crew.graph) ---
    # Bulk entity extraction is a *parsing* job, not a reasoning one, so it runs
    # on a cheap model by default. Overridable for quality experiments.
    graph_extract_model: str = os.getenv("GRAPH_EXTRACT_MODEL", "claude-haiku-4-5")
    # Reference lists are extracted in batches of this many characters per call.
    graph_batch_chars: int = int(os.getenv("GRAPH_BATCH_CHARS", "9000"))
    # A cited work must be seen this many times before it earns a graph node.
    # 1 keeps the full citation network; raise it to prune one-off mentions.
    graph_min_mentions: int = int(os.getenv("GRAPH_MIN_MENTIONS", "1"))
    # Topic assignment (see crew.graph.topics.Taxonomy.assign): a topic must
    # stand out from the text's own mean similarity by this many standard
    # deviations, *and* clear the absolute cosine floor.
    graph_topic_z: float = float(os.getenv("GRAPH_TOPIC_Z", "1.1"))
    graph_topic_threshold: float = float(os.getenv("GRAPH_TOPIC_THRESHOLD", "0.20"))

    # --- Connection recommender weights (see crew.graph.recommend) ---
    # Signals are each normalised to [0, 1], then blended with these weights.
    rec_w_topic: float = float(os.getenv("REC_W_TOPIC", "0.35"))      # shared research interests
    rec_w_profile: float = float(os.getenv("REC_W_PROFILE", "0.25"))  # profile embedding similarity
    rec_w_graph: float = float(os.getenv("REC_W_GRAPH", "0.25"))      # co-authorship proximity
    rec_w_citation: float = float(os.getenv("REC_W_CITATION", "0.15"))  # citation affinity
    # Recommending a same-institution colleague is usually less useful than
    # surfacing an external collaborator, so it is discounted (not excluded).
    rec_same_institution_penalty: float = float(os.getenv("REC_SAME_INST_PENALTY", "0.10"))

    # --- Paths (all git-ignored) ---
    papers_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / ".papers")
    index_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / ".index")
    graph_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / ".graph")
    memory_db: Path = field(default_factory=lambda: PACKAGE_ROOT / ".memory" / "sessions.sqlite3")

    @property
    def litellm_model(self) -> str:
        """Model string CrewAI/litellm expects for Anthropic."""
        return f"anthropic/{self.agent_model}"

    @property
    def litellm_graph_extract_model(self) -> str:
        """Model string for the graph's bulk entity extractor."""
        return f"anthropic/{self.graph_extract_model}"

    def ensure_dirs(self) -> None:
        for p in (self.papers_dir, self.index_dir, self.graph_dir, self.memory_db.parent):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
