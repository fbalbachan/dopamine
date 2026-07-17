# Design Decisions

This document explains the key choices behind the dopamine Q&A crew: the RAG
layer, the agent architecture, the memory model, and production & cost.

---

## 1. RAG layer

**Corpus.** The nine PDFs from `rootstrap/ai-agents-challenge-papers` are
downloaded to `crew/.papers/` at index-build time (raw GitHub URLs, so no
`git` dependency). The filename list is pinned in `config.py`.

**Page-level chunking is the citation unit.** Text is extracted **per page**
with `pypdf`, and every chunk carries its `(file, page)`. That is what makes
"traceable to a paper file + page" real rather than aspirational — a citation
is a page number that came out of the extractor, not something the model wrote.
Pages are split into ~1200-character windows with 200-character overlap so a
concept that straddles a boundary still lands intact in at least one chunk.

**Hybrid retrieval (dense + BM25 + RRF).** We combine:

- **Dense** — local `sentence-transformers/all-MiniLM-L6-v2` embeddings in a
  cosine **FAISS** index (`IndexFlatIP` over L2-normalized vectors).
- **Lexical** — **BM25** (`rank_bm25`) over the same chunks.

fused with **Reciprocal Rank Fusion**. The lexical half matters a lot for this
corpus: exact tokens like `D2`, `L-DOPA`, `substantia nigra`, receptor names,
and gene symbols are exactly where pure dense retrieval is weakest, and where
scientific questions are most precise. RRF needs no score calibration between
the two retrievers, which keeps the fusion robust.

**Local embeddings, no embedding API.** MiniLM runs on CPU, needs no API key,
and keeps per-query cost at zero for retrieval. `faiss-cpu` handles far more
than nine papers before any of this needs revisiting. The index and a chunk
cache are persisted to `crew/.index/` and rebuilt only when the corpus,
embedding model, or chunk parameters change (tracked by a corpus hash).

**Why not a vector DB service?** At nine papers (low thousands of chunks) a
flat FAISS index is exact, instant, and dependency-light. A hosted vector DB
would add latency, cost, and ops with no recall benefit at this scale. The swap
point is documented under Production below.

**Grounding guarantee.** Retrieval results are written into an
`EvidenceRegistry` under stable ids (`P1`, `P2`, …). The Synthesizer may only
cite those ids, and the orchestrator rebuilds the final `sources` list from the
registry — never from the model's free text. This structurally prevents
fabricated page numbers or URLs: a citation that isn't backed by a real
retrieval result simply can't appear in `sources`.

---

## 2. Agent architecture

**Framework: CrewAI**, as requested. Four modular agents, each a real CrewAI
`Agent` backed by Claude (`claude-opus-4-8` via CrewAI's litellm wrapper):

| Agent | Judgment it contributes | Tool the orchestrator runs for it |
|---|---|---|
| **MemoryKeeper** | Rewrites an elliptical follow-up into a standalone question using session history | — |
| **Retriever** | Plans 1–3 focused retrieval queries from the question | `search_papers` (hybrid RAG) |
| **WebSearcher** | Decides whether the web is needed and crafts one query | `search_web` (Brave) |
| **Synthesizer** | Writes the grounded answer and maps each claim to evidence ids | — |

**Agents decide; the orchestrator executes the tools.** The agents are
*tool-free*: each returns a structured decision (a query plan, a web-search
decision, or the final grounded answer), and the orchestrator (`chat.py`) runs
the RAG and Brave tools in Python, feeding the evidence back as task context.

This is a deliberate choice with two motivations:

1. **A hard framework incompatibility.** CrewAI's ReAct tool-executor drives a
   tool-using agent by *prefilling the assistant turn* (`Thought: …`). Claude
   4.8 (like the whole 4.6+ family) rejects assistant prefill — it requires the
   conversation to end with a user message — so a tool-equipped CrewAI agent on
   `claude-opus-4-8` returns a 400. Verified directly: a no-tool agent and
   `output_pydantic` structured output both work; only the ReAct tool path
   fails. Keeping agents tool-free avoids it without downgrading the model.
2. **Guaranteed grounding.** Because the orchestrator runs retrieval, it writes
   every result into the `EvidenceRegistry` itself. The Synthesizer only cites
   ids; `_resolve()` rebuilds `sources` from the registry. Citation correctness
   therefore doesn't depend on an agent using a tool correctly — it's enforced
   in Python.

**Orchestrated pipeline, not a free-for-all.** The spine is deterministic —
memory → scope → Retriever(plan) → RAG → WebSearcher(decide) → Brave →
Synthesizer → id resolution. This buys the three things the challenge asks for:
a hard **scope** short-circuit before any spend, a **typed** output contract,
and **guaranteed** citation grounding. The agents own the judgment calls; the
orchestrator owns the correctness invariants.

**Claude 4.8 API specifics.** Two settings follow from the model: no
`temperature`/`top_p`/`top_k` (sampling params are removed on Opus 4.7/4.8 and
return a 400), and CrewAI's telemetry / interactive "view traces?" prompt is
disabled (`CREWAI_TRACING_ENABLED=false`) so runs are non-interactive.

**Source-aware synthesis.** The Synthesizer returns a structured object
(`answer`, `claims[]`, `used_web_search`, `confidence`) via Pydantic. Each
claim lists the evidence ids that support it; `_resolve()` maps those ids to
indices into the real `sources` list. `used_web_search` is set from the
registry (did the Brave tool actually return anything?), not from the model's
self-report — so it can't drift from reality.

**Out-of-scope guardrail (two layers, cheap-first).**
1. **Retrieval floor** — best cosine similarity to any paper chunk. Comfortably
   above the floor → in; far below → out (unless in-domain vocabulary rescues a
   terse question).
2. **LLM tie-breaker** — only in a narrow gray band around the floor do we
   spend one **Claude Haiku** call to decide. Common cases cost nothing; only
   genuinely ambiguous ones pay for a small model.

An out-of-scope question returns the standard refusal: a polite message,
`refused: true`, `in_scope: false`, and empty `sources`/`claims`.

**WebSearcher backend: Brave Search API.** Brave returns real titles + URLs
(needed for citations), has a usable free tier, and is a first-class CrewAI
search backend. The tool is written directly against the Brave REST API so we
control the citation format and degrade gracefully — a missing key or a quota
error becomes a clean "web unavailable" signal rather than a crash, and the
crew still answers from the papers.

---

## 3. Memory model

**Per-`session_id`, SQLite-backed.** Each turn (user question + crew answer) is
appended to a `turns` table keyed by `session_id`. SQLite (not an in-process
dict) means a session survives process restarts and multiple interface
invocations, with zero external infrastructure.

**How follow-ups resolve.** On each question the MemoryKeeper is given the last
few turns and rewrites the new question into a **standalone** form
("Does *that* change in Parkinson's?" → "Does dopamine's role in reward
prediction error change in Parkinson's disease?"). Crucially, the *rewritten*
question is what feeds both the **scope** check and **retrieval** — so a
follow-up that is only in-scope by context is judged correctly, and retrieval
searches for the fully-resolved intent.

**Bounded context.** Only the most recent N turns (default 8 messages) are
surfaced, and each is length-capped, to keep prompt cost flat no matter how
long a session runs. Full history remains in SQLite for audit; only the recent
window is injected.

**Why not CrewAI's built-in memory?** We set `memory=False` on the Crew and run
our own store. CrewAI's memory is optimized for intra-run agent scratchpad and
would pull in its own embedding/vector stack; our requirement is a simple,
inspectable, per-session transcript keyed by an id the caller controls.

---

## 4. Production & cost

**Cost controls, by layer.**
- **Retrieval is free** (local MiniLM embeddings + FAISS/BM25) — no per-query
  embedding spend.
- **Scope guardrail** answers most questions with a vector lookup and no LLM
  call; only gray-band cases pay for one **Haiku** classification.
- **Web search** runs only when the WebSearcher judges the papers insufficient
  or the question is recency-sensitive, keeping Brave usage (and its quota) low.
- **Prompt cost is bounded** by the capped memory window and capped evidence
  snippets, so cost per turn doesn't grow with session length.
- The main reasoning model is configurable via `AGENT_MODEL`; dropping to a
  cheaper Claude tier for the Synthesizer is a one-line change.

**Reliability.**
- **Graceful degradation** — no Brave key or a quota error yields a clean
  "web unavailable" path; the crew still answers from the papers.
- **Deterministic grounding** — the registry design means citation correctness
  doesn't depend on the model behaving; it's enforced in Python.
- **Idempotent index** — rebuilds only on corpus change (hash-tracked), so
  redeploys don't re-embed needlessly.

**What I'd change to scale.**
- **Vector store** — swap flat FAISS for a persisted vector DB (pgvector /
  Qdrant / LanceDB, already a transitive dep) once the corpus grows past
  ~10⁵ chunks or needs multi-process sharing. The `Retriever` interface stays
  the same.
- **Reranking** — add a cross-encoder reranker over the fused top-k for higher
  precision on hard queries (deliberately omitted here to keep the install lean
  and latency low at nine papers).
- **Memory** — move the SQLite store to Postgres for concurrency, and add a
  rolling **summary** of older turns so very long sessions keep coreference
  ability without unbounded history.
- **Serving** — wrap `DopamineCrew.ask` behind a small FastAPI endpoint;
  it is already stateless apart from the shared memory store, so it scales
  horizontally with a shared DB.
- **Observability & caching** — add request tracing and cache identical
  (question, session) synthesis; enable Claude prompt caching on the stable
  system/tool prefix to cut input-token cost on repeated calls.
- **Safety** — the scope guardrail is the first gate; for production add output
  validation (every in-scope claim must carry ≥1 source) as a hard check before
  returning.
