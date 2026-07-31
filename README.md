# Dopamine Q&A Crew

A modular **CrewAI** crew that answers dopamine-domain questions by combining:

- **RAG** over the nine papers in [rootstrap/ai-agents-challenge-papers](https://github.com/rootstrap/ai-agents-challenge-papers) (hybrid dense + BM25 retrieval, page-level citations),
- **Live web search** (Brave) to fill gaps and verify recent claims,
- **Source-aware generation** — every factual claim is traceable to a paper (`file` + `page`) or a URL,
- **Conversational memory** keyed by `session_id` so follow-up questions resolve correctly,
- an **out-of-scope guardrail** that returns a polite refusal (flagged, empty sources).

Every answer is returned as a single **typed dict** (`crew.schema.CrewResponse`) that is JSON-serializable.

All code lives under [`crew/`](crew/).

> **Also included: [Scientist GraphRAG](crew/graph/README.md)** — a second
> surface over the same nine PDFs that extracts the *people*: researcher
> profiles, institutions, research interests and citation links, assembled into
> a knowledge graph that recommends who should connect with whom. Ships with an
> interactive HTML graph viewer.
>
>   A GraphRAG pipeline in crew/graph/ over the same nine PDFs. From a full build: 3,626 scientists, 1,359 papers, 18 institutions, 42,303 edges.                                                                                                                                                                                                                                                                                                           
  Extraction is split so each half does what it's good at: a cheap model parses the nine incompatible citation styles, and deterministic Python decides which surface forms are the same person. That keeps entity resolution      
  unit-testable and stable — a rebuild can't silently re-partition the graph.                                                                                                                                                                                                                                                                                                                                                                                       
  The recommender blends four normalized signals — IDF-weighted topic overlap, profile-embedding similarity, Adamic-Adar over the co-authorship network, and co-citation — and returns the per-signal breakdown plus a rationale   
  assembled in Python, never by the model. recommend_connections excludes existing co-authors; similar_scientists doesn't. Real output for Schultz: Nathaniel Daw (0.68, bridged via Dayan and Montague), Minoru Kimura, Ray Dolan 
  — with signals firing independently, Kimura scoring on topic alone and Smith on network alone.  
> 
>
> ```powershell
> .venv\Scripts\python.exe -m crew.graph.build          # build graph + viewer
> .venv\Scripts\python.exe -m crew.graph.interface --viz
>        Explore the scientist graph from the command line.                                                                                                                                                                     
                                                                                                                                                                                                                                 
       python -m crew.graph.interface                              # REPL                                                                                                                                                    
       python -m crew.graph.interface --who "Wolfram Schultz"      # one profile                                                                                                                                             
       python -m crew.graph.interface --recommend "Josh Berke"     # connections                                                                                                                                             
       python -m crew.graph.interface --similar "Marc Caron"       # similar profiles                                                                                                                                        
       python -m crew.graph.interface --topic "Parkinson"          # who works on it                                                                                                                                         
       python -m crew.graph.interface --institution "Duke"         # who is there                                                                                                                                            
       python -m crew.graph.interface --stats                      # graph summary                                                                                                                                           
       python -m crew.graph.interface -q "Who should Berke meet?"  # natural language    
> ```

---

## Architecture


flowchart TD
    U([User + session_id]) --> IF[interface.py CLI / REPL]
    IF --> ASK["DopamineCrew.ask()"]

    subgraph ORCH["Orchestrator (chat.py)"]
        ASK --> MEMR[MemoryKeeper agent<br/>rewrite follow-up → standalone Q]
        MEMR --> SCOPE{Scope guardrail<br/>scope.py}
        SCOPE -- out of scope --> REF[Polite refusal<br/>refused=true · sources=[]]
        SCOPE -- in scope --> CREW
    end

    subgraph CREW["Agent pipeline (crew)"]
        direction LR
        RET[Retriever agent<br/>plans queries] --> WEB[WebSearcher agent<br/>decides + web query] --> SYN[Synthesizer agent<br/>grounded answer + claims]
    end

    RET -.query plan.-> RAGRUN[[orchestrator: RAG search]]
    WEB -.web query.-> WEBRUN[[orchestrator: Brave search]]
    RAGRUN -->|"[P#]"| REG[(EvidenceRegistry<br/>ground-truth sources)]
    WEBRUN -->|"[W#]"| REG

    subgraph RAG["RAG layer (rag.py)"]
        FAISS[(FAISS cosine index)]
        BM25[(BM25 lexical)]
        FAISS & BM25 --> RRF[Reciprocal Rank Fusion]
    end
    RAGRUN -.hybrid search.-> RRF
    WEBRUN -.Brave API.-> BRAVE[(Brave Search)]

    SYN --> RES["_resolve()<br/>cited ids → real sources"]
    REG --> RES
    RES --> OUT[[CrewResponse dict]]
    REF --> OUT
    OUT --> MEM[(Session memory<br/>SQLite · memory.py)]
    OUT --> U

    PDFS[[9 PDFs<br/>rootstrap repo]] -->|"download + page-chunk + embed"| RAG


**Flow:** the MemoryKeeper turns a follow-up into a standalone question using the session's history; the scope guardrail refuses anything off-domain before spending tokens; the Retriever plans focused queries and the WebSearcher decides whether the live web is needed, while the **orchestrator executes** the RAG and Brave tools; the Synthesizer writes a grounded answer citing evidence ids; and `_resolve()` rebuilds the `sources` list from the **EvidenceRegistry** (populated by the tools, not the model) so no citation can point at a page or URL that retrieval didn't actually return.

> **Why the agents don't call the tools directly:** CrewAI's ReAct tool-executor prefills the assistant turn, which Claude 4.8 rejects (it requires the conversation to end with a user message). So the agents are tool-free — they contribute *judgment* (query planning, the web-search decision, synthesis) via structured outputs — and the orchestrator runs the tools in Python. This both sidesteps the incompatibility and lets the orchestrator *guarantee* citation grounding. See [DECISIONS.md](DECISIONS.md).

---

## Setup

Requires **Python 3.11+**.

```powershell
# 1. Install dependencies (wheels only — avoids source builds on Windows)
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r crew\requirements.txt

# 2. Configure secrets
copy crew\.env.example crew\.env
#   then edit crew\.env and set ANTHROPIC_API_KEY (and BRAVE_SEARCH_API_KEY)

# 3. Build the RAG index (downloads the 9 PDFs; first run also downloads the
#    embedding model — a few hundred MB, one time)
.venv\Scripts\python.exe -m crew.build_index
```

> **First-run note:** `build_index` and the first query download the
> sentence-transformers embedding model from Hugging Face. Expect a one-time
> pause of a minute or two with network access — it is not hung.

---

## Usage

### Interactive (multi-turn, memory within the session)

```powershell
.venv\Scripts\python.exe -m crew.interface
```

```
you> What is dopamine's role in reward prediction error?
you> Does that change in Parkinson's disease?   # follow-up resolves via memory
you> /quit
```

REPL commands: `/new` (fresh session), `/session <id>`, `/json` (toggle raw dict), `/quit`.

### One-shot (prints the full typed dict)

```powershell
.venv\Scripts\python.exe -m crew.interface -q "What are the D1 vs D2 receptor families?" --json
```

### Programmatic

```python
from crew import DopamineCrew

crew = DopamineCrew()
resp = crew.ask("What is dopamine's role in reward learning?", session_id="demo")

resp["answer"]            # str
resp["sources"]           # [{"type":"paper","file":...,"page":...,"snippet":...}, {"type":"web",...}]
resp["claims"]            # [{"statement":..., "source_indices":[0,2]}, ...]
resp["in_scope"]          # bool
resp["refused"]           # bool  (out-of-scope → True, sources == [])
resp["used_web_search"]   # bool
resp["confidence"]        # "high" | "medium" | "low"
```

---

## Tests

A `pytest` suite covers the deterministic paths — no API key required (the one
in-scope test asserts scope *passes*, then stops at the key gate).

```powershell
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r crew\requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

| File | Covers |
|---|---|
| `tests/test_schema.py` | refusal contract — all keys, flags, empty sources, JSON-safe |
| `tests/test_scope.py` | guardrail decision boundary (floor, lexicon rescue, gray-band tie-break) |
| `tests/test_resolve.py` | grounding — cited ids → real deduped sources, unknown ids dropped, `used_web_search` is ground truth |
| `tests/test_memory.py` | ordering, session isolation, bounded window, clear |
| `tests/test_config.py` | 9 papers pinned, exact filenames, model string |
| `tests/test_rag_integration.py` | real index: page-level citations, in- vs off-domain scores *(skips if index not built)* |
| `tests/test_crew_refusal.py` | end-to-end out-of-scope refusal via `DopamineCrew.ask` *(skips if index not built)* |
| `tests/test_graph_canon.py` | graph entity resolution — one researcher, one node id across all nine citation styles |
| `tests/test_graph_recommend.py` | recommender policy on a synthetic graph: co-author exclusion, IDF weighting, determinism |
| `tests/test_graph_grounding.py` | graph answers — invented `[G#]` ids dropped, scores pass through untouched |
| `tests/test_graph_topics.py` | topic assignment — alias word-boundaries, relative (not absolute) selection |
| `tests/test_graph_store.py` | graph persistence, lookup, co-authorship projection |
| `tests/test_graph_viz.py` | viewer export — deterministic layout, tag balance, both theme overrides, payload escaping |
| `tests/test_graph_extract.py` | extraction leniency — a `null` field must not silently discard a paper's whole author list |

Integration tests skip cleanly when the index hasn't been built; the unit tests
always run.

---

## Response shape (`CrewResponse`)

```jsonc
{
  "session_id": "demo",
  "question": "What is dopamine's role in reward learning?",
  "answer": "Dopamine neurons encode a reward prediction error ... [P2][P5]",
  "sources": [
    { "type": "paper", "file": "da_reward_learning_wise.pdf", "page": 4,
      "snippet": "..." },
    { "type": "web", "url": "https://...", "title": "...", "snippet": "..." }
  ],
  "claims": [
    { "statement": "Dopamine neurons signal reward prediction error.",
      "source_indices": [0] }
  ],
  "in_scope": true,
  "refused": false,
  "used_web_search": true,
  "confidence": "high"
}
```

An **out-of-scope** question returns the same shape with a polite refusal in
`answer`, `refused: true`, `in_scope: false`, and `sources`/`claims` empty.

---

## Layout

```
crew/
  interface.py     # CLI / REPL entry point
  chat.py          # DopamineCrew orchestrator (ask)
  agents.py        # Retriever, WebSearcher, Synthesizer, MemoryKeeper (CrewAI Agents)
  tools.py         # paper_search + web_search tools, EvidenceRegistry
  rag.py           # download → page-chunk → embed → FAISS + BM25 hybrid search
  build_index.py   # `python -m crew.build_index`
  memory.py        # SQLite conversational memory per session_id
  scope.py         # out-of-scope guardrail
  schema.py        # TypedDict response contract
  config.py        # settings + paths (reads crew/.env)
  graph/           # Scientist GraphRAG — see crew/graph/README.md
    build.py       # `python -m crew.graph.build` — PDFs → knowledge graph
    extract.py     # LLM extraction of front matter + reference lists
    canon.py       # name/institution entity resolution (deterministic)
    topics.py      # research-interest taxonomy + assignment
    store.py       # ScientistGraph: persistence, lookups, networkx views
    recommend.py   # multi-signal connection recommender
    viz.py         # self-contained interactive HTML viewer
    chat.py        # GraphCrew — natural-language questions over the graph
    interface.py   # `python -m crew.graph.interface`
    enrich.py      # optional OpenAlex enrichment (ORCID, h-index)
  requirements.txt
  requirements-dev.txt
  .env.example
tests/             # pytest suite for the deterministic paths
pytest.ini
README.md
DECISIONS.md       # RAG layer · agent architecture · memory model · production & cost
```

See [DECISIONS.md](DECISIONS.md) for the design rationale.
