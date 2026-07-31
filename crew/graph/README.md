# Scientist GraphRAG

A knowledge graph of the **people** behind the nine dopamine papers — who they
are, where they work, what they study, who they publish with — and a recommender
that suggests who should connect with whom.

Where the Q&A crew (`crew/`) answers *what the papers say*, this answers *who
wrote them and who should meet whom*. Same PDFs, same embedder, same grounding
discipline; a different retrieval substrate.

---

## Quick start

```powershell
# 1. Build the graph (extraction runs on a cheap model; ~5 min for the full corpus)
.venv\Scripts\python.exe -m crew.graph.build

# 2. Look around
.venv\Scripts\python.exe -m crew.graph.interface --stats
.venv\Scripts\python.exe -m crew.graph.interface --who "Wolfram Schultz"
.venv\Scripts\python.exe -m crew.graph.interface --recommend "Josh Berke"
.venv\Scripts\python.exe -m crew.graph.interface --viz          # open the map

# 3. Ask in plain English
.venv\Scripts\python.exe -m crew.graph.interface -q "Who should Marc Caron collaborate with?"
```

Build variants:

| Command | What it does |
|---|---|
| `build` | Full: front matter + every reference list + OpenAlex profiles |
| `build --no-refs` | Corpus authors only — fast, ~10 LLM calls |
| `build --force` | Ignore the corpus hash and rebuild (also re-derives the taxonomy) |
| `build --no-enrich` | Skip OpenAlex — no network calls at all |
| `build --enrich-all` | Enrich cited researchers too, not just corpus authors |
| `build --refresh-enrich` | Re-fetch profiles that are already present |
| `build --no-html` | Skip the viewer |

OpenAlex enrichment is **on by default** and needs no API key. It only ever adds
fields, and every failure — no network, rate limit, no confident match —
degrades to "leave this profile as it was", so a build with no network produces
exactly the graph that was built before enrichment existed. Set `GRAPH_ENRICH=0`
to turn it off permanently.

Refresh a profile without rebuilding:

```powershell
.venv\Scripts\python.exe -m crew.graph.interface --enrich "Marc Caron"   # one person
.venv\Scripts\python.exe -m crew.graph.interface --enrich                # fill the gaps
```

In the REPL: `/enrich <name>`, `/enrich`, `/refresh`.

---

## What gets extracted

| Node | Where it comes from | Carries |
|---|---|---|
| **Scientist** | Front matter + every reference list | name, aliases, institutions, research interests, papers, co-author degree |
| **Institution** | Author affiliations | canonical org name, departments, country, roster |
| **Topic** | LLM-derived taxonomy over the corpus | label, description, exact-match aliases |
| **Paper** | The nine PDFs + everything they cite | title, year, venue, DOI, authors, mention count |

| Edge | Meaning |
|---|---|
| `AUTHORED` | scientist → paper |
| `AFFILIATED_WITH` | scientist → institution |
| `CO_AUTHORED` | scientist ↔ scientist (symmetric, stored both ways) |
| `WORKS_ON` | scientist → topic, weighted by strength |
| `COVERS` | paper → topic |
| `CITES` | corpus paper → cited work |

Every node and edge carries `evidence`: the PDF filename and 1-indexed page it
was observed on. Nothing enters the graph without provenance, which is what lets
a recommendation be *defended* rather than merely asserted.

---

## Architecture

```
PDFs ──┬─▶ front matter ──▶ [LLM] ──▶ authors + affiliations ──┐
       │                                                        ├──▶ canon.py ──▶ ids
       └─▶ reference lists ─▶ [LLM] ──▶ cited works + authors ──┘        │
                                                                          ▼
                             topics.py ──▶ interests ──▶ build.py ──▶ graph.json
                                                                          │
                                            ┌─────────────────────────────┼───────────────┐
                                            ▼                             ▼               ▼
                                     recommend.py                     viz.py          chat.py
                                  (scored connections)            (graph.html)     (GraphRAG Q&A)
```

### The division of labour

The same rule the Q&A crew follows — **agents decide, the orchestrator
guarantees** — applies throughout:

| Concern | Owner | Why |
|---|---|---|
| Parsing messy bibliographies | LLM | Nine citation styles; regex-per-style is brittle |
| Deciding two names are one person | `canon.py` | Must be stable across rebuilds and unit-testable |
| Choosing the topic vocabulary | LLM (once) | A judgment about what the corpus is about |
| Assigning topics to papers | `topics.py` | Deterministic and free |
| Scoring recommendations | `recommend.py` | Numbers must not drift with model mood |
| Narrating an answer | LLM | Prose is what models are for |
| Deciding which citations are real | `chat.py::_resolve` | An invented `[G9]` is dropped, not printed |

---

## The recommender

Two different questions, two functions:

- **`recommend_connections(graph, sid)`** — *who should I meet?* Existing
  co-authors are excluded (suggesting them is useless), and same-institution
  candidates are discounted: the value of an introduction is highest across a
  structural hole.
- **`similar_scientists(graph, sid)`** — *who is like me?* Co-authors allowed.

Four normalised signals, blended with weights from `crew/config.py`:

| Signal | Weight | What it captures |
|---|---|---|
| `topic_overlap` | 0.35 | IDF-weighted Jaccard over interests. Sharing "dopamine receptors" (everyone) counts for far less than sharing "dopamine fasting" (five people) |
| `profile_similarity` | 0.25 | Cosine between profile-text embeddings — catches related work that shares no exact topic label |
| `graph_proximity` | 0.25 | Adamic-Adar over the co-authorship projection: a mutual collaborator with 3 co-authors is stronger evidence than one with 300 |
| `citation_affinity` | 0.15 | Co-citation — whose work gets cited by the same corpus papers |

Every recommendation returns its per-signal breakdown, the shared topics, the
bridging collaborators, and a `rationale` **assembled in Python from those
fields**. An LLM may narrate a recommendation; it never produces one.

```python
from crew.graph import ScientistGraph, recommend_connections

graph = ScientistGraph.load()
for rec in recommend_connections(graph, "berke|j", limit=5):
    print(f"{rec['score']:.3f}  {rec['name']:24s}  {rec['rationale']}")
```

---

## The viewer

`crew/.graph/graph.html` — one self-contained file, no CDN, no server. Open it
directly.

- **Layout is precomputed** in Python (`networkx.spring_layout`, fixed seed), so
  the map is deterministic and there is no physics loop burning CPU.
- **Canvas, not SVG** — a few thousand DOM nodes make pan and zoom stutter.
- **A legible core, not everything.** The full citation graph is thousands of
  people, which renders as a grey hairball. The viewer ships corpus authors,
  everyone they co-authored with, and the most-cited remainder up to
  `max_nodes` (default 600), and says so in the header. `graph.json` still holds
  the complete graph.

Click any node for its profile, interests, papers, collaborators, scored
recommendations, and the PDF pages backing them. Search, filter by node kind,
pan, zoom. Light and dark themes.

---

## External enrichment — why not Google Scholar

Google Scholar has **no public API**, and its terms prohibit automated access;
it enforces that with CAPTCHAs and IP blocks after a handful of programmatic
requests. Scraper libraries need rotating proxies to survive, and paid Scholar
APIs are just someone else absorbing that risk. Building on it would mean a
pipeline that is legally grey, silently rate-limited, and one markup change from
broken.

**OpenAlex** is used instead, and it runs by default. It is the open successor
to Microsoft Academic Graph: ~250M works, no API key, 100k requests/day, CC0
data. For this feature it is also a strictly better source — it exposes
canonical author ids, **ORCID**, last-known institution with a ROR id, topic
tags, works and citation counts, and an h-index.

ORCID matters most: it is the upgrade path for this graph's biggest weakness
(below).

One guard is worth calling out. OpenAlex's `search` ranks by relevance, not
identity, so taking the top hit would staple another researcher's h-index onto
a profile. A candidate is only accepted when its **surname agrees** with the
query — checked with `canon.parse_name`, the same resolver the graph is built
with. On the real corpus that matched 36/36 corpus authors, 25 of them with an
ORCID, and grew the institution count from 19 to 45.

Other legitimate options in the same family, unused but drop-in: Crossref (DOI
metadata), Semantic Scholar, PubMed E-utilities.

---

## Known limitations

| Limitation | Impact | Fix |
|---|---|---|
| **Names resolve on surname + first initial** | Two distinct `Wang D.` merge into one node | ORCID via enrichment; this is the standard bibliometric heuristic without it |
| A person cited as `Maknoon Razia D` and printed as `Dur E Maknoon Razia` may split | Duplicate nodes for compound surnames | Alias clustering, or ORCID |
| **Institutions cover corpus authors only** — 100% of the ~36 people who wrote the nine papers, ~1% of all 3,600 | Reference lists print names, never affiliations, so a cited-only researcher has no institution to record | `--enrich-all` fills these from OpenAlex |
| Enrichment matches on name, so a common name can match the wrong person | A plausible but wrong h-index (e.g. a corpus author named `Jing Wu`) | Match on ORCID or DOI-linked works rather than a name search |
| Topics come from titles for cited works | Coarser interests than corpus authors get from abstracts | Fetch abstracts via OpenAlex/Crossref |
| Co-citation is over nine citing papers | A coarse signal, deliberately weighted lowest | More corpus papers |
| Reference extraction quality varies | Two-column PDFs interleave entries; some are dropped | Layout-aware extraction (e.g. GROBID) |
| No recommendation eval set | Ranking changes cannot be shown to be improvements | Hold-out link prediction: hide known co-authorships, measure recall@k |

That last row is the one to fix first if this goes anywhere serious — it is the
same gap `ARCHITECTURE.md §14.3` flags as lever #0 for the retrieval layer, and
for the same reason: nothing below it can be *justified* without it.

---

## Testing

```powershell
.venv\Scripts\python.exe -m pytest tests -k graph
```

120 tests, no API key, no network, no built graph required. They cover the parts
whose behaviour is guaranteed by code rather than by a language model:

- **`test_graph_canon.py`** — every name string is real, taken from the nine
  PDFs. Getting one wrong silently splits or merges a researcher.
- **`test_graph_recommend.py`** — asserted against a synthetic 6-person graph
  with known topology, so ranking, exclusion and IDF weighting are checked
  against a structure we chose.
- **`test_graph_grounding.py`** — invented `[G#]` ids are dropped; scores pass
  through untouched.
- **`test_graph_topics.py`** — alias word-boundaries (`D2` ∉ `D20`), and that a
  flat similarity profile selects *nothing* rather than everything.
- **`test_graph_store.py`** — persistence round-trip, lookup, projection.
- **`test_graph_extract.py`** — leniency on model output. A schema that is too
  strict doesn't raise, it silently deletes a paper's whole author team: a real
  build lost the five Duke authors of the 1998 Missale review because the model
  correctly returned `"doi": null` for a paper with no printed DOI.
- **`test_graph_viz.py`** — the viewer is one template string, so a broken brace
  fails as a blank page rather than an exception: layout reproducibility, tag
  balance, both theme overrides, and that a `</script>` in a paper title cannot
  break out of the payload.
- **`test_graph_enrich.py`** — every network call is patched, so what is
  asserted is the *guards*, not OpenAlex: surname agreement, that offline
  enrichment leaves the graph byte-identical, that an added institution also
  gets its `AFFILIATED_WITH` edge, and that `stats` is recounted afterwards.
