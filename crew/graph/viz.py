"""Interactive HTML viewer for the scientist graph.

Produces one self-contained file — no CDN, no server, no build step — that opens
straight from disk:

    crew/.graph/graph.html

Three decisions shape it:

**The layout is computed in Python, not the browser.** ``networkx.spring_layout``
runs once at export with a fixed seed, so the map is deterministic: the same
graph always draws the same shape, and reopening the file does not reshuffle
everyone. It also means no physics loop burning CPU while you read.

**Only a legible core is drawn.** A full citation graph is several thousand
people, which renders as a grey hairball that answers no question. The viewer
ships the corpus authors, everyone they co-authored with, and the most-cited
remainder up to ``max_nodes`` — and says so in the header. ``graph.json`` still
holds everything for programmatic use.

**Rendering is canvas, not SVG.** A few thousand DOM nodes make pan and zoom
stutter; one canvas stays smooth, and hit-testing a click against node positions
is a loop rather than a layout engine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings

# The node kinds the viewer draws. Their colours live in CSS (one value per
# theme) and are read back at draw time, so the canvas re-themes with the page
# rather than carrying a second, drifting copy of the palette here.
NODE_KINDS = ("corpus_author", "scientist", "topic", "institution", "paper")


def _select_nodes(graph, max_nodes: int) -> List[str]:
    """Pick the subgraph worth drawing: corpus authors, their circle, then reach."""
    chosen: List[str] = []
    seen = set()

    def take(sid: str) -> None:
        if sid not in seen and sid in graph.scientists:
            seen.add(sid)
            chosen.append(sid)

    corpus = [s["id"] for s in graph.scientists.values() if s["corpus_author"]]
    for sid in corpus:
        take(sid)
    for sid in list(chosen):
        for other in graph.collaborators(sid):
            take(other)
    remainder = sorted(
        (s for s in graph.scientists.values() if s["id"] not in seen),
        key=lambda s: (s["degree"], s["cited_works"]),
        reverse=True,
    )
    for s in remainder:
        if len(chosen) >= max_nodes:
            break
        take(s["id"])
    return chosen[:max_nodes]


def build_view(graph, max_nodes: int = 600, recommendations_per: int = 5,
               embed=None) -> dict:
    """Assemble the JSON payload the viewer renders."""
    import networkx as nx

    from .recommend import Recommender

    scientist_ids = _select_nodes(graph, max_nodes)
    scientist_set = set(scientist_ids)

    # Topics and institutions attached to the selected people become hub nodes:
    # they give the layout something to cluster around and are few in number.
    topic_ids = sorted({
        t for sid in scientist_ids for t in graph.scientists[sid]["topics"]
        if t in graph.topics
    })
    institution_ids = sorted({
        i for sid in scientist_ids for i in graph.scientists[sid]["institutions"]
        if i in graph.institutions
    })
    paper_ids = [p["id"] for p in graph.papers.values() if p["in_corpus"]]

    keep = scientist_set | set(topic_ids) | set(institution_ids) | set(paper_ids)

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(keep)
    edges: List[dict] = []
    for e in graph.edges:
        src, tgt, rel = e["source"], e["target"], e["relation"]
        if src not in keep or tgt not in keep:
            continue
        if rel == "CITES":
            continue                     # paper-to-paper; not in this view
        if rel == "CO_AUTHORED" and src > tgt:
            continue                     # stored both ways; draw once
        weight = float(e.get("weight", 1.0))
        edges.append({"s": src, "t": tgt, "r": rel, "w": round(weight, 3)})
        layout_graph.add_edge(src, tgt, weight=max(weight, 0.1))

    # Deterministic layout. k controls spacing; scaling it by 1/sqrt(n) keeps
    # density readable whether the view holds 50 nodes or 600.
    n = max(layout_graph.number_of_nodes(), 1)
    pos = nx.spring_layout(
        layout_graph, seed=42, k=1.6 / (n ** 0.5), iterations=200, weight="weight"
    )
    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0

    # Recommendations are baked in at export time so the viewer needs no
    # runtime. Passing the embedder keeps the profile-similarity signal alive;
    # without it that quarter of the score would silently read zero.
    recommender = Recommender(graph, embed)
    nodes: List[dict] = []

    for sid in scientist_ids:
        s = graph.scientists[sid]
        recs = recommender.score_all(sid, exclude_collaborators=True,
                                     limit=recommendations_per)
        nodes.append({
            "id": sid,
            "kind": "corpus_author" if s["corpus_author"] else "scientist",
            "label": s["name"],
            "degree": s["degree"],
            "cited": s["cited_works"],
            "topics": [graph.topics[t]["label"] for t in s["topics"] if t in graph.topics],
            "institutions": [
                graph.institutions[i]["name"] for i in s["institutions"]
                if i in graph.institutions
            ],
            "papers": [
                {"title": p["title"] or p["id"], "year": p["year"],
                 "corpus": p["in_corpus"]}
                for p in graph.papers_of(sid)[:8]
            ],
            "collaborators": [
                graph.scientists[c]["name"] for c in graph.collaborators(sid)[:12]
                if c in graph.scientists
            ],
            "evidence": s.get("evidence", [])[:3],
            "external": s.get("external") or {},
            "recs": [
                {
                    "name": r["name"], "id": r["scientist_id"], "score": r["score"],
                    "why": r["rationale"], "topics": r["shared_topics"],
                    "bridges": r["shared_collaborators"],
                    # The per-signal breakdown is what makes a score defensible
                    # rather than oracular — the viewer renders it as four small
                    # multiples under each card.
                    "sig": {
                        "topics": r["signals"]["topic_overlap"],
                        "profile": r["signals"]["profile_similarity"],
                        "network": r["signals"]["graph_proximity"],
                        "citation": r["signals"]["citation_affinity"],
                    },
                }
                for r in recs
            ],
        })

    for tid in topic_ids:
        t = graph.topics[tid]
        nodes.append({
            "id": tid, "kind": "topic", "label": t["label"],
            "degree": len(t["scientists"]), "cited": 0,
            "description": t["description"],
            "topics": [], "institutions": [], "papers": [], "collaborators": [],
            "evidence": [], "external": {}, "recs": [],
        })
    for iid in institution_ids:
        inst = graph.institutions[iid]
        nodes.append({
            "id": iid, "kind": "institution", "label": inst["name"],
            "degree": len(inst["scientists"]), "cited": 0,
            "description": ", ".join(inst["departments"][:3]),
            "country": inst["country"],
            "topics": [], "institutions": [], "papers": [], "collaborators": [],
            "evidence": inst.get("evidence", [])[:2], "external": {}, "recs": [],
        })
    for pid in paper_ids:
        p = graph.papers[pid]
        nodes.append({
            "id": pid, "kind": "paper", "label": p["title"] or p["file"],
            "degree": len(p["authors"]), "cited": p["mentions"],
            "description": f"{p['venue']} {p['year'] or ''}".strip(),
            "topics": [graph.topics[t]["label"] for t in p["topics"] if t in graph.topics],
            "institutions": [], "papers": [], "collaborators": [],
            "evidence": p.get("evidence", [])[:2], "external": {}, "recs": [],
        })

    for node in nodes:
        x, y = pos.get(node["id"], (0.0, 0.0))
        node["x"] = round((x - min_x) / span_x, 5)
        node["y"] = round((y - min_y) / span_y, 5)

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": graph.stats,
        "shown": len(nodes),
        "total_scientists": len(graph.scientists),
    }


def render_fragment(view: dict, title: str = "Dopamine Research Graph") -> str:
    """The viewer as a ``<style> + markup + <script>`` fragment.

    Returned without ``<html>``/``<head>``/``<body>`` so it can be embedded
    either in the standalone file below or published directly as an Artifact.
    """
    payload = json.dumps(view, ensure_ascii=False, separators=(",", ":"))
    # A literal "</script>" anywhere in the data (a paper title, an odd
    # affiliation) would close the tag early and break the page. Escaping the
    # angle brackets is inert inside a JSON string literal.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)


def export_html(graph, path: Optional[Path] = None, max_nodes: int = 600,
                embed=None) -> Path:
    """Write the standalone viewer next to the graph JSON."""
    path = path or (settings.graph_dir / "graph.html")
    view = build_view(graph, max_nodes=max_nodes, embed=embed)
    fragment = render_fragment(view)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>Dopamine Research Graph</title>\n</head>\n<body>\n"
        f"{fragment}\n</body>\n</html>\n",
        encoding="utf-8",
    )
    return path


# The viewer. One template string so the export has no asset dependencies, and
# so the file you open from disk and the page you publish are the same artefact.
#
# Palette note: node hues are declared twice, once per theme, because a dot that
# reads well on warm paper turns muddy on a near-black ground. They are read
# from CSS custom properties at draw time rather than hard-coded in JS, so the
# canvas re-themes along with the rest of the page.
_TEMPLATE = r"""
<style>
  :root {
    color-scheme: light dark;
    /* Neutrals biased warm, toward the amber accent — a pure grey reads as
       unconsidered beside it. */
    --ground:   #f6f5f2;
    --surface:  #fffefc;
    --sunken:   #edeae4;
    --line:     #e0dcd4;
    --line-firm:#cfc9be;
    --ink:      #1a1c22;
    --ink-soft: #6b6862;
    --ink-faint:#94908a;
    /* Categorical node hues. Kind is also carried by the legend and the
       inspector, so colour is never the only channel. */
    --k-corpus_author: #c2661d;
    --k-scientist:     #3f5f96;
    --k-institution:   #0d7d70;
    --k-topic:         #7d4f95;
    --k-paper:         #86817a;
    --accent:   #c2661d;
    --accent-soft: #f2e2d2;
    --shadow: 0 1px 2px rgba(26,28,34,.06), 0 8px 24px rgba(26,28,34,.07);
    --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono",
                 Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground:#101218; --surface:#181b22; --sunken:#0b0d12; --line:#282c36;
      --line-firm:#3a3f4c; --ink:#e8e6e1; --ink-soft:#9a978f; --ink-faint:#6e6b66;
      --k-corpus_author:#e8913f; --k-scientist:#7899d4; --k-institution:#2bab9b;
      --k-topic:#ab7cc4; --k-paper:#8f8a83;
      --accent:#e8913f; --accent-soft:#3a2a19;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
    }
  }
  /* The viewer's own toggle must beat the media query in BOTH directions. */
  :root[data-theme="light"] {
    --ground:#f6f5f2; --surface:#fffefc; --sunken:#edeae4; --line:#e0dcd4;
    --line-firm:#cfc9be; --ink:#1a1c22; --ink-soft:#6b6862; --ink-faint:#94908a;
    --k-corpus_author:#c2661d; --k-scientist:#3f5f96; --k-institution:#0d7d70;
    --k-topic:#7d4f95; --k-paper:#86817a; --accent:#c2661d; --accent-soft:#f2e2d2;
    --shadow: 0 1px 2px rgba(26,28,34,.06), 0 8px 24px rgba(26,28,34,.07);
  }
  :root[data-theme="dark"] {
    --ground:#101218; --surface:#181b22; --sunken:#0b0d12; --line:#282c36;
    --line-firm:#3a3f4c; --ink:#e8e6e1; --ink-soft:#9a978f; --ink-faint:#6e6b66;
    --k-corpus_author:#e8913f; --k-scientist:#7899d4; --k-institution:#2bab9b;
    --k-topic:#ab7cc4; --k-paper:#8f8a83; --accent:#e8913f; --accent-soft:#3a2a19;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
  }

  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ground); color: var(--ink);
    font-family: var(--font-ui); font-size: 14px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .micro {
    font-size: 10px; text-transform: uppercase; letter-spacing: .1em;
    font-weight: 650; color: var(--ink-faint);
  }
  button { font: inherit; color: inherit; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  #app { display: flex; flex-direction: column; height: 100vh; }

  /* Instrument bar, not a hero: this page is operated, not read. */
  #bar {
    display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
    padding: 12px 20px; border-bottom: 1px solid var(--line);
    background: var(--surface);
  }
  #bar h1 { margin: 0; font-size: 15px; font-weight: 640; letter-spacing: -.015em; }
  #bar h1 span.sep { color: var(--accent); }
  .stats { display: flex; gap: 18px; margin-left: auto; flex-wrap: wrap; }
  .stat { display: flex; flex-direction: column; gap: 1px; }
  .stat b {
    font-family: var(--font-mono); font-variant-numeric: tabular-nums;
    font-size: 15px; font-weight: 600; letter-spacing: -.02em;
  }

  #frame { flex: 1; display: flex; min-height: 0; }

  #rail {
    width: 226px; flex-shrink: 0; border-right: 1px solid var(--line);
    background: var(--surface); padding: 16px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 20px;
  }
  .field { display: flex; flex-direction: column; gap: 7px; }
  #search {
    padding: 7px 10px; border: 1px solid var(--line-firm); border-radius: 6px;
    background: var(--ground); color: var(--ink); font: inherit; width: 100%;
  }
  #search::placeholder { color: var(--ink-faint); }
  .legend { display: flex; flex-direction: column; gap: 2px; }
  .lg {
    display: flex; align-items: center; gap: 9px; padding: 5px 7px;
    border-radius: 6px; border: 1px solid transparent; cursor: pointer;
    background: none; text-align: left; width: 100%; font-size: 13px;
  }
  .lg:hover { background: var(--sunken); }
  .lg[aria-pressed="false"] { color: var(--ink-faint); }
  .lg[aria-pressed="false"] .swatch {
    background: transparent !important; box-shadow: inset 0 0 0 1.5px var(--line-firm);
  }
  .swatch { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .lg .n {
    margin-left: auto; font-family: var(--font-mono);
    font-variant-numeric: tabular-nums; font-size: 11px; color: var(--ink-faint);
  }
  .ctl {
    padding: 6px 11px; border-radius: 6px; border: 1px solid var(--line-firm);
    background: var(--ground); cursor: pointer; font-size: 13px; text-align: left;
  }
  .ctl:hover { border-color: var(--accent); color: var(--accent); }
  .rail-note { margin: 0; font-size: 11.5px; color: var(--ink-faint); line-height: 1.5; }

  #map { flex: 1; position: relative; min-width: 0; background: var(--ground); }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.grabbing { cursor: grabbing; }
  #tip {
    position: absolute; pointer-events: none; display: none; max-width: 250px;
    background: var(--surface); border: 1px solid var(--line-firm);
    border-radius: 7px; padding: 7px 10px; box-shadow: var(--shadow);
    font-size: 12.5px; z-index: 5;
  }
  #tip b { display: block; font-weight: 600; }

  #panel {
    width: 350px; flex-shrink: 0; border-left: 1px solid var(--line);
    background: var(--surface); overflow-y: auto; padding: 20px;
  }
  .placeholder { color: var(--ink-faint); font-size: 13px; max-width: 32ch; margin: 0; }
  .tag {
    display: inline-block; font-size: 10px; text-transform: uppercase;
    letter-spacing: .09em; font-weight: 700; padding: 3px 8px;
    border-radius: 4px; color: #fff;
  }
  #panel h2 {
    margin: 10px 0 3px; font-size: 20px; font-weight: 620;
    letter-spacing: -.022em; line-height: 1.2; text-wrap: balance;
  }
  .meta { color: var(--ink-soft); font-size: 12.5px; }
  .sect { margin-top: 22px; }
  .sect > .micro { display: block; margin-bottom: 8px; }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip {
    background: var(--sunken); border-radius: 5px; padding: 3px 8px;
    font-size: 12px; color: var(--ink-soft);
  }
  .chip.on { background: var(--accent-soft); color: var(--accent); font-weight: 550; }
  .list { display: flex; flex-direction: column; gap: 8px; }
  .item { font-size: 13px; line-height: 1.42; }
  .item .yr {
    font-family: var(--font-mono); font-size: 11.5px; color: var(--ink-faint);
  }

  /* Recommendation card: score is the headline; the four signals sit beneath
     as small multiples so the *reason* for the score reads at a glance. */
  .rec {
    display: block; width: 100%; text-align: left; cursor: pointer;
    border: 1px solid var(--line); border-radius: 9px; padding: 11px 12px;
    background: var(--ground); margin-bottom: 8px;
  }
  .rec:hover { border-color: var(--accent); }
  .rec-head {
    display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  }
  .rec-name { font-weight: 600; font-size: 13.5px; }
  .rec-score {
    font-family: var(--font-mono); font-variant-numeric: tabular-nums;
    font-size: 14px; font-weight: 600; color: var(--accent);
  }
  .rec-why {
    font-size: 12px; color: var(--ink-soft); margin-top: 4px; line-height: 1.45;
  }
  .sigs { display: flex; gap: 4px; margin-top: 10px; }
  .sig { flex: 1; }
  .sig .track {
    height: 3px; background: var(--sunken); border-radius: 2px; overflow: hidden;
  }
  .sig .fill { display: block; height: 100%; background: var(--accent); }
  .sig .lbl {
    display: block; font-size: 8.5px; text-transform: uppercase;
    letter-spacing: .06em; color: var(--ink-faint); margin-top: 4px;
  }
  .ev {
    font-family: var(--font-mono); font-size: 11px; color: var(--ink-soft);
    border-left: 2px solid var(--line-firm); padding: 2px 0 2px 9px;
    margin-bottom: 8px; line-height: 1.5;
  }
  .ev b { color: var(--ink); font-weight: 600; }

  @media (prefers-reduced-motion: no-preference) {
    .rec, .lg, .ctl { transition: border-color .12s ease, background .12s ease; }
  }
  @media (max-width: 900px) {
    #frame { flex-direction: column; }
    #rail {
      width: auto; border-right: none; border-bottom: 1px solid var(--line);
      flex-direction: row; flex-wrap: wrap; align-items: flex-start; gap: 14px;
    }
    #rail .field { flex: 1 1 170px; }
    #map { min-height: 46vh; }
    #panel {
      width: auto; border-left: none; border-top: 1px solid var(--line);
      max-height: 48vh;
    }
  }
</style>

<div id="app">
  <header id="bar">
    <h1>Dopamine research network <span class="sep">/</span>
      <span class="mono" id="corpus"></span></h1>
    <div class="stats" id="stats"></div>
  </header>

  <div id="frame">
    <aside id="rail">
      <div class="field">
        <label class="micro" for="search">Find</label>
        <input id="search" type="search" placeholder="Name, topic, institution"
               autocomplete="off">
      </div>
      <div class="field">
        <span class="micro">Show</span>
        <div class="legend" id="legend"></div>
      </div>
      <div class="field">
        <span class="micro">View</span>
        <button class="ctl" id="reset" type="button">Reset map</button>
        <button class="ctl" id="theme" type="button">Switch theme</button>
      </div>
      <p class="rail-note" id="note"></p>
    </aside>

    <div id="map">
      <canvas id="cv"></canvas>
      <div id="tip" role="tooltip"></div>
    </div>

    <aside id="panel">
      <p class="placeholder">Select anyone on the map to see their institutions,
        research interests, collaborators and suggested connections.</p>
    </aside>
  </div>
</div>

<script>
"use strict";
const DATA = __DATA__;

const KIND_LABEL = {
  corpus_author: "Paper author",
  scientist: "Cited scientist",
  institution: "Institution",
  topic: "Research topic",
  paper: "Corpus paper",
};
const KIND_ORDER = ["corpus_author", "scientist", "topic", "institution", "paper"];
const SIGNALS = [
  ["topics", "Topic"], ["profile", "Profile"],
  ["network", "Network"], ["citation", "Cited"],
];

const byId = new Map(DATA.nodes.map(n => [n.id, n]));
const near = new Map(DATA.nodes.map(n => [n.id, new Set()]));
for (const e of DATA.edges) {
  if (near.has(e.s)) near.get(e.s).add(e.t);
  if (near.has(e.t)) near.get(e.t).add(e.s);
}
const counts = {};
for (const n of DATA.nodes) counts[n.kind] = (counts[n.kind] || 0) + 1;

const shownKinds = new Set(KIND_ORDER);
let selected = null, hovered = null;
let view = { x: 0, y: 0, k: 1 };

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const tip = document.getElementById("tip");
const panel = document.getElementById("panel");

/* --- header -------------------------------------------------------------- */
document.getElementById("corpus").textContent =
  (DATA.stats.corpus_papers || 0) + " papers";
document.getElementById("stats").innerHTML = [
  ["Researchers", DATA.total_scientists.toLocaleString()],
  ["On this map", DATA.shown.toLocaleString()],
  ["Links", DATA.edges.length.toLocaleString()],
  ["Institutions", (DATA.stats.institutions || 0).toLocaleString()],
].map(([k, v]) => `<div class="stat"><span class="micro">${k}</span><b>${v}</b></div>`)
 .join("");
document.getElementById("note").textContent =
  `The ${DATA.shown.toLocaleString()} best-connected nodes of ` +
  `${DATA.total_scientists.toLocaleString()} people extracted from the papers ` +
  `and their reference lists.`;

/* --- legend doubles as the filter --------------------------------------- */
const legend = document.getElementById("legend");
for (const kind of KIND_ORDER) {
  if (!counts[kind]) continue;
  const b = document.createElement("button");
  b.className = "lg";
  b.type = "button";
  b.setAttribute("aria-pressed", "true");
  b.innerHTML =
    `<span class="swatch" style="background:var(--k-${kind})"></span>` +
    `<span>${KIND_LABEL[kind]}</span><span class="n">${counts[kind]}</span>`;
  b.onclick = () => {
    const on = b.getAttribute("aria-pressed") === "true";
    b.setAttribute("aria-pressed", String(!on));
    if (on) shownKinds.delete(kind); else shownKinds.add(kind);
    draw();
  };
  legend.appendChild(b);
}

document.getElementById("reset").onclick = () => {
  view = { x: 0, y: 0, k: 1 };
  selected = null;
  renderPanel(null);
  draw();
};
document.getElementById("theme").onclick = () => {
  const root = document.documentElement;
  const cur = root.getAttribute("data-theme");
  const dark = cur ? cur === "dark"
                   : matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-theme", dark ? "light" : "dark");
  paint = readPalette();
  draw();
};

const search = document.getElementById("search");
search.addEventListener("input", () => {
  const q = search.value.trim().toLowerCase();
  if (!q) return;
  const hit = DATA.nodes.find(n => n.label.toLowerCase().includes(q));
  if (hit) focusNode(hit);
});

/* --- palette read from CSS, so the canvas re-themes with the page -------- */
function readPalette() {
  const s = getComputedStyle(document.documentElement);
  const p = { kind: {} };
  for (const k of KIND_ORDER) p.kind[k] = s.getPropertyValue("--k-" + k).trim();
  for (const v of ["ink", "ink-soft", "ink-faint", "line", "line-firm",
                   "ground", "accent"]) {
    p[v] = s.getPropertyValue("--" + v).trim();
  }
  return p;
}
let paint = readPalette();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  paint = readPalette();
  draw();
});

/* --- geometry ------------------------------------------------------------ */
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.parentElement.getBoundingClientRect();
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: r.width, h: r.height };
}
let dims = resize();
const PAD = 52;

function screenOf(n) {
  return {
    x: (n.x * (dims.w - 2 * PAD) + PAD) * view.k + view.x,
    y: (n.y * (dims.h - 2 * PAD) + PAD) * view.k + view.y,
  };
}
function radiusOf(n) {
  if (n.kind === "topic") return 8 + Math.min(Math.sqrt(n.degree) * 1.4, 11);
  if (n.kind === "paper") return 8.5;
  if (n.kind === "institution") return 6.5 + Math.min(n.degree, 6);
  return 3.6 + Math.min(Math.sqrt(n.degree + n.cited) * 1.05, 10);
}
function scaleOf(n) {
  return radiusOf(n) * Math.min(Math.max(view.k, 0.6), 1.85);
}
const isVisible = n => shownKinds.has(n.kind);

/* --- drawing ------------------------------------------------------------- */
let queued = false;
function draw() {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; render(); });
}

function render() {
  ctx.clearRect(0, 0, dims.w, dims.h);
  const focus = selected || hovered;
  const ring = focus ? near.get(focus.id) : null;

  for (const e of DATA.edges) {
    const a = byId.get(e.s), b = byId.get(e.t);
    if (!a || !b || !isVisible(a) || !isVisible(b)) continue;
    const active = focus && (e.s === focus.id || e.t === focus.id);
    ctx.globalAlpha = focus ? (active ? 0.8 : 0.045) : 0.14;
    ctx.strokeStyle = active ? paint.accent : paint["line-firm"];
    ctx.lineWidth = active ? 1.5 : 1;
    const p = screenOf(a), q = screenOf(b);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(q.x, q.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  const labels = [];
  for (const n of DATA.nodes) {
    if (!isVisible(n)) continue;
    const p = screenOf(n), r = scaleOf(n);
    const dim = focus && n.id !== focus.id && !(ring && ring.has(n.id));
    ctx.globalAlpha = dim ? 0.14 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = paint.kind[n.kind];
    ctx.fill();
    if (n === focus) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = paint.ink;
      ctx.stroke();
    }
    const worthLabel = n.kind === "topic" || n.kind === "corpus_author"
                    || n.kind === "paper" || view.k > 1.75 || r > 10;
    if (!dim && (worthLabel || n === focus)) labels.push({ n, p, r });
  }
  ctx.globalAlpha = 1;

  const family = getComputedStyle(document.body).fontFamily;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const item of labels) {
    const emph = item.n.kind === "topic" || item.n === focus;
    ctx.font = (emph ? "600 " : "") + "11px " + family;
    let text = item.n.label;
    if (text.length > 32) text = text.slice(0, 31) + "…";
    // Halo keeps labels legible where they cross edges, in either theme.
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = paint.ground;
    ctx.strokeText(text, item.p.x, item.p.y + item.r + 4);
    ctx.fillStyle = emph ? paint.ink : paint["ink-soft"];
    ctx.fillText(text, item.p.x, item.p.y + item.r + 4);
  }
}

/* --- interaction --------------------------------------------------------- */
function nodeAt(mx, my) {
  let best = null, bestD = Infinity;
  for (const n of DATA.nodes) {
    if (!isVisible(n)) continue;
    const p = screenOf(n), r = scaleOf(n) + 4;
    const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
    if (d <= r * r && d < bestD) { best = n; bestD = d; }
  }
  return best;
}

let dragging = false, moved = false, last = null;
cv.addEventListener("pointerdown", ev => {
  dragging = true;
  moved = false;
  last = { x: ev.clientX, y: ev.clientY };
  cv.classList.add("grabbing");
  cv.setPointerCapture(ev.pointerId);
});
cv.addEventListener("pointermove", ev => {
  const rect = cv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  if (dragging) {
    const dx = ev.clientX - last.x, dy = ev.clientY - last.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    view.x += dx;
    view.y += dy;
    last = { x: ev.clientX, y: ev.clientY };
    draw();
    return;
  }
  const hit = nodeAt(mx, my);
  if (hit !== hovered) { hovered = hit; draw(); }
  if (hit) {
    const sub = (hit.kind === "scientist" || hit.kind === "corpus_author")
      ? KIND_LABEL[hit.kind] + " · " + hit.degree + " co-authors"
      : KIND_LABEL[hit.kind];
    tip.style.display = "block";
    tip.style.left = Math.min(mx + 14, rect.width - 262) + "px";
    tip.style.top = (my + 16) + "px";
    tip.innerHTML = `<b>${esc(hit.label)}</b>` +
                    `<span class="meta">${esc(sub)}</span>`;
  } else {
    tip.style.display = "none";
  }
});
cv.addEventListener("pointerup", ev => {
  if (!dragging) return;
  dragging = false;
  cv.classList.remove("grabbing");
  if (!moved) {
    const rect = cv.getBoundingClientRect();
    selected = nodeAt(ev.clientX - rect.left, ev.clientY - rect.top);
    renderPanel(selected);
    draw();
  }
});
cv.addEventListener("pointercancel", () => {
  dragging = false;
  cv.classList.remove("grabbing");
});
cv.addEventListener("wheel", ev => {
  ev.preventDefault();
  const rect = cv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const next = Math.min(Math.max(view.k * Math.exp(-ev.deltaY * 0.0015), 0.4), 9);
  // Zoom toward the cursor: whatever is under the pointer stays put.
  view.x = mx - (mx - view.x) * (next / view.k);
  view.y = my - (my - view.y) * (next / view.k);
  view.k = next;
  draw();
}, { passive: false });

function focusNode(n) {
  selected = n;
  view.k = Math.max(view.k, 2);
  const p = screenOf(n);
  view.x += dims.w / 2 - p.x;
  view.y += dims.h / 2 - p.y;
  renderPanel(n);
  draw();
}
window.addEventListener("resize", () => { dims = resize(); draw(); });

/* --- inspector ----------------------------------------------------------- */
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function block(label, body) {
  return body ? `<div class="sect"><span class="micro">${label}</span>${body}</div>` : "";
}
function chips(items, highlight) {
  if (!items || !items.length) return "";
  const cls = highlight ? "chip on" : "chip";
  return `<div class="chips">${items.map(t =>
    `<span class="${cls}">${esc(t)}</span>`).join("")}</div>`;
}

function renderPanel(n) {
  if (!n) {
    panel.innerHTML = '<p class="placeholder">Select anyone on the map to see ' +
      'their institutions, research interests, collaborators and suggested ' +
      'connections.</p>';
    return;
  }
  const out = [];
  out.push(`<span class="tag" style="background:var(--k-${n.kind})">` +
           `${KIND_LABEL[n.kind]}</span>`);
  out.push(`<h2>${esc(n.label)}</h2>`);

  const meta = [];
  if (n.kind === "scientist" || n.kind === "corpus_author") {
    meta.push(n.degree + " co-authors");
    if (n.cited) meta.push(n.cited + " referenced works");
  } else if (n.degree) {
    meta.push(n.degree + " people");
  }
  if (n.country) meta.push(n.country);
  if (n.description) meta.push(n.description);
  if (meta.length) out.push(`<div class="meta">${esc(meta.join(" · "))}</div>`);

  const x = n.external || {};
  if (x.works_count) {
    const bits = [x.works_count + " works",
                  (x.cited_by_count || 0).toLocaleString() + " citations"];
    if (x.h_index) bits.push("h-index " + x.h_index);
    if (x.orcid) bits.push(x.orcid.replace("https://orcid.org/", "ORCID "));
    out.push(`<div class="meta mono" style="margin-top:5px">` +
             `${esc(bits.join(" · "))}</div>`);
  }

  out.push(block("Institutions", chips(n.institutions, false)));
  out.push(block("Research interests", chips(n.topics, true)));

  if (n.papers && n.papers.length) {
    out.push(block("Papers", `<div class="list">${n.papers.map(p =>
      `<div class="item">${esc(p.title)}` +
      (p.year ? ` <span class="yr">${p.year}</span>` : "") +
      (p.corpus ? ` <span class="chip on">in corpus</span>` : "") +
      `</div>`).join("")}</div>`));
  }
  if (n.collaborators && n.collaborators.length) {
    out.push(block("Collaborators", chips(n.collaborators, false)));
  }

  if (n.recs && n.recs.length) {
    out.push(block("Suggested connections", n.recs.map(r => {
      const sigs = SIGNALS.map(pair => {
        const raw = (r.sig && r.sig[pair[0]] != null) ? r.sig[pair[0]] : 0;
        const pct = Math.round(Math.max(0, Math.min(raw, 1)) * 100);
        return `<div class="sig"><div class="track">` +
               `<i class="fill" style="width:${pct}%"></i></div>` +
               `<span class="lbl">${pair[1]}</span></div>`;
      }).join("");
      return `<button class="rec" type="button" data-goto="${esc(r.id)}">` +
        `<div class="rec-head"><span class="rec-name">${esc(r.name)}</span>` +
        `<span class="rec-score">${r.score.toFixed(2)}</span></div>` +
        `<div class="rec-why">${esc(r.why)}</div>` +
        `<div class="sigs">${sigs}</div></button>`;
    }).join("")));
  }

  if (n.evidence && n.evidence.length) {
    out.push(block("Extracted from", n.evidence.map(e =>
      `<div class="ev"><b>${esc(e.file)}</b> p.${e.page}<br>${esc(e.snippet)}</div>`
    ).join("")));
  }

  panel.innerHTML = out.join("");
  panel.scrollTop = 0;
  panel.querySelectorAll("[data-goto]").forEach(el => {
    el.onclick = () => {
      const target = byId.get(el.getAttribute("data-goto"));
      if (target) focusNode(target);
    };
  });
}

draw();
</script>
"""
