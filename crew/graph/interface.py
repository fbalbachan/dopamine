"""Explore the scientist graph from the command line.

    python -m crew.graph.interface                              # REPL
    python -m crew.graph.interface --who "Wolfram Schultz"      # one profile
    python -m crew.graph.interface --recommend "Josh Berke"     # connections
    python -m crew.graph.interface --similar "Marc Caron"       # similar profiles
    python -m crew.graph.interface --topic "Parkinson"          # who works on it
    python -m crew.graph.interface --institution "Duke"         # who is there
    python -m crew.graph.interface --stats                      # graph summary
    python -m crew.graph.interface -q "Who should Berke meet?"  # natural language
    python -m crew.graph.interface --viz                        # open the viewer

REPL commands: ``/who <name>``  ``/rec <name>``  ``/similar <name>``
``/topic <t>``  ``/inst <i>``  ``/stats``  ``/viz``  ``/json``  ``/quit``.
Anything else is treated as a natural-language question.

The structured lookups (``--who``, ``--recommend``, …) are pure graph queries
and need no API key; only the natural-language path calls Claude.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
import webbrowser
from typing import List, Optional

# Author names carry accents, CJK characters and ligatures that the Windows
# console's default cp1252 cannot encode.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from ..config import settings
from .recommend import Recommender
from .schema import Recommendation
from .store import ScientistGraph

BAR_WIDTH = 18


def _bar(value: float) -> str:
    filled = int(round(max(0.0, min(value, 1.0)) * BAR_WIDTH))
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def _resolve_one(graph: ScientistGraph, name: str) -> Optional[str]:
    matches = graph.find_scientist(name)
    if not matches:
        print(f"No scientist matching '{name}'. Try /stats to see who is in the graph.")
        return None
    if len(matches) > 1 and matches[0][1] < 0.95:
        print("Did you mean:")
        for sid, score in matches[:5]:
            print(f"  · {graph.scientists[sid]['name']}  ({score:.2f})")
    return matches[0][0]


def print_profile(graph: ScientistGraph, sid: str) -> None:
    s = graph.scientists[sid]
    kind = "corpus author" if s["corpus_author"] else "cited researcher"
    print(f"\n{s['name']}  ({kind})")
    print("─" * (len(s["name"]) + len(kind) + 5))

    external = s.get("external") or {}
    if external.get("works_count"):
        from .enrich import external_summary

        print(f"  OpenAlex     {external_summary(s)}")

    institutions = [i["name"] for i in graph.institutions_of(sid)]
    if institutions:
        print(f"  Institutions {', '.join(institutions)}")
    topics = [graph.topics[t]["label"] for t in s["topics"] if t in graph.topics]
    if topics:
        print(f"  Interests    {', '.join(topics)}")
    print(f"  Network      {s['degree']} co-authors · "
          f"{s['cited_works']} referenced works · {len(s['papers'])} papers")

    papers = graph.papers_of(sid)
    if papers:
        print("\n  Papers")
        for p in papers[:6]:
            year = f" ({p['year']})" if p["year"] else ""
            tag = "  [corpus]" if p["in_corpus"] else ""
            print(f"    · {p['title'] or p['id']}{year}{tag}")

    collaborators = graph.collaborators(sid)
    if collaborators:
        names = [graph.scientists[c]["name"] for c in collaborators[:12]
                 if c in graph.scientists]
        more = f" (+{len(collaborators) - 12} more)" if len(collaborators) > 12 else ""
        print(f"\n  Collaborators\n    {', '.join(names)}{more}")

    if s["evidence"]:
        print("\n  Evidence")
        for ev in s["evidence"][:2]:
            print(f"    · {ev['file']} p.{ev['page']}")


def print_recommendations(recs: List[Recommendation], header: str) -> None:
    print(f"\n{header}")
    print("─" * len(header))
    if not recs:
        print("  (nothing scored above zero — the graph may be too sparse here)")
        return
    for i, r in enumerate(recs, start=1):
        print(f"\n  {i}. {r['name']}   {r['score']:.3f}  {_bar(r['score'])}")
        if r["institutions"]:
            print(f"     at {', '.join(r['institutions'])}")
        if r["shared_topics"]:
            print(f"     shared interests: {', '.join(r['shared_topics'])}")
        if r["shared_collaborators"]:
            print(f"     via: {', '.join(r['shared_collaborators'])}")
        sig = r["signals"]
        print(f"     topics {sig['topic_overlap']:.2f} · "
              f"profile {sig['profile_similarity']:.2f} · "
              f"network {sig['graph_proximity']:.2f} · "
              f"citation {sig['citation_affinity']:.2f}"
              + ("  · same institution" if sig["same_institution"] else ""))


def print_stats(graph: ScientistGraph) -> None:
    st = graph.stats or {}
    print(f"\n{graph.summary()}")
    if st.get("relations"):
        print("\n  Relations")
        for rel, count in sorted(st["relations"].items(), key=lambda kv: -kv[1]):
            print(f"    {rel:16s} {count:,}")
    print("\n  Most connected")
    for sid in graph.top_scientists(12):
        s = graph.scientists[sid]
        mark = "*" if s["corpus_author"] else " "
        print(f"    {mark} {s['name']:32s} {s['degree']:4d} co-authors")
    print("\n  Research topics")
    for t in sorted(graph.topics.values(), key=lambda t: -len(t["scientists"]))[:12]:
        print(f"      {t['label']:48s} {len(t['scientists']):4d} people")


def _open_viewer() -> None:
    path = settings.graph_dir / "graph.html"
    if not path.exists():
        print("No viewer yet — run `python -m crew.graph.build` first.")
        return
    print(f"Opening {path}")
    webbrowser.open(path.as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the dopamine scientist graph.")
    parser.add_argument("--who", help="show a scientist's profile")
    parser.add_argument("--recommend", help="suggest new connections for a scientist")
    parser.add_argument("--similar", help="find scientists with similar profiles")
    parser.add_argument("--topic", help="who works on a research topic")
    parser.add_argument("--institution", help="who is at an institution")
    parser.add_argument("--stats", action="store_true", help="graph summary")
    parser.add_argument("--viz", action="store_true", help="open the HTML viewer")
    parser.add_argument("-q", "--question", help="ask a natural-language question")
    parser.add_argument("-n", "--limit", type=int, default=8, help="results to show")
    parser.add_argument("-s", "--session", default=None, help="session id")
    parser.add_argument("--json", action="store_true", help="raw JSON output")
    args = parser.parse_args()

    if args.viz and not any([args.who, args.recommend, args.similar, args.topic,
                             args.institution, args.stats, args.question]):
        _open_viewer()
        return

    try:
        graph = ScientistGraph.load()
    except FileNotFoundError as exc:
        print(exc)
        return

    # The embedder is only needed for profile similarity — load it lazily so
    # plain lookups (--who, --topic, --stats) stay instant.
    cached: List[Recommender] = []

    def recommender() -> Recommender:
        if not cached:
            from .embedding import default_embedder

            cached.append(Recommender(graph, default_embedder()))
        return cached[0]

    did_something = False

    if args.stats:
        print_stats(graph)
        did_something = True

    if args.who:
        sid = _resolve_one(graph, args.who)
        if sid:
            if args.json:
                print(json.dumps(graph.scientists[sid], indent=2, ensure_ascii=False))
            else:
                print_profile(graph, sid)
        did_something = True

    if args.recommend:
        sid = _resolve_one(graph, args.recommend)
        if sid:
            recs = recommender().score_all(sid, exclude_collaborators=True,
                                           limit=args.limit)
            if args.json:
                print(json.dumps(recs, indent=2, ensure_ascii=False))
            else:
                print_recommendations(
                    recs, f"Suggested connections for {graph.scientists[sid]['name']}"
                )
        did_something = True

    if args.similar:
        sid = _resolve_one(graph, args.similar)
        if sid:
            recs = recommender().score_all(sid, exclude_collaborators=False,
                                           limit=args.limit)
            if args.json:
                print(json.dumps(recs, indent=2, ensure_ascii=False))
            else:
                print_recommendations(
                    recs, f"Scientists similar to {graph.scientists[sid]['name']}"
                )
        did_something = True

    if args.topic:
        tid = graph.find_topic(args.topic)
        if not tid:
            print(f"No topic matching '{args.topic}'.")
        else:
            topic = graph.topics[tid]
            print(f"\n{topic['label']}\n{'─' * len(topic['label'])}")
            print(f"  {topic['description']}\n")
            people = sorted(
                (graph.scientists[s] for s in topic["scientists"]
                 if s in graph.scientists),
                key=lambda s: (s["corpus_author"], s["degree"]), reverse=True,
            )
            for s in people[:args.limit * 3]:
                mark = "*" if s["corpus_author"] else " "
                print(f"  {mark} {s['name']}")
            print(f"\n  {len(people)} researchers total")
        did_something = True

    if args.institution:
        iid = graph.find_institution(args.institution)
        if not iid:
            print(f"No institution matching '{args.institution}'.")
        else:
            inst = graph.institutions[iid]
            print(f"\n{inst['name']}"
                  f"{' · ' + inst['country'] if inst['country'] else ''}")
            print("─" * len(inst["name"]))
            if inst["departments"]:
                print(f"  {'; '.join(inst['departments'][:4])}\n")
            for sid in inst["scientists"]:
                if sid in graph.scientists:
                    print(f"  · {graph.scientists[sid]['name']}")
        did_something = True

    if args.question:
        _ask(graph, args.question, args.session or uuid.uuid4().hex[:8], args.json)
        did_something = True

    if args.viz:
        _open_viewer()
        did_something = True

    if not did_something:
        _repl(graph, recommender, args.session or uuid.uuid4().hex[:8], args.json)


def _ask(graph: ScientistGraph, question: str, session_id: str, as_json: bool) -> None:
    from .chat import GraphCrew

    crew = GraphCrew(graph)
    resp = crew.ask(question, session_id=session_id)
    if as_json:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return
    tag = "REFUSED" if resp["refused"] else f"{resp['confidence'].upper()} confidence"
    print(f"\n[{tag} · intent={resp['intent']}]\n{resp['answer']}\n")
    if resp["sources"]:
        print("Sources:")
        for i, src in enumerate(resp["sources"]):
            where = (f"{src['file']} p.{src['page']}"
                     if src["file"] != "graph" else "knowledge graph")
            print(f"  [{i}] {where}")
    if resp["recommendations"]:
        print_recommendations(resp["recommendations"], "Scored recommendations")
    print()


def _repl(graph: ScientistGraph, recommender, session_id: str, as_json: bool) -> None:
    print(f"Scientist graph ready — {graph.summary()}")
    print("Commands: /who /rec /similar /topic /inst /stats /viz /json /quit")
    print("Or just ask a question in plain English.\n")
    while True:
        try:
            line = input("graph> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/stats":
            print_stats(graph)
            continue
        if line == "/viz":
            _open_viewer()
            continue
        if line == "/json":
            as_json = not as_json
            print(f"json output = {as_json}")
            continue

        command, _, argument = line.partition(" ")
        argument = argument.strip()
        if command == "/who" and argument:
            sid = _resolve_one(graph, argument)
            if sid:
                print_profile(graph, sid)
        elif command in ("/rec", "/recommend") and argument:
            sid = _resolve_one(graph, argument)
            if sid:
                print_recommendations(
                    recommender().score_all(sid, exclude_collaborators=True, limit=8),
                    f"Suggested connections for {graph.scientists[sid]['name']}",
                )
        elif command == "/similar" and argument:
            sid = _resolve_one(graph, argument)
            if sid:
                print_recommendations(
                    recommender().score_all(sid, exclude_collaborators=False, limit=8),
                    f"Scientists similar to {graph.scientists[sid]['name']}",
                )
        elif command == "/topic" and argument:
            tid = graph.find_topic(argument)
            if not tid:
                print(f"No topic matching '{argument}'.")
            else:
                topic = graph.topics[tid]
                print(f"\n{topic['label']} — {len(topic['scientists'])} researchers")
                for sid in topic["scientists"][:25]:
                    if sid in graph.scientists:
                        print(f"  · {graph.scientists[sid]['name']}")
        elif command == "/inst" and argument:
            iid = graph.find_institution(argument)
            if not iid:
                print(f"No institution matching '{argument}'.")
            else:
                inst = graph.institutions[iid]
                print(f"\n{inst['name']}")
                for sid in inst["scientists"]:
                    if sid in graph.scientists:
                        print(f"  · {graph.scientists[sid]['name']}")
        else:
            _ask(graph, line, session_id, as_json)


if __name__ == "__main__":
    main()
