"""Simple interface to chat with the crew.

Interactive (multi-turn, memory within the session):

    python -m crew.interface

One-shot (prints the full typed dict as JSON):

    python -m crew.interface -q "What is dopamine's role in reward learning?" --json

REPL commands: /new  (start a fresh session)   /session <id>   /json  /quit
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import List

# Answers contain neuroscience glyphs (α, β, ligatures) the Windows console's
# default cp1252 can't encode. Force UTF-8 so printing never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from .chat import DopamineCrew
from .schema import CrewResponse, Source


def _format_sources(sources: List[Source]) -> str:
    if not sources:
        return "  (no sources)"
    lines = []
    for i, s in enumerate(sources):
        if s["type"] == "paper":
            lines.append(f"  [{i}] paper · {s['file']} p.{s['page']}")
        else:
            lines.append(f"  [{i}] web   · {s['title']} — {s['url']}")
    return "\n".join(lines)


def _print_human(resp: CrewResponse) -> None:
    tag = "REFUSED" if resp["refused"] else f"{resp['confidence'].upper()} confidence"
    web = " · web used" if resp["used_web_search"] else ""
    print(f"\n[{tag}{web}]\n{resp['answer']}\n")
    print("Sources:")
    print(_format_sources(resp["sources"]))
    if resp["claims"]:
        print("\nClaim → sources:")
        for c in resp["claims"]:
            refs = ", ".join(f"[{i}]" for i in c["source_indices"]) or "(unsupported)"
            print(f"  • {c['statement']}  {refs}")
    print()


def _run_once(crew: DopamineCrew, question: str, session_id: str, as_json: bool) -> None:
    resp = crew.ask(question, session_id=session_id)
    if as_json:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    else:
        _print_human(resp)


def _repl(crew: DopamineCrew, session_id: str, as_json: bool) -> None:
    print("Dopamine crew ready. Ask a question, or /quit to exit.")
    print(f"session_id = {session_id}\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/new":
            session_id = uuid.uuid4().hex[:8]
            print(f"new session_id = {session_id}")
            continue
        if line.startswith("/session"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                session_id = parts[1].strip()
                print(f"session_id = {session_id}")
            continue
        if line == "/json":
            as_json = not as_json
            print(f"json output = {as_json}")
            continue
        _run_once(crew, line, session_id, as_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the dopamine crew.")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("-s", "--session", default=None, help="session id")
    parser.add_argument("--json", action="store_true", help="print the raw typed dict")
    args = parser.parse_args()

    session_id = args.session or uuid.uuid4().hex[:8]
    crew = DopamineCrew()

    if args.question:
        _run_once(crew, args.question, session_id, args.json)
    else:
        _repl(crew, session_id, args.json)


if __name__ == "__main__":
    main()
