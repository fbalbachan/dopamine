"""CLI: download the nine papers and build the hybrid RAG index.

    python -m crew.build_index          # build if missing / stale
    python -m crew.build_index --force  # force a full rebuild
"""
from __future__ import annotations

import argparse
import sys

from .rag import build_index

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dopamine RAG index.")
    parser.add_argument("--force", action="store_true", help="force a full rebuild")
    args = parser.parse_args()

    print("Downloading papers and building the hybrid index (this can take a "
          "minute on first run — the embedding model is downloaded once)…")
    retriever = build_index(force=args.force)
    m = retriever.manifest
    print(f"\nIndex ready: {m.get('count')} chunks from {len(m.get('papers', []))} papers.")
    print(f"Embedding model: {m.get('embed_model')} (dim {m.get('dim')}).")


if __name__ == "__main__":
    main()
