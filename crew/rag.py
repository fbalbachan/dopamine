"""RAG layer over the nine dopamine papers.

Pipeline
--------
1. **Download** the PDFs from the challenge repo into ``.papers/``.
2. **Extract** text page-by-page (pypdf) so every chunk keeps its 1-indexed
   page number -> citations resolve to ``file + page``.
3. **Chunk** each page into overlapping character windows.
4. **Embed** chunks with a local sentence-transformers model (no API key) and
   store them in a cosine FAISS index; keep a BM25 lexical index too.
5. **Search** with a hybrid of dense + BM25 fused by Reciprocal Rank Fusion —
   the lexical half matters for exact tokens (receptor names, "D2", "L-DOPA").

The index is persisted to ``.index/`` and rebuilt only when the corpus changes.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

from .config import PAPER_FILENAMES, PAPERS_REPO_RAW, settings

_WORD_RE = re.compile(r"[A-Za-z0-9\-']+")


@dataclass
class Chunk:
    file: str
    page: int          # 1-indexed
    text: str


@dataclass
class Hit:
    file: str
    page: int
    snippet: str
    score: float


# --------------------------------------------------------------------------- #
# Corpus acquisition + preprocessing
# --------------------------------------------------------------------------- #
def download_papers(force: bool = False) -> List[Path]:
    """Download any missing PDFs. Returns local paths (in canonical order)."""
    settings.ensure_dirs()
    paths: List[Path] = []
    for name in PAPER_FILENAMES:
        dest = settings.papers_dir / name
        if dest.exists() and not force:
            paths.append(dest)
            continue
        url = f"{PAPERS_REPO_RAW}/{urllib.parse.quote(name)}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        paths.append(dest)
    return paths


def _extract_pages(pdf_path: Path) -> List[str]:
    """Return a list of page texts (index i == page i+1)."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def _chunk_page(text: str) -> List[str]:
    """Split one page's text into overlapping character windows."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    size, overlap = settings.chunk_chars, settings.chunk_overlap
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _corpus_hash(paths: List[Path]) -> str:
    h = sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    h.update(settings.embed_model.encode())
    h.update(f"{settings.chunk_chars}-{settings.chunk_overlap}".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Index build
# --------------------------------------------------------------------------- #
def build_index(force: bool = False) -> "Retriever":
    """Build (or reuse) the persisted hybrid index and return a Retriever."""
    import faiss
    from sentence_transformers import SentenceTransformer

    settings.ensure_dirs()
    paths = download_papers(force=force)
    manifest_path = settings.index_dir / "manifest.json"
    chunks_path = settings.index_dir / "chunks.json"
    faiss_path = settings.index_dir / "faiss.index"
    corpus_hash = _corpus_hash(paths)

    if (
        not force
        and manifest_path.exists()
        and chunks_path.exists()
        and faiss_path.exists()
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("corpus_hash") == corpus_hash:
            return Retriever.load()

    # (Re)build.
    chunks: List[Chunk] = []
    for path in paths:
        for page_no, page_text in enumerate(_extract_pages(path), start=1):
            for piece in _chunk_page(page_text):
                chunks.append(Chunk(file=path.name, page=page_no, text=piece))

    if not chunks:
        raise RuntimeError("No text extracted from the PDFs — cannot build index.")

    embedder = SentenceTransformer(settings.embed_model)
    embeddings = embedder.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,          # cosine == inner product
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(faiss_path))
    # UTF-8 explicitly: PDF text contains ligatures/non-Latin-1 chars that the
    # Windows default (cp1252) can't encode.
    chunks_path.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_hash": corpus_hash,
                "embed_model": settings.embed_model,
                "dim": int(embeddings.shape[1]),
                "count": len(chunks),
                "papers": [p.name for p in paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Retriever.load()


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """Hybrid dense + BM25 retriever over the persisted index."""

    def __init__(self, chunks: List[Chunk], faiss_index, manifest: dict):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.index = faiss_index
        self.manifest = manifest
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        self._embedder = None  # lazy

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls) -> "Retriever":
        import faiss

        chunks_path = settings.index_dir / "chunks.json"
        faiss_path = settings.index_dir / "faiss.index"
        manifest_path = settings.index_dir / "manifest.json"
        if not (chunks_path.exists() and faiss_path.exists()):
            raise FileNotFoundError(
                "No index found. Run `python -m crew.build_index` first."
            )
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in raw]
        index = faiss.read_index(str(faiss_path))
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        return cls(chunks, index, manifest)

    # -- embedding -----------------------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embed_model)
        return self._embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

    # -- scope signal --------------------------------------------------------
    def max_similarity(self, query: str) -> float:
        """Best cosine similarity between the query and any chunk (for scope)."""
        q = self._embed([query])
        scores, _ = self.index.search(q, 1)
        return float(scores[0][0]) if len(scores[0]) else 0.0

    # -- hybrid search -------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None) -> List[Hit]:
        k = k or settings.top_k
        pool = max(k * 4, 20)

        # Dense.
        q = self._embed([query])
        dscores, didx = self.index.search(q, min(pool, len(self.chunks)))
        dense_ranked = [int(i) for i in didx[0] if i >= 0]

        # Lexical.
        bm = self._bm25.get_scores(_tokenize(query))
        lex_ranked = list(np.argsort(bm)[::-1][:pool])

        # Reciprocal Rank Fusion.
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
        for rank, idx in enumerate(lex_ranked):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank + 1)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
        hits: List[Hit] = []
        for idx, score in fused:
            c = self.chunks[idx]
            snippet = c.text if len(c.text) <= 400 else c.text[:400] + "…"
            hits.append(Hit(file=c.file, page=c.page, snippet=snippet, score=round(score, 5)))
        return hits
