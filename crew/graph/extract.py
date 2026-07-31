"""Entity extraction from the nine PDFs.

Two passes per paper, over two very different kinds of text:

**Front matter** (pages 1-2) yields the rich profiles — full names, departments,
organisations, countries, emails. This is where corpus authors get everything a
cited author never has.

**Reference lists** yield the network. Nine papers use nine citation styles
(``Berke JD:``, ``[1] A.I. Levey,``, ``469. WISE, R. A., AND P. P. ROMPRE.``,
two-column layouts that interleave two references per extracted line). Writing a
regex per style is exactly the brittle work an LLM removes, so a cheap model
does the parsing and returns structured records.

The model never invents graph structure: it returns *surface forms*, which
:mod:`crew.graph.canon` resolves to ids and :mod:`crew.graph.build` turns into
nodes and edges. Every record carries the page it came from, so provenance
survives all the way to a recommendation's ``evidence`` list.

Without an API key the reference pass falls back to a heuristic splitter — less
accurate on titles and venues, but it still recovers author lists, so the graph
builds fully offline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import BaseModel, Field, model_validator

from ..config import settings


class _Lenient(BaseModel):
    """Base model that treats a JSON ``null`` as "not present".

    Extraction prompts explicitly tell the model to use ``null`` when a field
    isn't printed — and plenty of papers have no DOI on page 1. Without this,
    ``{"doi": null}`` fails validation on a ``str`` field, the caller's
    ``except`` swallows it, and the paper silently loses *every* author. That
    is exactly how the 1998 Missale review lost its five Duke authors on an
    early build, so the leniency is load-bearing, not cosmetic.
    """

    @model_validator(mode="before")
    @classmethod
    def _nulls_to_blank(cls, data):
        if not isinstance(data, dict):
            return data
        cleaned = {}
        for key, value in data.items():
            field = cls.model_fields.get(key)
            if value is None and field is not None and field.annotation is str:
                value = ""
            cleaned[key] = value
        return cleaned

# Where a reference/bibliography section begins.
_REF_HEADING_RE = re.compile(
    r"^\s*(references?|bibliography|literature\s+cited|works\s+cited)\s*:?\s*$",
    re.I | re.M,
)
# Numbered reference entries: "1.", "[1]", " 12 ." at the start of a line.
_REF_ENTRY_RE = re.compile(r"(?m)^\s*\[?(\d{1,3})[\.\]]?\s+(?=[A-ZÀ-ɏ])")


class ExtractedAuthor(_Lenient):
    name: str = Field(..., description="Author name exactly as printed.")
    organization: str = Field("", description="Top-level org (university/company).")
    department: str = Field("", description="Department, school or institute.")
    country: str = Field("", description="Country, if printed.")
    email: str = Field("", description="Email, if printed.")


class PaperMeta(_Lenient):
    """Front-matter record for one of the nine papers."""
    title: str = ""
    authors: List[ExtractedAuthor] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    abstract: str = ""


class CitedWork(_Lenient):
    """One entry from a reference list."""
    authors: List[str] = Field(default_factory=list, description="Surface forms.")
    title: str = ""
    venue: str = ""
    year: Optional[int] = None
    page: int = 0          # page of the citing PDF this entry was read from


def _complete(system: str, user: str, max_tokens: int = 8000) -> str:
    """One call to the cheap extraction model. Raises on failure."""
    from litellm import completion

    resp = completion(
        model=settings.litellm_graph_extract_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp["choices"][0]["message"]["content"]


def _json_slice(raw: str, opener: str = "[", closer: str = "]"):
    start, end = raw.find(opener), raw.rfind(closer)
    if start == -1 or end == -1:
        raise ValueError("no JSON found in model output")
    return json.loads(raw[start : end + 1])


# --------------------------------------------------------------------------- #
# Front matter
# --------------------------------------------------------------------------- #
def extract_front_matter(pages: List[str], filename: str,
                         warn: Optional[Callable[[str], None]] = None) -> PaperMeta:
    """Parse title/authors/affiliations from the first pages of a paper.

    ``warn`` is called with a reason if extraction fails. Failing loudly matters
    here: an empty result costs a paper every author, affiliation and topic it
    would have contributed, and the graph still builds — just quietly wrong.
    """
    head = "\n".join(pages[:2])[:6000]
    if not settings.anthropic_api_key or not head.strip():
        return PaperMeta()
    try:
        raw = _complete(
            system=(
                "You extract bibliographic metadata from the first page of a "
                "research paper. Return ONLY a JSON object. Never invent a "
                "value — use \"\" or null when something is not printed."
            ),
            user=(
                f"First pages of {filename}:\n\n{head}\n\n"
                "Return JSON:\n"
                "{\"title\": str, \"year\": int|null, \"venue\": str, "
                "\"doi\": str, \"abstract\": str (first 500 chars if present), "
                "\"authors\": [{\"name\": str, \"organization\": str, "
                "\"department\": str, \"country\": str, \"email\": str}]}\n\n"
                "Rules:\n"
                "- `authors` are the authors OF THIS PAPER, in printed order. "
                "Ignore editors, reviewers, and names in the reference list.\n"
                "- `organization` is the top-level body (e.g. 'University of "
                "Cambridge'), `department` the sub-unit (e.g. 'Department of "
                "Physiology'). Split them even when printed as one string.\n"
                "- Some PDFs extract without spaces between words "
                "(\"UniversityofNottingham\") — restore the spaces.\n"
                "- Match each author to their own affiliation using the "
                "superscript markers when present.\n"
                "- If this page is a ResearchGate cover sheet, use the author "
                "and institution names it lists."
            ),
            max_tokens=3000,
        )
        meta = PaperMeta(**_json_slice(raw, "{", "}"))
        if not meta.authors and warn:
            warn(f"    ! no authors found in the front matter of {filename}")
        return meta
    except Exception as exc:
        if warn:
            warn(f"    ! front-matter extraction failed for {filename}: "
                 f"{type(exc).__name__}: {exc}")
        return PaperMeta()


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #
def find_reference_pages(pages: List[str]) -> List[int]:
    """Return the 0-indexed pages that hold the reference list.

    Finds the *last* "References" heading (papers mention the word in running
    text too) and takes everything from there on. Falls back to the pages that
    look densest in numbered entries.
    """
    start = None
    for i, text in enumerate(pages):
        if _REF_HEADING_RE.search(text or ""):
            start = i
    if start is not None:
        return list(range(start, len(pages)))

    # No heading survived extraction (two-column PDFs often lose it): use the
    # pages whose numbered-entry density says "this is a bibliography".
    scored = [(i, len(_REF_ENTRY_RE.findall(t or ""))) for i, t in enumerate(pages)]
    dense = [i for i, n in scored if n >= 5]
    if not dense:
        return []
    return list(range(min(dense), len(pages)))


def _batch_pages(pages: List[str], page_numbers: List[int]) -> List[tuple]:
    """Group reference text into (text, first_page) batches under the char cap."""
    batches, buf, buf_start, size = [], [], None, 0
    for idx in page_numbers:
        text = " ".join((pages[idx] or "").split())
        if not text:
            continue
        if buf and size + len(text) > settings.graph_batch_chars:
            batches.append(("\n".join(buf), buf_start))
            buf, size, buf_start = [], 0, None
        if buf_start is None:
            buf_start = idx + 1          # 1-indexed page
        buf.append(text)
        size += len(text)
    if buf:
        batches.append(("\n".join(buf), buf_start))
    return batches


def extract_references(pages: List[str], filename: str,
                       progress: Optional[Callable[[str], None]] = None) -> List[CitedWork]:
    """Extract every reference-list entry from one paper."""
    ref_pages = find_reference_pages(pages)
    if not ref_pages:
        return []
    batches = _batch_pages(pages, ref_pages)
    if not batches:
        return []

    works: List[CitedWork] = []
    for n, (text, first_page) in enumerate(batches, start=1):
        if progress:
            progress(f"    references batch {n}/{len(batches)} (p.{first_page})")
        batch_works = (
            _llm_references(text, filename, first_page)
            if settings.anthropic_api_key
            else _heuristic_references(text, first_page)
        )
        works.extend(batch_works)
    return works


def _llm_references(text: str, filename: str, first_page: int) -> List[CitedWork]:
    try:
        raw = _complete(
            system=(
                "You parse bibliographies into structured records. Return ONLY "
                "a JSON array. Copy names exactly as printed — do not expand "
                "initials, translate, or correct spelling."
            ),
            user=(
                f"Reference-list text from {filename} (starting p.{first_page}). "
                "PDF extraction may have interleaved two columns or split entries "
                "mid-line; reconstruct entries as best you can.\n\n"
                f"{text}\n\n"
                "Return a JSON array, one object per reference:\n"
                "{\"authors\": [str], \"title\": str, \"venue\": str, "
                "\"year\": int|null}\n\n"
                "Rules:\n"
                "- One entry per numbered reference. Skip anything that is not a "
                "reference (page headers, running titles, footers).\n"
                "- `authors` is one string per person, as printed "
                "(\"Berke JD\", \"A.I. Levey\", \"WISE, R. A.\"). Drop 'et al.'.\n"
                "- Omit an entry entirely rather than guessing its authors."
            ),
        )
        parsed = _json_slice(raw)
    except Exception:
        return _heuristic_references(text, first_page)

    out: List[CitedWork] = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            work = CitedWork(**item)
        except Exception:
            continue
        work.page = first_page
        if work.authors:
            out.append(work)
    return out


def _heuristic_references(text: str, first_page: int) -> List[CitedWork]:
    """Offline fallback: split on numbered markers, take the leading author run.

    Deliberately conservative — it recovers authors (which is what the network
    needs) and leaves title/venue empty rather than guessing at them.
    """
    from .canon import split_author_list

    pieces = _REF_ENTRY_RE.split(text)
    entries = [p for p in pieces[1:] if len(p) > 20] if len(pieces) > 1 else []
    if not entries:
        entries = [p for p in re.split(r"(?<=\.)\s{2,}", text) if len(p) > 40]

    out: List[CitedWork] = []
    for entry in entries:
        # Authors run until the first sentence-ending period followed by a
        # capitalised word that starts the title, or a year in parentheses.
        head = re.split(r"\(\d{4}\)|\.\s+[A-Z]", entry, maxsplit=1)[0]
        head = head[:300]
        names = [n for n in split_author_list(head) if len(n) > 2]
        if not names or len(names) > 30:
            continue
        year = None
        match = re.search(r"\b(19|20)\d{2}\b", entry[:400])
        if match:
            year = int(match.group(0))
        out.append(CitedWork(authors=names[:20], title="", venue="", year=year,
                             page=first_page))
    return out


# --------------------------------------------------------------------------- #
# Page loading (shared with the RAG layer's extractor)
# --------------------------------------------------------------------------- #
def load_pages(pdf_path: Path) -> List[str]:
    """Page texts for a PDF, index i == page i+1. Never raises per page."""
    from ..rag import _extract_pages

    return _extract_pages(pdf_path)
