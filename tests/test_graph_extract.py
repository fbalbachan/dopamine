"""Extraction robustness.

The extraction models are fed raw LLM JSON. When validation rejects a record the
caller returns an empty result and the build continues — so a schema that is too
strict does not raise, it silently deletes a paper's entire author team. These
tests pin the leniency that prevents that.
"""
from __future__ import annotations

from crew.graph.extract import (
    CitedWork,
    ExtractedAuthor,
    PaperMeta,
    _json_slice,
    find_reference_pages,
)


def test_null_doi_does_not_reject_the_whole_paper():
    """The regression: the 1998 Missale review has no DOI on page 1, the model
    correctly returned `"doi": null`, and a strict `str` field threw — costing
    the paper all five of its Duke authors."""
    meta = PaperMeta(
        title="Dopamine Receptors: From Structure to Function",
        year=1998,
        venue="Physiological Reviews",
        doi=None,
        abstract=None,
        authors=[{"name": "Marc G. Caron", "organization": "Duke University"}],
    )
    assert meta.doi == ""
    assert meta.abstract == ""
    assert meta.title.startswith("Dopamine Receptors")
    assert len(meta.authors) == 1


def test_null_author_fields_are_blanked_not_rejected():
    author = ExtractedAuthor(name="Wolfram Schultz", organization=None,
                             department=None, country=None, email=None)
    assert author.organization == ""
    assert author.email == ""
    assert author.name == "Wolfram Schultz"


def test_null_cited_work_fields_are_blanked():
    work = CitedWork(authors=["Berke JD"], title=None, venue=None, year=None)
    assert work.title == ""
    assert work.venue == ""
    assert work.year is None


def test_year_may_legitimately_be_null():
    """`year` is Optional by design — leniency must not turn it into a string."""
    assert PaperMeta(title="x", year=None).year is None


def test_missing_optional_fields_use_defaults():
    meta = PaperMeta(title="Only a title")
    assert meta.authors == []
    assert meta.venue == ""


def test_json_slice_tolerates_markdown_fences():
    """Models routinely wrap JSON in ```json fences."""
    raw = 'Here you go:\n```json\n{"title": "x"}\n```\n'
    assert _json_slice(raw, "{", "}") == {"title": "x"}


def test_json_slice_raises_when_there_is_no_json():
    try:
        _json_slice("no json here", "{", "}")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_reference_pages_found_from_a_heading():
    pages = ["intro", "methods", "References\n1. Someone A. A title.", "2. More"]
    assert find_reference_pages(pages) == [2, 3]


def test_reference_pages_use_the_last_heading():
    """Papers mention the word 'References' in running text as well."""
    pages = ["see References below", "body", "References\n1. A. Title."]
    assert find_reference_pages(pages) == [2]


def test_reference_pages_fall_back_to_entry_density():
    """Two-column PDFs often lose the heading entirely."""
    dense = "\n".join(f"{i}. Author{i} A. Some title here." for i in range(1, 9))
    assert find_reference_pages(["body", "body", dense]) == [2]


def test_no_reference_section_returns_empty():
    assert find_reference_pages(["just", "body", "text"]) == []
