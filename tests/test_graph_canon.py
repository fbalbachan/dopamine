"""Entity resolution — the deterministic core of the graph build.

Every case here is a real string taken from one of the nine PDFs. The corpus
uses nine different citation styles, and getting any of these wrong silently
splits one researcher into two nodes (or merges two into one), which corrupts
every co-authorship edge and every recommendation downstream. These are the
regression tests for that.
"""
from __future__ import annotations

import pytest

from crew.graph.canon import (
    best_display,
    canonical_institution,
    parse_name,
    split_author_list,
)


@pytest.mark.parametrize(
    "surface",
    [
        "Wolfram Schultz",     # front matter, full name
        "Schultz W",           # Vancouver style
        "Schultz, W.",         # comma style
        "W. Schultz",          # leading initial
        "SCHULTZ, W. R.",      # all-caps, extra middle initial
    ],
)
def test_all_surface_forms_of_one_person_share_a_key(surface):
    """The whole graph depends on this: one person, one node id."""
    assert parse_name(surface).key == "schultz|w"


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("Berke JD", "berke|j"),
        ("Josh Berke", "berke|j"),
        ("A.I. Levey", "levey|a"),
        ("CRISTINA MISSALE", "missale|c"),
        ("MARC G. CARON", "caron|m"),
        ("Charles A. Marsden", "marsden|c"),
        ("Tannock R", "tannock|r"),
        ("McGee R", "mcgee|r"),
        ("O'Brien P", "o-brien|p"),
        ("van der Meer MA", "van-der-meer|m"),
    ],
)
def test_key_extraction_across_styles(surface, expected):
    assert parse_name(surface).key == expected


def test_leading_initial_wins_over_expanded_middle_name():
    """'S. Russel Nash' keys on S, not R — the first given name is the initial."""
    assert parse_name("S. RUSSEL NASH").key == "nash|s"


def test_affiliation_markers_are_stripped():
    """Superscript affiliation markers survive PDF extraction as digits."""
    assert parse_name("Zhi Dong Zhou1,3*").key == "zhou|z"


def test_non_names_are_rejected():
    assert parse_name("et al.") is None
    assert parse_name("") is None
    assert parse_name("   ") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "CRISTINA MISSALE, S. RUSSEL NASH, SUSAN W. ROBINSON, "
            "MOHAMED JABER, AND MARC G. CARON",
            ["missale|c", "nash|s", "robinson|s", "jaber|m", "caron|m"],
        ),
        (
            "Zhi Dong Zhou1,3*, Ling Xiao Yi1, Dennis Qing Wang4, "
            "Tit Meng Lim5 and Eng King Tan1,2,3*",
            ["zhou|z", "yi|l", "wang|d", "lim|t", "tan|e"],
        ),
        ("Loh, E. A. & Roberts, D. C. S.", ["loh|e", "roberts|d"]),
        ("WISE, R. A., AND P. P. ROMPRE", ["wise|r", "rompre|p"]),
        (
            "Latif S, Jahangeer M, Ashiq M, Ghaffar A, et al.",
            ["latif|s", "jahangeer|m", "ashiq|m", "ghaffar|a"],
        ),
        (
            "A.I. Levey, S.M. Hersch, D.B. Rye",
            ["levey|a", "hersch|s", "rye|d"],
        ),
    ],
)
def test_author_lists_split_correctly(raw, expected):
    """Commas separate authors in one style and sit inside names in another."""
    keys = [parse_name(n).key for n in split_author_list(raw) if parse_name(n)]
    assert keys == expected


def test_et_al_is_dropped_not_treated_as_an_author():
    assert all("al" != parse_name(n).surname.lower()
               for n in split_author_list("Wu J, Xiao H, et al.") if parse_name(n))


def test_best_display_prefers_the_most_complete_form():
    assert best_display(["Schultz W", "Wolfram Schultz", "SCHULTZ, W."]) == "Wolfram Schultz"


def test_best_display_falls_back_to_initials_when_that_is_all_there_is():
    assert best_display(["Tannock R"]) == "Tannock R"


def test_institution_ids_ignore_department_prefixes_and_case():
    assert canonical_institution("Department of Physiology") == "physiology"
    assert canonical_institution("University of Cambridge") == \
        canonical_institution("  the university of cambridge  ")


def test_distinct_institutions_do_not_collapse():
    assert canonical_institution("Duke University Medical Center") != \
        canonical_institution("University of Cambridge")
