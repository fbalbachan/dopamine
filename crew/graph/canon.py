"""Entity resolution: turning messy surface forms into stable node ids.

Nine papers, nine citation styles. The same person appears as ``Wolfram
Schultz``, ``Schultz W``, ``SCHULTZ, W.``, ``W. Schultz`` and ``Schultz, W. R.``
depending on which reference list you read. Author-name disambiguation is a
genuinely unsolved problem in bibliometrics; without ORCIDs the field's standard
heuristic is **surname + first initial**, and that is what is implemented here.

The split of labour matters: the LLM extractor (:mod:`crew.graph.extract`) reads
prose and hands back *surface forms*; this module decides, deterministically,
which surface forms denote the same person. Keeping resolution out of the model
is what makes it unit-testable and stable across rebuilds — re-running the build
cannot silently re-partition the graph.

Known limitation: surname + first initial merges genuinely distinct researchers
who share both (two ``Wang D.`` are one node). ``crew/graph/README.md`` records
this; ORCID resolution is the fix if it ever matters.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

# Surname particles that belong *with* the surname: "van der Meer" -> "van der meer".
_PARTICLES = {
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "du",
    "la", "le", "el", "al", "bin", "ibn", "ter", "ten", "st", "san", "santa",
}

# Tokens that are never part of a name.
_NOISE = {
    "et", "al", "jr", "sr", "ii", "iii", "iv", "phd", "md", "dr", "prof",
    "professor", "and", "the",
}

# Affiliation/footnote markers: digits, stars, daggers, section/ORCID glyphs.
# Hyphens are deliberately NOT in this class — they belong to hyphenated names.
_MARKER_RE = re.compile(r"[\d\*†‡§¶®©]+")
# Keep commas: in "Wise, R. A." the comma is what tells us which part is the
# surname, so it must survive cleaning.
_PUNCT_RE = re.compile(r"[^\w\s',\-\.]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Name:
    """A parsed personal name."""
    surname: str        # display form, e.g. "van der Meer"
    given: str          # full given name(s) when known, else ""
    initials: str       # e.g. "MA"; always at least the first initial when known

    @property
    def key(self) -> str:
        """Canonical node id: ``surname-slug|first-initial``.

        The separator is ``|`` because it cannot occur in a slug, so ids stay
        unambiguous to split and safe to use as JSON keys.
        """
        return f"{_slug(self.surname)}|{(self.initials[:1] or '').lower()}"

    @property
    def display(self) -> str:
        """Best human-readable form available."""
        if self.given:
            return f"{self.given} {self.surname}".strip()
        if self.initials:
            return f"{self.surname} {self.initials}".strip()
        return self.surname

    @property
    def is_complete(self) -> bool:
        """True when we have a real given name, not just initials."""
        return bool(self.given)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _slug(text: str) -> str:
    text = _strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def slug(text: str) -> str:
    """Public slugifier — node ids elsewhere in the package are built from it."""
    return _slug(text)


def _looks_like_initials(token: str) -> bool:
    """True for 'W', 'JD', 'W.', 'A.I.', 'R.A.' — but not for 'Wolfram'."""
    bare = token.replace(".", "")
    if not bare or not bare.isalpha():
        return False
    if "." in token:
        return len(bare) <= 4
    # No dots: only treat a short ALL-CAPS run as initials ("JD", "W").
    return bare.isupper() and len(bare) <= 3


def _titlecase(token: str) -> str:
    """Fix ALL-CAPS reference styles ('MISSALE' -> 'Missale') without wrecking
    names that are legitimately mixed-case ('McGee', "O'Brien")."""
    if token.isupper() or token.islower():
        parts = re.split(r"([\-'])", token.lower())
        return "".join(p if p in "-'" else p.capitalize() for p in parts)
    return token


def _clean(raw: str) -> str:
    raw = raw.replace(" ", " ")
    raw = _MARKER_RE.sub(" ", raw)
    raw = _PUNCT_RE.sub(" ", raw.replace(",", " , "))
    return _WS_RE.sub(" ", raw).strip()


def parse_name(raw: str) -> Optional[Name]:
    """Parse one personal name in any of the corpus's citation styles.

    Returns ``None`` for input that isn't a name (empty, pure noise, a lone
    initial with no surname).

    >>> parse_name("Schultz W").key
    'schultz|w'
    >>> parse_name("Wolfram Schultz").key
    'schultz|w'
    >>> parse_name("SCHULTZ, W. R.").key
    'schultz|w'
    """
    if not raw:
        return None

    has_comma = "," in raw
    text = _clean(raw)
    if not text:
        return None

    tokens = [t for t in text.split() if t != ","]
    tokens = [t for t in tokens if t.replace(".", "").lower() not in _NOISE]
    if not tokens:
        return None

    surname_tokens: List[str] = []
    rest_tokens: List[str] = []
    if has_comma:
        # "Surname, Initials" / "Surname, Given" — the comma is authoritative.
        left, _, right = text.partition(",")
        surname_tokens = [t for t in left.split()
                          if t.replace(".", "").lower() not in _NOISE]
        rest_tokens = [t for t in right.split()
                       if t.replace(".", "").lower() not in _NOISE]
    # A comma that came from a stripped affiliation marker ("Zhou1,3*") leaves
    # nothing on one side; that is not a "Surname, Initials" split.
    if not surname_tokens or not rest_tokens:
        surname_tokens, rest_tokens = _split_no_comma(tokens)

    if not surname_tokens:
        return None

    surname = " ".join(
        t if t.lower() in _PARTICLES else _titlecase(t) for t in surname_tokens
    )
    surname = surname.replace(".", "").strip()
    if not surname or not any(c.isalpha() for c in surname):
        return None

    given_parts = [t for t in rest_tokens if not _looks_like_initials(t)]
    given = " ".join(_titlecase(t.replace(".", "")) for t in given_parts).strip()

    # Initials follow token order, so "S. Russel Nash" keys on S (not R) — the
    # leading initial is the first given name, whatever the later tokens expand.
    initials = "".join(
        t.replace(".", "")[:1].upper() for t in rest_tokens if t.replace(".", "")
    )

    return Name(surname=surname, given=given, initials=initials)


def _split_no_comma(tokens: List[str]) -> tuple[List[str], List[str]]:
    """Split ``["Schultz", "W"]`` / ``["Wolfram", "Schultz"]`` into (surname, rest)."""
    # Trailing initials: "Schultz W", "Berke JD", "Zhou ZD".
    trailing = 0
    for tok in reversed(tokens):
        if _looks_like_initials(tok):
            trailing += 1
        else:
            break
    if trailing and trailing < len(tokens):
        return tokens[:-trailing], tokens[-trailing:]

    # Leading initials: "W. Schultz", "A.I. Levey", "S. RUSSEL NASH".
    leading = 0
    for tok in tokens:
        if _looks_like_initials(tok):
            leading += 1
        else:
            break
    if leading and leading < len(tokens):
        # What follows may still be "Given Surname" (S. RUSSEL NASH), so apply
        # the full-name rule to the remainder rather than swallowing it whole.
        surname_tokens, extra_given = _surname_tail(tokens[leading:])
        return surname_tokens, tokens[:leading] + extra_given

    return _surname_tail(tokens)


def _surname_tail(tokens: List[str]) -> tuple[List[str], List[str]]:
    """Full-name rule: the surname is the last token plus any leading particles."""
    if len(tokens) <= 1:
        return tokens, []
    start = len(tokens) - 1
    while start > 0 and tokens[start - 1].lower() in _PARTICLES:
        start -= 1
    return tokens[start:], tokens[:start]


def split_author_list(raw: str) -> List[str]:
    """Split a raw author *string* into individual name strings.

    Handles the separators the corpus actually uses: ``and``, ``&``, ``;``, and
    commas. Commas are ambiguous — they separate authors in ``Wu J, Xiao H`` but
    sit *inside* a name in ``Wise, R. A., and Rompre, P.`` — so a list is only
    split on commas when it does not look like ``Surname, Initials`` pairs.
    """
    if not raw:
        return []
    # Strip affiliation markers first: "Zhou1,3*" otherwise splits into a bogus
    # second author "3*" on the marker's own comma.
    text = _MARKER_RE.sub("", raw)
    text = re.sub(r"\bet\s+al\.?", " ", text, flags=re.I)
    text = re.sub(r"\s+(and|&)\s+", ";", text, flags=re.I)
    text = text.replace("&", ";")

    if ";" in text:
        parts = [p for p in text.split(";") if p.strip()]
        # Each ';'-part may itself hold comma-separated authors.
        out: List[str] = []
        for part in parts:
            out.extend(_split_commas(part))
        return [p.strip() for p in out if p.strip()]
    return [p.strip() for p in _split_commas(text) if p.strip()]


def _split_commas(text: str) -> List[str]:
    if "," not in text:
        return [text]
    pieces = [p.strip() for p in text.split(",") if p.strip()]
    # "Wise, R. A." -> the odd pieces are initials, so re-pair them.
    if len(pieces) >= 2 and all(
        _looks_like_initials(t) for t in pieces[1].split()
    ):
        paired, i = [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and all(
                _looks_like_initials(t) for t in pieces[i + 1].split()
            ):
                paired.append(f"{pieces[i]}, {pieces[i + 1]}")
                i += 2
            else:
                paired.append(pieces[i])
                i += 1
        return paired
    return pieces


# --- Institutions ---------------------------------------------------------- #

# Ordered longest-first so "University of California, San Francisco" is not
# shortened before the more specific match is tried.
_ORG_NOISE_RE = re.compile(
    r"^(the|departments? of|dept\.? of|division of|school of|institute of|"
    r"faculty of|centre for|center for|laboratory of|labs? of)\s+",
    re.I,
)


def canonical_institution(name: str) -> str:
    """Stable id for an organisation name.

    Only light normalisation is applied — casing, punctuation, articles, and a
    leading ``Department of`` — because aggressive rewriting merges genuinely
    different organisations. The extractor is asked to return the *top-level*
    organisation, which does most of this work upstream.
    """
    text = " ".join((name or "").split())
    if not text:
        return ""
    prev = None
    while prev != text:
        prev = text
        text = _ORG_NOISE_RE.sub("", text).strip()
    return _slug(text)


def canonical_topic(label: str) -> str:
    """Stable id for a research topic label."""
    return _slug(label)


def best_display(candidates: List[str]) -> str:
    """Pick the most informative surface form for a person.

    Prefers a real given name over initials, then the longest form — so
    ``Wolfram Schultz`` wins over ``Schultz W``.
    """
    parsed = [(c, parse_name(c)) for c in candidates if c]
    named = [(c, n) for c, n in parsed if n and n.is_complete]
    pool = named or [(c, n) for c, n in parsed if n]
    if not pool:
        return candidates[0] if candidates else ""
    return max(pool, key=lambda cn: (len(cn[1].given), len(cn[1].display)))[1].display
