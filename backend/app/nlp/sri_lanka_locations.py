"""Small deterministic Sri Lankan place-name gazetteer.

Supplements (never replaces) spaCy NER in `SpacyEntityExtractor`. Observed
directly: `en_core_web_sm` mislabels well-known Sri Lankan city names as
PERSON rather than GPE (confirmed for "Kandy"). This module provides a
narrow, whole-word, case-insensitive fallback for exactly that situation,
plus the small country-name check used to compose an adjacent
"<location>, <country>" phrase back into one location string.

Kept intentionally small and specific to this project's domain (compare
`app/nlp/mobility_nlp.py`'s `KNOWN_LOCATIONS`, the existing precedent for
this pattern) rather than a general-purpose gazetteer.
"""

import re

KNOWN_LOCATIONS: tuple[str, ...] = (
    "Kandy",
    "Colombo",
    "Galle",
    "Jaffna",
    "Negombo",
    "Matara",
    "Kurunegala",
    "Anuradhapura",
    "Trincomalee",
    "Batticaloa",
    "Nuwara Eliya",
    "Ratnapura",
)

# Country names recognized as a valid "<location>, <country>" composition
# qualifier. Scoped to what's relevant to this project's domain -- extend
# deliberately, not speculatively.
KNOWN_COUNTRIES: tuple[str, ...] = ("Sri Lanka",)

# Longest-name-first so a multi-word entry (e.g. "Nuwara Eliya") can never be
# shadowed by a shorter one sharing a prefix.
_LOCATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in sorted(KNOWN_LOCATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_KNOWN_COUNTRIES_CASEFOLDED = frozenset(name.casefold() for name in KNOWN_COUNTRIES)


def find_known_locations(text: str) -> list[tuple[int, int, str]]:
    """Return (start_char, end_char, original_text) for each gazetteer match.

    Matching is whole-word (via `\\b`) to avoid substring false positives
    (e.g. matching "Kandy" inside a longer unrelated word), case-insensitive,
    and preserves the original spelling/casing from the query via
    `match.group(0)` rather than the canonical gazetteer entry.
    """
    return [
        (match.start(), match.end(), match.group(0))
        for match in _LOCATION_PATTERN.finditer(text)
    ]


def is_known_country(text: str) -> bool:
    """Whether `text` names a country this project recognizes for composition."""
    return text.strip().casefold() in _KNOWN_COUNTRIES_CASEFOLDED
