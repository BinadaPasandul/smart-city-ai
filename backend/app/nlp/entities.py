"""Lazy spaCy NER adapter for location-like entities."""

import re
from threading import Lock
from typing import Protocol

from app.nlp.sri_lanka_locations import find_known_locations, is_known_country

# Matches a bare comma plus optional whitespace and nothing else -- the only
# gap text treated as "these two entities form one location phrase". This
# deliberately does not match "and", "near", or any other connector, so
# genuinely separate locations (e.g. "Colombo and Kandy") are never merged.
_COMPOSITION_GAP = re.compile(r",\s*")


class EntityModelUnavailable(RuntimeError):
    """Raised when spaCy or its configured model is not installed locally."""


class EntityExtractor(Protocol):
    def extract_locations(self, text: str) -> list[str]:
        """Return recognized location phrases in textual order."""


class SpacyEntityExtractor:
    """Load the configured English pipeline on first use; never download at runtime."""

    LOCATION_LABELS = frozenset({"GPE", "LOC", "FAC"})

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._model_name = model_name
        self._nlp = None
        self._load_error: Exception | None = None
        self._lock = Lock()

    def _load(self):
        if self._nlp is not None:
            return self._nlp
        if self._load_error is not None:
            raise EntityModelUnavailable("configured spaCy model is unavailable") from None
        with self._lock:
            if self._nlp is not None:
                return self._nlp
            if self._load_error is not None:
                raise EntityModelUnavailable("configured spaCy model is unavailable") from None
            try:
                import spacy

                self._nlp = spacy.load(
                    self._model_name,
                    exclude=["parser", "tagger", "lemmatizer", "attribute_ruler"],
                )
            except (ImportError, OSError, ValueError) as exc:
                self._load_error = exc
                raise EntityModelUnavailable(
                    "configured spaCy model is unavailable"
                ) from None
        return self._nlp

    def extract_locations(self, text: str) -> list[str]:
        nlp = self._load()
        # Keep calls through the shared Language object serialized for worker safety.
        with self._lock:
            doc = nlp(text)

        spans: list[tuple[int, int, str]] = [
            (entity.start_char, entity.end_char, entity.text)
            for entity in doc.ents
            if entity.label_ in self.LOCATION_LABELS
        ]

        # Supplement with the deterministic Sri Lankan gazetteer for known
        # place names spaCy's small model sometimes mislabels (observed:
        # "Kandy" as PERSON) -- only where spaCy didn't already claim that
        # span as a location, so a correct spaCy match is never duplicated.
        for start, end, value in find_known_locations(text):
            if not any(s_start < end and start < s_end for s_start, s_end, _ in spans):
                spans.append((start, end, value))

        spans.sort(key=lambda span: span[0])
        composed = self._compose_adjacent(spans, text)

        seen: set[str] = set()
        locations: list[str] = []
        for value in composed:
            value = value.strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                locations.append(value)
        return locations

    @staticmethod
    def _compose_adjacent(spans: list[tuple[int, int, str]], text: str) -> list[str]:
        """Join a '<location>, <country>' pair into one phrase; leave everything else alone.

        Only composes when the text between two location spans is exactly a
        comma (plus optional whitespace) AND the second span names a known
        country -- e.g. "Colombo, Sri Lanka" becomes one phrase, but
        "Colombo, Galle" or "Colombo and Kandy" never do, since neither
        "Galle" nor "Kandy" is a recognized country. This intentionally does
        not touch the >1-location clarification gate in the understanding
        pipeline; it only improves what `extract_locations` hands to it.
        """
        if not spans:
            return []
        merged: list[tuple[int, int, str]] = [spans[0]]
        for start, end, value in spans[1:]:
            prev_start, prev_end, _ = merged[-1]
            gap = text[prev_end:start]
            if _COMPOSITION_GAP.fullmatch(gap) and is_known_country(value):
                merged[-1] = (prev_start, end, text[prev_start:end])
            else:
                merged.append((start, end, value))
        return [value for _, _, value in merged]
