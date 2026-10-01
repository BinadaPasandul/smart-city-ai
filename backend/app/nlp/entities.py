"""Lazy spaCy NER adapter for location-like entities."""

from threading import Lock
from typing import Protocol


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
        seen: set[str] = set()
        locations: list[str] = []
        for entity in doc.ents:
            if entity.label_ not in self.LOCATION_LABELS:
                continue
            value = entity.text.strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                locations.append(value)
        return locations
