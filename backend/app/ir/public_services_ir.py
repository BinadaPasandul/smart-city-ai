"""Keyword-based Information Retrieval over the Public Services seed dataset.

This module loads `backend/data/seed/public_services_data.json` and exposes a
single typed `PublicServicesIR.search()` entry point that ranks records with a
deterministic, explainable keyword-overlap score (see `PublicServicesIR`).

No embeddings, vector database, NLP, or LLM calls are used here. NLP-based
query understanding (entity/location extraction, etc.) belongs in
`app/nlp/public_services_nlp.py`, and calling this module from the specialist
agent belongs in `app/agents/public_services/agent.py`; neither is wired up
by this module.
"""

import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "public_services_data.json"


class PublicServiceCategory(str, Enum):
    """The six record categories present in the Public Services seed dataset."""

    HOSPITALS = "hospitals"
    POLICE_STATIONS = "police_stations"
    FIRE_STATIONS = "fire_stations"
    GOVERNMENT_SERVICES = "government_services"
    EMERGENCY_INFORMATION = "emergency_information"
    CITIZEN_COMPLAINTS = "citizen_complaints"


class InvalidPublicServiceCategoryError(ValueError):
    """Raised when a caller supplies a category string that is not recognized."""


# Which raw JSON fields feed the search index for each category.
# `title_field` is weighted more heavily than `text_fields`/`list_fields` (see
# PublicServicesIR._score_record). Fields not listed here (e.g. `contact`,
# `operating_hours`, `id`) are preserved on the record but are not searchable,
# since they are not meaningful free-text query targets.
_CATEGORY_FIELDS: dict[PublicServiceCategory, dict[str, Any]] = {
    PublicServiceCategory.HOSPITALS: {
        "title_field": "name",
        "text_fields": ["type", "location", "district", "address"],
        "list_fields": ["services"],
    },
    PublicServiceCategory.POLICE_STATIONS: {
        "title_field": "name",
        "text_fields": ["location", "district", "address"],
        "list_fields": ["services"],
    },
    PublicServiceCategory.FIRE_STATIONS: {
        "title_field": "name",
        "text_fields": ["location", "district", "address"],
        "list_fields": ["emergency_services"],
    },
    PublicServiceCategory.GOVERNMENT_SERVICES: {
        "title_field": "name",
        "text_fields": ["department", "location", "district"],
        "list_fields": ["services"],
    },
    PublicServiceCategory.EMERGENCY_INFORMATION: {
        "title_field": "service",
        "text_fields": ["emergency_type", "description"],
        "list_fields": [],
    },
    PublicServiceCategory.CITIZEN_COMPLAINTS: {
        "title_field": "complaint_type",
        "text_fields": ["responsible_department", "description"],
        "list_fields": ["submission_method"],
    },
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split into whitespace-separated tokens.

    Mirrors the normalization used by `DeterministicQueryRouter` in
    `app/agents/orchestrator/router.py` so query handling stays consistent
    across the project.
    """
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()
    return normalized.split() if normalized else []


class PublicServiceRecord(BaseModel):
    """One seed-data record with its original fields preserved for grounding."""

    category: PublicServiceCategory
    record_id: str
    title: str
    fields: dict[str, Any] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    """Provenance for a search result, for building a grounded AgentSource later."""

    source_file: str
    source_type: str
    description: str | None = None


class SearchMatch(BaseModel):
    """One ranked search result."""

    record: PublicServiceRecord
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    source: SourceInfo


class _IndexedRecord(BaseModel):
    """Internal: a record plus its precomputed, weighted token sets."""

    model_config = {"arbitrary_types_allowed": True}

    record: PublicServiceRecord
    title_tokens: frozenset[str]
    body_tokens: frozenset[str]


class PublicServicesIR:
    """Typed, keyword-based retrieval over the Public Services seed dataset.

    Relevance scoring (deterministic, keyword-overlap based):
        For each record, tokens are split into `title_tokens` (from the
        record's primary name/title field only) and `body_tokens` (from the
        other configured searchable fields, e.g. location, district,
        address, services, description).

        For every distinct query token that appears in `title_tokens`, 2
        points are added; for every distinct query token that appears only
        in `body_tokens`, 1 point is added. The raw point total is divided
        by `2 * len(query_tokens)`, so the maximum possible score is 1.0
        (every query token matches the title). Records that match zero
        query tokens are excluded from the results entirely.

        This is a simple weighted term-overlap baseline, not TF-IDF or
        BM25 -- it does not account for term frequency across the corpus.
        It is intentionally kept swappable: `search()` is the only public
        entry point, so the scoring method can later be replaced with BM25
        or hybrid semantic retrieval without changing how callers use this
        module.
    """

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        raw = self._load_raw_data()
        self._source = SourceInfo(
            source_file=self.data_path.name,
            source_type=raw.get("metadata", {}).get("source_type", "unknown"),
            description=raw.get("metadata", {}).get("description"),
        )
        self._index: dict[PublicServiceCategory, list[_IndexedRecord]] = self._build_index(raw)

    def _load_raw_data(self) -> dict[str, Any]:
        """Read and parse the seed JSON file, tolerating a missing/invalid file."""
        try:
            with self.data_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            logger.error("Public services data file not found at %s", self.data_path)
            return {}
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse public services data JSON at %s: %s", self.data_path, exc)
            return {}

    def _build_index(self, raw: dict[str, Any]) -> dict[PublicServiceCategory, list[_IndexedRecord]]:
        """Build the per-category token index for every supported category."""
        index: dict[PublicServiceCategory, list[_IndexedRecord]] = {}
        for category, field_map in _CATEGORY_FIELDS.items():
            raw_records = raw.get(category.value, [])
            if not isinstance(raw_records, list):
                logger.warning("Expected a list for category '%s'; got %s. Skipping.", category.value, type(raw_records))
                raw_records = []
            index[category] = [self._index_record(category, field_map, item) for item in raw_records]
        return index

    @staticmethod
    def _index_record(
        category: PublicServiceCategory, field_map: dict[str, Any], item: dict[str, Any]
    ) -> _IndexedRecord:
        title_field = field_map["title_field"]
        title_value = str(item.get(title_field, ""))

        body_parts: list[str] = []
        for field_name in field_map["text_fields"]:
            value = item.get(field_name)
            if isinstance(value, str):
                body_parts.append(value)
        for field_name in field_map["list_fields"]:
            values = item.get(field_name)
            if isinstance(values, list):
                body_parts.extend(str(v) for v in values)

        record = PublicServiceRecord(
            category=category,
            record_id=str(item.get("id", "")),
            title=title_value,
            fields=item,
        )
        return _IndexedRecord(
            record=record,
            title_tokens=frozenset(_tokenize(title_value)),
            body_tokens=frozenset(_tokenize(" ".join(body_parts))),
        )

    def available_categories(self) -> list[PublicServiceCategory]:
        """Return all categories this IR module recognizes."""
        return list(_CATEGORY_FIELDS.keys())

    def records(self, category: PublicServiceCategory) -> list[PublicServiceRecord]:
        """Return every loaded record for one category, unranked."""
        return [indexed.record for indexed in self._index.get(category, [])]

    def search(
        self,
        query: str,
        category: PublicServiceCategory | str | None = None,
        top_k: int = 10,
    ) -> list[SearchMatch]:
        """Return records matching `query`, ranked by descending relevance score.

        Args:
            query: Free-text search query. A blank/whitespace-only query
                always returns an empty list; it is not treated as "match
                everything".
            category: Restrict the search to one category. Accepts a
                `PublicServiceCategory` or its string value (e.g.
                "hospitals"). `None` searches across all categories.
            top_k: Maximum number of results to return. Must be positive.

        Returns:
            Ranked `SearchMatch` results with `score > 0`. An empty list
            means nothing matched -- callers must not invent a fallback
            answer for that case.

        Raises:
            InvalidPublicServiceCategoryError: `category` is a string that
                does not match a known category.
            ValueError: `top_k` is not positive.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = frozenset(_tokenize(query))
        if not query_tokens:
            return []

        categories = self._resolve_categories(category)

        matches: list[SearchMatch] = []
        for cat in categories:
            for indexed in self._index.get(cat, []):
                match = self._score_record(indexed, query_tokens)
                if match is not None:
                    matches.append(match)

        matches.sort(key=lambda m: (-m.score, m.record.category.value, m.record.record_id))
        return matches[:top_k]

    def _score_record(self, indexed: _IndexedRecord, query_tokens: frozenset[str]) -> SearchMatch | None:
        """Score one record against `query_tokens`; return None when nothing matches."""
        title_hits = query_tokens & indexed.title_tokens
        body_hits = (query_tokens & indexed.body_tokens) - title_hits
        matched_terms = title_hits | body_hits
        if not matched_terms:
            return None

        raw_score = 2 * len(title_hits) + len(body_hits)
        score = raw_score / (2 * len(query_tokens))
        return SearchMatch(
            record=indexed.record,
            score=min(score, 1.0),
            matched_terms=sorted(matched_terms),
            source=self._source,
        )

    def _resolve_categories(
        self, category: PublicServiceCategory | str | None
    ) -> list[PublicServiceCategory]:
        if category is None:
            return self.available_categories()
        if isinstance(category, PublicServiceCategory):
            return [category]
        try:
            return [PublicServiceCategory(category)]
        except ValueError as exc:
            raise InvalidPublicServiceCategoryError(
                f"Unknown public services category: '{category}'"
            ) from exc
