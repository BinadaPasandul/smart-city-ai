"""Deterministic, rule-based NLP baseline for Public Services queries.

Converts a raw natural-language query into a structured `PublicServicesQuery`
that a future integration step can pass to `PublicServicesIR.search()`. No
LLM, embeddings, or network calls are used here -- every detection rule is a
documented keyword/phrase match over the normalized query text, so behavior
is fully deterministic and locally testable.

This module does not call `PublicServicesIR` or `PublicServicesAgent`; it
only reads the seed JSON file once, at construction time, to build its
location/district vocabulary from real data instead of a hand-maintained
list that could drift out of sync.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.ir.public_services_ir import PublicServiceCategory

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "public_services_data.json"

# Seed-data categories that carry "location"/"district" fields. Emergency
# information and citizen complaint records are not location-bound (see
# backend/data/seed/public_services_data.json), so they are excluded here.
_LOCATION_BEARING_CATEGORIES = (
    "hospitals",
    "police_stations",
    "fire_stations",
    "government_services",
)

# Rule-based category keyword phrases. Matching is by whole-phrase substring
# on the normalized query (see `_normalize`), so multi-word phrases like
# "police station" are matched literally, not as separate tokens.
_CATEGORY_KEYWORDS: dict[PublicServiceCategory, tuple[str, ...]] = {
    PublicServiceCategory.HOSPITALS: (
        "hospital", "hospitals", "medical", "doctor", "clinic",
    ),
    PublicServiceCategory.POLICE_STATIONS: (
        "police", "police station", "police stations", "law enforcement",
    ),
    PublicServiceCategory.FIRE_STATIONS: (
        "fire station", "fire stations", "fire service", "fire brigade", "fire emergency",
    ),
    PublicServiceCategory.GOVERNMENT_SERVICES: (
        "government service", "government services", "government office",
        "department", "license", "licence", "registration", "passport",
    ),
    PublicServiceCategory.EMERGENCY_INFORMATION: (
        "emergency number", "emergency contact", "ambulance",
        "emergency service", "emergency services", "hotline",
    ),
    PublicServiceCategory.CITIZEN_COMPLAINTS: (
        "complaint", "complaints", "report a problem", "report an issue",
        "streetlight", "broken streetlight", "garbage", "garbage collection",
        "waste collection", "water leak", "road damage", "pothole",
    ),
}

# Deterministic tie-break order used only when two or more categories match
# the same (highest) number of keyword phrases in one query. Life/safety
# oriented categories are placed first, general administrative services
# last, mirroring the order a human triaging a citizen request would use.
_CATEGORY_PRIORITY: tuple[PublicServiceCategory, ...] = (
    PublicServiceCategory.EMERGENCY_INFORMATION,
    PublicServiceCategory.FIRE_STATIONS,
    PublicServiceCategory.HOSPITALS,
    PublicServiceCategory.POLICE_STATIONS,
    PublicServiceCategory.CITIZEN_COMPLAINTS,
    PublicServiceCategory.GOVERNMENT_SERVICES,
)

# Broad emergency-intent wording. Deliberately separate from
# EMERGENCY_INFORMATION's category keywords: this only flags that the
# citizen's wording expresses urgency, not which specialist category
# applies (e.g. "emergency hospital" is category=hospitals, is_emergency=True).
_EMERGENCY_INTENT_TERMS: tuple[str, ...] = (
    "emergency", "urgent", "immediately", "critical", "life threatening",
)

# Query phrases mapped to the exact `complaint_type` values used in
# backend/data/seed/public_services_data.json, so a match here can be
# passed straight through to IR/agent code without further translation.
_COMPLAINT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "streetlight_malfunction": ("broken streetlight", "streetlight", "street light", "street lamp"),
    "waste_collection_missed": ("garbage collection", "waste collection", "garbage", "trash collection"),
    "water_supply_issue": ("water leak", "water leakage", "water supply", "water shortage"),
    "road_damage": ("road damage", "damaged road", "pothole", "potholes"),
    "illegal_construction": ("illegal construction", "unauthorized construction", "unpermitted construction"),
    "drainage_blockage": ("drainage", "blocked drain", "drain blockage", "clogged drain"),
    "noise_complaint": ("noise complaint", "noise disturbance", "loud noise"),
}

# Conversational filler removed only when building `search_query`. Category,
# location, emergency, and complaint-type detection run on the full
# normalized text (before filler removal), so multi-word keyword phrases
# such as "report a problem" still match literally.
_FILLER_WORDS: frozenset[str] = frozenset({
    "i", "a", "an", "the", "is", "are", "was", "were", "am", "be",
    "need", "needs", "needed", "want", "wants", "wanted", "please",
    "find", "get", "looking", "for", "to", "of", "on", "at", "in", "near",
    "nearest", "there", "any", "some", "me", "my", "you", "tell", "about",
    "can", "could", "would", "do", "does", "did", "where", "what", "who",
    "how", "when", "which", "this", "that", "these", "those", "and", "or",
})


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace to single spaces.

    Mirrors `DeterministicQueryRouter._normalize` in
    `app/agents/orchestrator/router.py` so query normalization stays
    consistent across the project.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


class PublicServicesQuery(BaseModel):
    """Structured result of parsing one natural-language Public Services query.

    Optional fields are `None` whenever the rule-based detectors cannot
    confidently determine them -- nothing here is inferred or guessed.
    """

    original_query: str
    category: PublicServiceCategory | None = None
    location: str | None = None
    district: str | None = None
    is_emergency: bool = False
    complaint_type: str | None = None
    search_query: str = Field(default="")


class PublicServicesNLP:
    """Rule-based query understanding for the Public Services domain.

    Detects the target category, an explicit location/district mention, a
    general emergency-intent signal, and (for complaint-shaped queries) a
    known complaint type -- then produces a filler-trimmed `search_query`
    suitable for `PublicServicesIR.search()`. See module docstring for the
    determinism/no-LLM constraints this class operates under.
    """

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self._known_locations, self._known_districts = self._load_known_entities()

    def parse(self, query: str) -> PublicServicesQuery:
        """Parse one raw query into a `PublicServicesQuery`."""
        normalized = _normalize(query)
        if not normalized:
            logger.info("Received an empty or whitespace-only public services query.")
            return PublicServicesQuery(original_query=query, search_query="")

        padded = f" {normalized} "
        category = self._detect_category(padded)
        location, district = self._detect_location_and_district(padded)
        is_emergency = self._detect_emergency_intent(padded)
        complaint_type = self._detect_complaint_type(padded)
        search_query = self._build_search_query(normalized)

        logger.info(
            "Parsed public services query category=%s location=%s district=%s "
            "is_emergency=%s complaint_type=%s",
            category.value if category else None,
            location,
            district,
            is_emergency,
            complaint_type,
        )
        return PublicServicesQuery(
            original_query=query,
            category=category,
            location=location,
            district=district,
            is_emergency=is_emergency,
            complaint_type=complaint_type,
            search_query=search_query,
        )

    def _load_known_entities(self) -> tuple[dict[str, str], dict[str, str]]:
        """Derive known location/district vocabularies from the seed dataset.

        Returns two dicts mapping the normalized (lowercase) form of each
        name to its original display form, e.g. `{"nuwara eliya": "Nuwara
        Eliya"}`. Falls back to empty vocabularies (all location/district
        detection then stays `None`) if the seed file is missing or invalid.
        """
        try:
            with self.data_path.open("r", encoding="utf-8") as handle:
                raw: dict[str, Any] = json.load(handle)
        except FileNotFoundError:
            logger.error("Seed data file not found at %s; location/district detection disabled.", self.data_path)
            return {}, {}
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse seed data JSON at %s: %s", self.data_path, exc)
            return {}, {}

        locations: dict[str, str] = {}
        districts: dict[str, str] = {}
        for category_key in _LOCATION_BEARING_CATEGORIES:
            for item in raw.get(category_key, []):
                location_value = item.get("location")
                if isinstance(location_value, str) and location_value.strip():
                    locations[_normalize(location_value)] = location_value
                district_value = item.get("district")
                if isinstance(district_value, str) and district_value.strip():
                    districts[_normalize(district_value)] = district_value
        return locations, districts

    @staticmethod
    def _detect_category(padded: str) -> PublicServiceCategory | None:
        """Pick the category with the most matching keyword phrases.

        Ties are broken using `_CATEGORY_PRIORITY`. Returns `None` when no
        category has any matching phrase.
        """
        match_counts = {
            category: sum(1 for phrase in phrases if f" {phrase} " in padded)
            for category, phrases in _CATEGORY_KEYWORDS.items()
        }
        highest = max(match_counts.values())
        if highest == 0:
            return None
        tied = {category for category, count in match_counts.items() if count == highest}
        if len(tied) == 1:
            return next(iter(tied))
        return next(category for category in _CATEGORY_PRIORITY if category in tied)

    def _detect_location_and_district(self, padded: str) -> tuple[str | None, str | None]:
        """Detect at most one location and one district, by leftmost mention.

        A "<place> district" phrase (e.g. "Colombo district") is treated as
        a district mention. Any other known place name mentioned in the
        query is treated as a location, even if the same name also appears
        in the district vocabulary elsewhere in the dataset (e.g. "Colombo"
        is both a hospital location and a common district value).
        """
        district: str | None = None
        district_candidates = [
            (padded.find(f" {district_lower} district "), display)
            for district_lower, display in self._known_districts.items()
            if f" {district_lower} district " in padded
        ]
        if district_candidates:
            district = min(district_candidates)[1]

        location: str | None = None
        location_candidates = [
            (padded.find(f" {location_lower} "), display)
            for location_lower, display in self._known_locations.items()
            if f" {location_lower} " in padded and display != district
        ]
        if location_candidates:
            location = min(location_candidates)[1]

        return location, district

    @staticmethod
    def _detect_emergency_intent(padded: str) -> bool:
        """Return True when the query expresses explicit urgency wording."""
        return any(f" {term} " in padded for term in _EMERGENCY_INTENT_TERMS)

    @staticmethod
    def _detect_complaint_type(padded: str) -> str | None:
        """Detect the leftmost-mentioned known complaint type, if any."""
        candidates = [
            (padded.find(f" {phrase} "), complaint_type)
            for complaint_type, phrases in _COMPLAINT_TYPE_KEYWORDS.items()
            for phrase in phrases
            if f" {phrase} " in padded
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    @staticmethod
    def _build_search_query(normalized: str) -> str:
        """Strip conversational filler tokens, preserving the original order.

        Falls back to the full normalized text if filtering would remove
        every token (e.g. a query made entirely of filler words), so
        `search_query` is never empty for a non-blank input.
        """
        tokens = normalized.split()
        filtered = [token for token in tokens if token not in _FILLER_WORDS]
        return " ".join(filtered) if filtered else normalized
