"""Simple keyword router used before LLM-based routing is introduced."""

import re
from typing import Protocol


class QueryRouter(Protocol):
    """Interface allowing the routing strategy to be replaced independently."""

    def route(self, query: str) -> str | None:
        """Return a specialist name, or None when no category matches."""


class DeterministicQueryRouter:
    """Classify queries by keyword matches without external services."""

    KEYWORDS: dict[str, tuple[str, ...]] = {
        "mobility": (
            "traffic", "public transport", "transport", "bus", "buses", "train", "trains",
            "route", "routes", "parking", "park", "ev charging", "charging station",
        ),
        "environment": (
            "weather", "rain", "air quality", "pollution", "waste", "environment",
            "environmental", "air pollution",
        ),
        "public_services": (
            "hospital", "hospitals", "police", "fire station", "fire stations", "emergency",
            "government service", "government services", "citizen complaint", "city service complaint",
        ),
    }

    # On equal match counts, prefer public services, then environment, then mobility.
    TIE_PRIORITY = ("public_services", "environment", "mobility")

    @staticmethod
    def _normalize(query: str) -> str:
        """Lowercase and normalize punctuation/spacing for phrase matching."""
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", query.lower())).strip()

    def route(self, query: str) -> str | None:
        """Choose the category with the most distinct matching keywords."""
        normalized = self._normalize(query)
        if not normalized:
            return None

        match_counts = {
            category: sum(f" {keyword} " in f" {normalized} " for keyword in keywords)
            for category, keywords in self.KEYWORDS.items()
        }
        highest_count = max(match_counts.values())
        if highest_count == 0:
            return None

        tied = {category for category, count in match_counts.items() if count == highest_count}
        return next(category for category in self.TIE_PRIORITY if category in tied)
