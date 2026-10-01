"""Deterministic query-intent parsing for Environment Agent requests.

This module classifies wording as weather, air quality, both, or unknown.
It performs no location resolution or data retrieval.
"""

import logging
import re
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EnvironmentIntent(str, Enum):
    """Supported request intents for environment information."""

    WEATHER = "weather"
    AIR_QUALITY = "air_quality"
    BOTH = "both"
    UNKNOWN = "unknown"


class EnvironmentQuery(BaseModel):
    """Parsed intent and normalized text for one Environment query."""

    original_query: str
    intent: EnvironmentIntent
    search_query: str = ""


# Match normalized whole phrases to avoid accidental substring matches.
# PM2.5 is normalized to "pm2 5" before matching.
_WEATHER_TERMS = (
    "weather", "temperature", "temperatures", "rain", "rainfall",
    "precipitation", "wind", "windy", "humidity", "humid", "sunny",
    "sunshine", "cloudy", "cloud cover", "thunderstorm", "weather forecast",
)
_AIR_QUALITY_TERMS = (
    "air quality", "air pollution", "pollution", "polluted air",
    "aqi", "pm2 5", "pm10", "particulate matter", "particulates",
    "smog", "air quality index", "pollution levels",
)
_FORECAST_TERMS = ("forecast", "forecasts", "outlook")


def _normalize(text: str) -> str:
    """Lowercase, replace punctuation with spaces, and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _contains_term(padded_query: str, terms: tuple[str, ...]) -> bool:
    return any(f" {term} " in padded_query for term in terms)


class EnvironmentNLP:
    """Classify Environment questions using documented keyword matching."""

    def parse(self, query: str) -> EnvironmentQuery:
        """Return a typed intent without making assumptions for unrelated text.

        Specific weather terms and air-quality terms are matched independently.
        A generic "forecast" or "outlook" means weather only when no air-quality
        term is present, so "air quality forecast" remains air-quality intent.
        """
        normalized = _normalize(query)
        if not normalized:
            return EnvironmentQuery(
                original_query=query,
                intent=EnvironmentIntent.UNKNOWN,
                search_query="",
            )

        padded = f" {normalized} "
        has_air_quality = _contains_term(padded, _AIR_QUALITY_TERMS)
        has_weather = _contains_term(padded, _WEATHER_TERMS)
        if not has_air_quality and _contains_term(padded, _FORECAST_TERMS):
            has_weather = True

        if has_weather and has_air_quality:
            intent = EnvironmentIntent.BOTH
        elif has_weather:
            intent = EnvironmentIntent.WEATHER
        elif has_air_quality:
            intent = EnvironmentIntent.AIR_QUALITY
        else:
            intent = EnvironmentIntent.UNKNOWN

        logger.info("Parsed environment query intent=%s", intent.value)
        return EnvironmentQuery(
            original_query=query,
            intent=intent,
            search_query=normalized,
        )


__all__ = ["EnvironmentIntent", "EnvironmentNLP", "EnvironmentQuery"]
