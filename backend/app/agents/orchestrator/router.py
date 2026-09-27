"""Simple keyword router used before LLM-based routing is introduced."""

import re
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpecialistAgentName(str, Enum):
    """Names the Orchestrator is allowed to select."""

    MOBILITY = "mobility"
    ENVIRONMENT = "environment"
    PUBLIC_SERVICES = "public_services"


class RoutingDecision(BaseModel):
    """Validated routing outcome; it contains no user-facing answer."""

    model_config = ConfigDict(extra="forbid")

    agent_name: SpecialistAgentName | None
    confidence: float = Field(strict=True, ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)
    needs_clarification: bool

    @model_validator(mode="after")
    def clarification_has_no_agent(self) -> "RoutingDecision":
        if self.needs_clarification and self.agent_name is not None:
            raise ValueError("A clarification decision cannot select a specialist")
        return self


class RoutingResult(BaseModel):
    """Decision plus the strategy that produced it."""

    decision: RoutingDecision
    routing_method: Literal["gemini", "deterministic_fallback"]


class QueryRouter(Protocol):
    """Interface allowing the routing strategy to be replaced independently."""

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        """Return a validated decision; None agent means no category matched."""


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

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        """Choose the category with the most distinct matching keywords."""
        normalized = self._normalize(query)
        if not normalized:
            decision = RoutingDecision(
                agent_name=None,
                confidence=1.0,
                reason="The query is empty after normalization.",
                needs_clarification=False,
            )
            return RoutingResult(decision=decision, routing_method="deterministic_fallback")

        match_counts = {
            category: sum(f" {keyword} " in f" {normalized} " for keyword in keywords)
            for category, keywords in self.KEYWORDS.items()
        }
        highest_count = max(match_counts.values())
        if highest_count == 0:
            decision = RoutingDecision(
                agent_name=None,
                confidence=1.0,
                reason="No supported city-service keywords matched.",
                needs_clarification=False,
            )
            return RoutingResult(decision=decision, routing_method="deterministic_fallback")

        tied = {category for category, count in match_counts.items() if count == highest_count}
        selected = next(category for category in self.TIE_PRIORITY if category in tied)
        decision = RoutingDecision(
            agent_name=SpecialistAgentName(selected),
            confidence=1.0,
            reason=f"Matched {highest_count} keyword(s) for {selected}.",
            needs_clarification=False,
        )
        return RoutingResult(decision=decision, routing_method="deterministic_fallback")
