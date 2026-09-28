"""Typed multi-agent routing decisions and deterministic keyword routing."""

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
    """Validated routing plan; it contains no user-facing answer."""

    model_config = ConfigDict(extra="forbid")

    agent_names: list[SpecialistAgentName] = Field(max_length=3)
    confidence: float = Field(strict=True, ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)
    needs_clarification: bool

    @model_validator(mode="after")
    def clarification_has_no_agent(self) -> "RoutingDecision":
        if len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("A routing decision cannot contain duplicate specialists")
        if self.needs_clarification and self.agent_names:
            raise ValueError("A clarification decision cannot select specialists")
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

    # If first-match positions also tie, prefer public services, environment, mobility.
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
                agent_names=[],
                confidence=1.0,
                reason="The query is empty after normalization.",
                needs_clarification=False,
            )
            return RoutingResult(decision=decision, routing_method="deterministic_fallback")

        match_counts = {
            category: sum(f" {keyword} " in f" {normalized} " for keyword in keywords)
            for category, keywords in self.KEYWORDS.items()
        }
        selected = [category for category, count in match_counts.items() if count > 0]
        if not selected:
            decision = RoutingDecision(
                agent_names=[],
                confidence=1.0,
                reason="No supported city-service keywords matched.",
                needs_clarification=False,
            )
            return RoutingResult(decision=decision, routing_method="deterministic_fallback")

        priority = {category: index for index, category in enumerate(self.TIE_PRIORITY)}
        padded = f" {normalized} "
        first_match = {
            category: min(
                padded.find(f" {keyword} ")
                for keyword in self.KEYWORDS[category]
                if f" {keyword} " in padded
            )
            for category in selected
        }
        selected.sort(
            key=lambda category: (
                -match_counts[category],
                first_match[category],
                priority[category],
            )
        )
        decision = RoutingDecision(
            agent_names=[SpecialistAgentName(category) for category in selected],
            confidence=1.0,
            reason="Matched categories ordered by match count, then first mention in the query.",
            needs_clarification=False,
        )
        return RoutingResult(decision=decision, routing_method="deterministic_fallback")
