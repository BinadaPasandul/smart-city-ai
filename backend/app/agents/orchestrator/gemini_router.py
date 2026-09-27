"""Gemini structured-output router and deterministic fallback wrapper."""

import asyncio
import logging
from typing import Any

from pydantic import ValidationError

from app.agents.orchestrator.router import (
    QueryRouter,
    RoutingDecision,
    RoutingResult,
)

logger = logging.getLogger(__name__)

ROUTING_INSTRUCTIONS = """Classify the user's request into exactly one supported city-service category.
mobility: traffic, public transportation, buses, trains, routes, transport navigation,
parking, EV charging, or transport conditions.
environment: weather, air quality, pollution, environmental conditions, or waste.
public_services: hospitals, police, fire stations, emergency services, government
services, or citizen complaints.
If the request does not contain enough context to determine a relevant category,
set agent_name to null and needs_clarification to true. If it is unrelated to these
city services, set agent_name to null and needs_clarification to false.
Return only the requested structured classification. Never answer the user's question.
"""


class GeminiRoutingError(Exception):
    """Safe, categorized Gemini routing failure for fallback handling."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class GeminiQueryRouter:
    """Classify a query using Gemini JSON output, without generating an answer."""

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = "gemini-2.5-flash",
        client: Any | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client
        self._timeout_seconds = timeout_seconds

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        if not self._api_key and self._client is None:
            raise GeminiRoutingError("api_key_unavailable")

        client = self._get_client()
        try:
            response = await asyncio.wait_for(
                client.models.generate_content(
                    model=self._model,
                    contents=f"User request:\n{query}",
                    config=self._generation_config(),
                ),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise GeminiRoutingError("timeout") from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise GeminiRoutingError("empty_structured_output")
        try:
            decision = RoutingDecision.model_validate_json(text)
        except (ValidationError, ValueError, TypeError) as exc:
            raise GeminiRoutingError("invalid_structured_output") from exc
        return RoutingResult(decision=decision, routing_method="gemini")

    def _get_client(self) -> Any:
        if self._client is None:
            # Defer the SDK import and client creation until a configured request arrives.
            from google import genai

            self._client = genai.Client(api_key=self._api_key).aio
        return self._client

    @staticmethod
    def _generation_config() -> dict[str, Any]:
        return {
            "system_instruction": ROUTING_INSTRUCTIONS,
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": RoutingDecision,
        }


class FallbackQueryRouter:
    """Use Gemini first and deterministic keyword routing when it cannot decide."""

    def __init__(self, primary: QueryRouter, fallback: QueryRouter) -> None:
        self._primary = primary
        self._fallback = fallback

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        fallback_category = "no_specialist_decision"
        try:
            result = await self._primary.route(query, request_id=request_id)
            if result.decision.needs_clarification:
                return result
            if result.decision.agent_name is not None:
                return result
        except GeminiRoutingError as exc:
            fallback_category = exc.category
        except Exception as exc:
            # Keep exception text out of logs; provider errors can contain request details.
            fallback_category = type(exc).__name__

        result = await self._fallback.route(query, request_id=request_id)
        selected = result.decision.agent_name.value if result.decision.agent_name else "none"
        logger.warning(
            "Routing fallback request_id=%s reason=%s selected_agent=%s",
            request_id or "unavailable",
            fallback_category,
            selected,
        )
        return result
