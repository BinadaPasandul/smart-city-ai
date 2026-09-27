"""Minimal in-process City Orchestrator agent."""

import logging

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse
from app.agents.registry import AgentNotFoundError, AgentRegistry
from app.agents.orchestrator.gemini_router import FallbackQueryRouter, GeminiQueryRouter
from app.agents.orchestrator.router import DeterministicQueryRouter, QueryRouter
from app.core.config import settings

logger = logging.getLogger(__name__)


class CityOrchestratorAgent(BaseAgent):
    """Route one request to one registered specialist and return its result."""

    def __init__(self, registry: AgentRegistry, router: QueryRouter | None = None) -> None:
        super().__init__(
            "orchestrator",
            description="Routes citizen queries to one registered specialist agent.",
            capabilities=("gemini_routing", "deterministic_fallback"),
        )
        self._registry = registry
        self._router = router or FallbackQueryRouter(
            GeminiQueryRouter(settings.gemini_api_key),
            DeterministicQueryRouter(),
        )

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Route and execute exactly one specialist while preserving request identity."""
        if not request.query.strip():
            return self._failure(
                request,
                AgentErrorCode.INVALID_REQUEST,
                "The request query must not be empty.",
            )

        result = await self._router.route(request.query, request_id=str(request.request_id))
        decision = result.decision
        route_metadata = {"routing_method": result.routing_method}
        if decision.needs_clarification:
            return self._failure(
                request,
                AgentErrorCode.NEEDS_CLARIFICATION,
                "Could you provide a little more detail so I can direct your request?",
                metadata=route_metadata,
            )

        if decision.agent_name is None:
            return self._failure(
                request,
                AgentErrorCode.UNSUPPORTED_REQUEST,
                "The request does not match a supported city service category.",
                metadata=route_metadata,
            )

        agent_name = decision.agent_name.value

        try:
            specialist = self._registry.get(agent_name)
        except AgentNotFoundError:
            logger.warning("Routed agent '%s' is not registered", agent_name)
            return self._failure(
                request,
                AgentErrorCode.AGENT_NOT_FOUND,
                f"The routed agent '{agent_name}' is not available.",
                metadata=route_metadata,
            )

        try:
            response = await specialist.execute(request)
            return response.model_copy(update={"metadata": {**response.metadata, **route_metadata}})
        except Exception:
            logger.exception("Agent '%s' failed while handling request %s", agent_name, request.request_id)
            return self._failure(
                request,
                AgentErrorCode.AGENT_EXECUTION_FAILED,
                "The selected agent could not complete the request.",
                metadata=route_metadata,
            )

    def _failure(
        self,
        request: AgentRequest,
        code: AgentErrorCode,
        message: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=False,
            metadata=metadata or {},
            error=AgentError(code=code, message=message),
        )
