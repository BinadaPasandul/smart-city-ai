"""Minimal in-process City Orchestrator agent."""

import logging

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse
from app.agents.registry import AgentNotFoundError, AgentRegistry
from app.agents.orchestrator.router import DeterministicQueryRouter, QueryRouter

logger = logging.getLogger(__name__)


class CityOrchestratorAgent(BaseAgent):
    """Route one request to one registered specialist and return its result."""

    def __init__(self, registry: AgentRegistry, router: QueryRouter | None = None) -> None:
        super().__init__(
            "orchestrator",
            description="Routes citizen queries to one registered specialist agent.",
            capabilities=("deterministic_routing",),
        )
        self._registry = registry
        self._router = router or DeterministicQueryRouter()

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Route and execute exactly one specialist while preserving request identity."""
        if not request.query.strip():
            return self._failure(
                request,
                AgentErrorCode.INVALID_REQUEST,
                "The request query must not be empty.",
            )

        agent_name = self._router.route(request.query)
        if agent_name is None:
            return self._failure(
                request,
                AgentErrorCode.UNSUPPORTED_REQUEST,
                "The request does not match a supported city service category.",
            )

        try:
            specialist = self._registry.get(agent_name)
        except AgentNotFoundError:
            logger.warning("Routed agent '%s' is not registered", agent_name)
            return self._failure(
                request,
                AgentErrorCode.AGENT_NOT_FOUND,
                f"The routed agent '{agent_name}' is not available.",
            )

        try:
            return await specialist.execute(request)
        except Exception:
            logger.exception("Agent '%s' failed while handling request %s", agent_name, request.request_id)
            return self._failure(
                request,
                AgentErrorCode.AGENT_EXECUTION_FAILED,
                "The selected agent could not complete the request.",
            )

    def _failure(self, request: AgentRequest, code: AgentErrorCode, message: str) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=False,
            error=AgentError(code=code, message=message),
        )
