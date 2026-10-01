"""HTTP endpoint for routing citizen requests to specialist agents."""

import logging

from fastapi import APIRouter, Depends, Request

from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse
from app.agents.orchestrator.agent import CityOrchestratorAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


def get_orchestrator(request: Request) -> CityOrchestratorAgent:
    """Return the orchestrator initialized for this FastAPI application."""
    return request.app.state.orchestrator


@router.post("/query", response_model=AgentResponse)
async def submit_agent_request(
    agent_request: AgentRequest,
    orchestrator: CityOrchestratorAgent = Depends(get_orchestrator),
) -> AgentResponse:
    """Submit a shared agent request to the application orchestrator."""
    try:
        return await orchestrator.execute(agent_request)
    except Exception:
        logger.exception("Unexpected error while routing request %s", agent_request.request_id)
        return AgentResponse(
            request_id=agent_request.request_id,
            agent_name="orchestrator",
            success=False,
            error=AgentError(
                code=AgentErrorCode.AGENT_EXECUTION_FAILED,
                message="The request could not be processed.",
            ),
        )
