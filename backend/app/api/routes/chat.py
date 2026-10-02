"""HTTP adapter for the internal City Orchestrator pipeline."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.agents.contracts import AgentRequest
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.api.dependencies import enforce_chat_rate_limit, get_orchestrator
from app.api.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Submit a citizen request",
    description="Routes a validated message through the City Orchestrator and returns its structured result.",
    dependencies=[Depends(enforce_chat_rate_limit)],
    responses={
        401: {"description": "A valid Bearer token is required when authentication is enabled."},
        429: {"description": "Chat rate limit exceeded; see Retry-After."},
        500: {"description": "Unexpected API processing error."},
    },
)
async def chat(
    request: Request,
    body: ChatRequest,
    orchestrator: CityOrchestratorAgent = Depends(get_orchestrator),
) -> ChatResponse:
    """Adapt the public chat DTO to the internal request/response contract."""
    agent_request = AgentRequest(
        request_id=request.state.request_id,
        query=body.message,
        context={"user_context": body.context},
    )
    try:
        agent_response = await orchestrator.execute(agent_request)
    except Exception as exc:
        logger.error(
            "Chat request failed request_id=%s endpoint=/chat exception_type=%s",
            agent_request.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "request_id": str(agent_request.request_id),
                "success": False,
                "error": {
                    "code": "internal_error",
                    "message": "The request could not be processed at this time.",
                },
            },
        ) from None

    response = ChatResponse.from_agent_response(agent_response)
    logger.info(
        "Chat request completed request_id=%s endpoint=/chat success=%s status=%s agents=%s synthesis=%s",
        response.request_id,
        response.success,
        response.metadata.execution_status,
        response.metadata.selected_agents,
        response.metadata.synthesis_method,
    )
    return response
