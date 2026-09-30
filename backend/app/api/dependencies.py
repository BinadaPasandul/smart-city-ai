"""Dependencies for resolving shared application services."""

import logging

from fastapi import Depends, HTTPException, Request

from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.security.auth import get_current_principal
from app.security.principal import AuthenticatedPrincipal
from app.security.rate_limit import InMemoryRateLimiter

logger = logging.getLogger(__name__)


def get_orchestrator(request: Request) -> CityOrchestratorAgent:
    """Return the application-scoped orchestrator instance."""
    return request.app.state.orchestrator


def get_rate_limiter(request: Request) -> InMemoryRateLimiter:
    """Return the shared process-local chat limiter."""
    return request.app.state.rate_limiter


async def enforce_chat_rate_limit(
    request: Request,
    principal: AuthenticatedPrincipal | None = Depends(get_current_principal),
    limiter: InMemoryRateLimiter = Depends(get_rate_limiter),
) -> None:
    """Limit by verified subject, or direct peer address in local auth-disabled mode."""
    if principal is not None:
        key = f"subject:{principal.subject}"
    else:
        peer = request.client.host if request.client is not None else "unknown-client"
        key = f"client:{peer}"
    allowed, retry_after = await limiter.consume(key)
    if not allowed:
        logger.warning(
            "Chat rate limit rejected request_id=%s identity_type=%s",
            getattr(request.state, "request_id", "unavailable"),
            "subject" if principal else "client",
        )
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limit_exceeded", "message": "Too many requests. Please retry later."},
            headers={"Retry-After": str(retry_after)},
        )
