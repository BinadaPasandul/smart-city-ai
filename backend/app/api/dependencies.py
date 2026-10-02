"""Dependencies for resolving shared application services."""

import logging

from fastapi import Depends, HTTPException, Request

from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.core.config import get_settings
from app.db.session import get_db_session
from app.repositories.user_repository import UserRepository
from app.security.auth import get_current_principal
from app.security.password import PasswordHasher
from app.security.principal import AuthenticatedPrincipal
from app.security.rate_limit import InMemoryRateLimiter
from app.security.token import JwtIssuer
from app.services.auth_service import AuthService
from sqlalchemy.ext.asyncio import AsyncSession

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


async def enforce_auth_rate_limit(request: Request) -> None:
    """Use an auth-specific, per-peer and per-endpoint budget."""
    peer = request.client.host if request.client is not None else "unknown-client"
    allowed, retry_after = await request.app.state.auth_rate_limiter.consume(
        f"auth:{request.url.path}:{peer}"
    )
    if not allowed:
        logger.warning(
            "Auth rate limit rejected request_id=%s endpoint=%s",
            getattr(request.state, "request_id", "unavailable"), request.url.path,
        )
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limit_exceeded", "message": "Too many requests. Please retry later."},
            headers={"Retry-After": str(retry_after)},
        )


def get_auth_service(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> AuthService:
    """Build a request-scoped service around one session and shared hashing policy."""
    try:
        issuer = JwtIssuer(get_settings())
    except ValueError:
        logger.error("JWT issuance configuration is incomplete")
        raise HTTPException(
            status_code=503,
            detail={"code": "auth_unavailable", "message": "Account service is unavailable."},
        ) from None
    hasher: PasswordHasher = request.app.state.password_hasher
    return AuthService(UserRepository(session), hasher, issuer)
