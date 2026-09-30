"""JWT bearer verification dependency; it does not issue or refresh tokens."""

import logging
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.security.principal import AuthenticatedPrincipal

logger = logging.getLogger(__name__)
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")


def _unauthorized(category: str) -> HTTPException:
    logger.warning("Authentication rejected category=%s", category)
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "unauthorized", "message": "A valid bearer token is required."},
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> AuthenticatedPrincipal | None:
    """Return validated subject claims, or None when auth is disabled locally."""
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("missing_bearer_token")
    if settings.jwt_secret is None or not settings.jwt_issuer or not settings.jwt_audience:
        # Settings validation should prevent this configuration from reaching requests.
        logger.error("Authentication configuration is incomplete")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "auth_unavailable", "message": "Authentication is temporarily unavailable."},
        )

    try:
        import jwt

        claims: dict[str, Any] = jwt.decode(
            credentials.credentials,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
        subject = claims.get("sub")
        issuer = claims.get("iss")
        token_id = claims.get("jti")
        if not isinstance(subject, str) or not subject.strip():
            raise InvalidTokenError("invalid subject")
        if not isinstance(issuer, str) or (token_id is not None and not isinstance(token_id, str)):
            raise InvalidTokenError("invalid identity claims")
        return AuthenticatedPrincipal(subject=subject, issuer=issuer, token_id=token_id)
    except ImportError:
        logger.error("JWT verification dependency is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "auth_unavailable", "message": "Authentication is temporarily unavailable."},
        ) from None
    except jwt.ExpiredSignatureError:
        raise _unauthorized("expired_token") from None
    except jwt.InvalidTokenError:
        raise _unauthorized("invalid_token") from None
    except (TypeError, ValueError):
        raise _unauthorized("malformed_token") from None
