"""Public HTTP adapters for account registration and login."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.api.dependencies import enforce_auth_rate_limit, get_auth_service
from app.api.schemas.auth import LoginRequest, LoginResponse, RegisterRequest, UserPublic
from app.repositories.user_repository import DuplicateAccountError
from app.services.auth_service import AuthService, InvalidCredentialsError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(enforce_auth_rate_limit)])


def _persistence_error(request: Request, error: SQLAlchemyError) -> HTTPException:
    logger.error(
        "Account persistence failed request_id=%s exception_type=%s",
        getattr(request.state, "request_id", "unavailable"), type(error).__name__,
    )
    return HTTPException(
        status_code=503 if isinstance(error, OperationalError) else 500,
        detail={"code": "database_unavailable", "message": "Account service is unavailable."},
    )


@router.post(
    "/register", status_code=201, response_model=UserPublic,
    summary="Register a user account",
    responses={409: {"description": "Account already exists"}, 503: {"description": "Account service unavailable"}},
)
async def register(
    request: Request, body: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> UserPublic:
    try:
        user = await service.register(
            email=str(body.email), password=body.password.get_secret_value(),
            display_name=body.display_name,
        )
    except DuplicateAccountError:
        raise HTTPException(
            status_code=409,
            detail={"code": "account_exists", "message": "Account cannot be registered."},
        ) from None
    except SQLAlchemyError as exc:
        raise _persistence_error(request, exc) from None
    return UserPublic.model_validate(user)


@router.post(
    "/login", response_model=LoginResponse,
    summary="Issue an access token for valid credentials",
    responses={401: {"description": "Invalid credentials"}, 503: {"description": "Account service unavailable"}},
)
async def login(
    request: Request, body: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> LoginResponse:
    try:
        token, expires_in = await service.login(
            email=str(body.email), password=body.password.get_secret_value()
        )
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_credentials", "message": "Invalid email or password."},
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except SQLAlchemyError as exc:
        raise _persistence_error(request, exc) from None
    return LoginResponse(access_token=token, expires_in=expires_in)
