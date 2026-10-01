"""FastAPI application entry point."""

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.bootstrap import create_agent_registry, create_orchestrator, create_web_search_service
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import DatabaseManager
from app.security.password import PasswordHasher
from app.security.rate_limit import InMemoryRateLimiter

configure_logging(settings.log_level)

@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        yield
    finally:
        await application.state.web_search.aclose()
        if application.state.db_manager is not None:
            await application.state.db_manager.aclose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.state.agent_registry = create_agent_registry()
app.state.web_search = create_web_search_service()
app.state.orchestrator = create_orchestrator(app.state.agent_registry, app.state.web_search)
app.state.db_manager = DatabaseManager(settings.database_url) if settings.database_url else None
app.state.password_hasher = PasswordHasher()
app.state.auth_rate_limiter = InMemoryRateLimiter(
    settings.auth_rate_limit_requests, settings.auth_rate_limit_window_seconds,
)
app.state.rate_limiter = InMemoryRateLimiter(
    settings.chat_rate_limit_requests,
    settings.chat_rate_limit_window_seconds,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Request-ID"],
)
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.middleware("http")
async def add_security_and_request_headers(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Assign server request IDs and add API response security headers."""
    request_id = uuid4()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = str(request_id)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path == f"{settings.api_v1_prefix}/chat" or request.url.path.startswith(
        f"{settings.api_v1_prefix}/auth/"
    ):
        response.headers["Cache-Control"] = "no-store"
    return response
