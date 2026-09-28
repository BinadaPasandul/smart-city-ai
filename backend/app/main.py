"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.bootstrap import create_agent_registry, create_orchestrator
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name)
app.state.agent_registry = create_agent_registry()
app.state.orchestrator = create_orchestrator(app.state.agent_registry)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.api_v1_prefix)
