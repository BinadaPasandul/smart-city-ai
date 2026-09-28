"""Dependencies for resolving shared application services."""

from fastapi import Request

from app.agents.orchestrator.agent import CityOrchestratorAgent


def get_orchestrator(request: Request) -> CityOrchestratorAgent:
    """Return the application-scoped orchestrator instance."""
    return request.app.state.orchestrator
