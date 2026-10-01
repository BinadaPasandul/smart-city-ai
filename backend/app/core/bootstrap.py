"""Minimal application-service bootstrap and specialist registration hook."""

from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.public_services.agent import PublicServicesAgent
from app.agents.registry import AgentRegistry


def create_agent_registry() -> AgentRegistry:
    """Create the shared registry and register the available specialist agents.

    Mobility and Environment agents are not implemented yet and are
    intentionally left unregistered; the orchestrator already handles a
    selected-but-unregistered specialist by returning AGENT_NOT_FOUND.
    """
    registry = AgentRegistry()
    registry.register(PublicServicesAgent())
    return registry


def create_orchestrator(registry: AgentRegistry) -> CityOrchestratorAgent:
    """Build one reusable orchestrator around the application's shared registry."""
    return CityOrchestratorAgent(registry)
