"""Minimal application-service bootstrap and specialist registration hook."""

from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.registry import AgentRegistry


def create_agent_registry() -> AgentRegistry:
    """Create the shared empty registry; integration can register real agents here."""
    return AgentRegistry()


def create_orchestrator(registry: AgentRegistry) -> CityOrchestratorAgent:
    """Build one reusable orchestrator around the application's shared registry."""
    return CityOrchestratorAgent(registry)
