"""Minimal application-service bootstrap and specialist registration hook."""

from app.agents.mobility.agent import MobilityAgent
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.environment import EnvironmentAgent
from app.agents.public_services.agent import PublicServicesAgent
from app.agents.registry import AgentRegistry


def create_agent_registry() -> AgentRegistry:
    """Create shared registry and register available specialist agents."""
    registry = AgentRegistry()
    registry.register(MobilityAgent())
    registry.register(PublicServicesAgent())
    registry.register(EnvironmentAgent())
    return registry


def create_orchestrator(registry: AgentRegistry) -> CityOrchestratorAgent:
    """Build one reusable orchestrator around the application's shared registry."""
    return CityOrchestratorAgent(registry)

