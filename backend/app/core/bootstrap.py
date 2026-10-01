"""Minimal application-service bootstrap and specialist registration hook."""

from app.agents.mobility.agent import MobilityAgent
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.web_search import WebSearchService
from app.agents.environment import EnvironmentAgent
from app.agents.public_services.agent import PublicServicesAgent
from app.agents.registry import AgentRegistry
from app.core.config import settings
from app.ir.web_search import TavilyWebSearchProvider


def create_agent_registry() -> AgentRegistry:
    """Create shared registry and register available specialist agents."""
    registry = AgentRegistry()
    registry.register(MobilityAgent())
    registry.register(PublicServicesAgent())
    registry.register(EnvironmentAgent())
    return registry


def create_web_search_service() -> WebSearchService:
    """Build the optional shared search provider; credentials remain server-side."""
    provider = (
        TavilyWebSearchProvider(
            settings.tavily_api_key.get_secret_value(),
            timeout_seconds=settings.web_search_timeout_seconds,
        )
        if settings.web_search_enabled and settings.tavily_api_key is not None
        else None
    )
    return WebSearchService(
        provider,
        enabled=settings.web_search_enabled,
        on_partial_failure=settings.web_search_on_partial_failure,
        timeout_seconds=settings.web_search_timeout_seconds,
        max_results=settings.web_search_max_results,
        query_max_length=settings.web_search_query_max_length,
    )


def create_orchestrator(
    registry: AgentRegistry, web_search: WebSearchService | None = None
) -> CityOrchestratorAgent:
    """Build one reusable orchestrator around the application's shared registry."""
    return CityOrchestratorAgent(registry, web_search=web_search)
