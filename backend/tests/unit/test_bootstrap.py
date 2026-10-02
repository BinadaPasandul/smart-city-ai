from app.agents.environment import EnvironmentAgent
from app.agents.public_services.agent import PublicServicesAgent
from app.agents.registry import AgentRegistry
from app.core.bootstrap import create_agent_registry


def test_create_agent_registry_returns_an_agent_registry() -> None:
    registry = create_agent_registry()

    assert isinstance(registry, AgentRegistry)


def test_public_services_agent_is_registered() -> None:
    registry = create_agent_registry()

    agent = registry.get("public_services")

    assert isinstance(agent, PublicServicesAgent)


def test_environment_agent_is_registered() -> None:
    registry = create_agent_registry()

    agent = registry.get("environment")

    assert isinstance(agent, EnvironmentAgent)


def test_registered_public_services_agent_name_is_exact() -> None:
    registry = create_agent_registry()

    agent = registry.get("public_services")

    assert agent.name == "public_services"


def test_registry_lists_the_registered_specialist() -> None:
    registry = create_agent_registry()

    names = [agent.name for agent in registry.list_agents()]

    assert set(names) == {"mobility", "environment", "public_services"}
