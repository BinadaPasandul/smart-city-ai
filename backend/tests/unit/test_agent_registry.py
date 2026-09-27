import pytest

from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse
from app.agents.registry import AgentNotFoundError, AgentRegistry, DuplicateAgentError


class FakeAgent(BaseAgent):
    async def execute(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(request_id=request.request_id, agent_name=self.name, success=True)


def test_register_retrieve_and_list_agents() -> None:
    registry = AgentRegistry()
    agent = FakeAgent("mobility")

    registry.register(agent)

    assert registry.get("mobility") is agent
    assert registry.list_agents() == [agent]


def test_duplicate_name_registration_is_rejected() -> None:
    registry = AgentRegistry()
    registry.register(FakeAgent("mobility"))

    with pytest.raises(DuplicateAgentError, match="already registered"):
        registry.register(FakeAgent("mobility"))


def test_unknown_agent_lookup_has_predictable_error() -> None:
    with pytest.raises(AgentNotFoundError, match="not registered"):
        AgentRegistry().get("missing")


def test_multiple_agent_implementations_coexist() -> None:
    registry = AgentRegistry()
    mobility = FakeAgent("mobility")
    environment = FakeAgent("environment")
    registry.register(mobility)
    registry.register(environment)

    assert registry.list_agents() == [mobility, environment]
