from uuid import uuid4

import pytest

from app.agents.base import BaseAgent
from app.agents.contracts import AgentErrorCode, AgentRequest, AgentResponse
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.registry import AgentRegistry


class FakeAgent(BaseAgent):
    def __init__(self, name: str, *, fail: bool = False) -> None:
        super().__init__(name)
        self.fail = fail
        self.received: list[AgentRequest] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.received.append(request)
        if self.fail:
            raise RuntimeError("private provider details")
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=f"Handled by {self.name}",
        )


@pytest.fixture
def populated_registry() -> tuple[AgentRegistry, dict[str, FakeAgent]]:
    registry = AgentRegistry()
    agents = {
        "mobility": FakeAgent("mobility"),
        "environment": FakeAgent("environment"),
        "public_services": FakeAgent("public_services"),
    }
    for agent in agents.values():
        registry.register(agent)
    return registry, agents


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "name"),
    [
        ("Where can I park near Colombo Fort?", "mobility"),
        ("How is the air quality today?", "environment"),
        ("Where is the nearest hospital?", "public_services"),
    ],
)
async def test_orchestrator_routes_to_registered_specialist_and_preserves_response(
    populated_registry: tuple[AgentRegistry, dict[str, FakeAgent]], query: str, name: str
) -> None:
    registry, agents = populated_registry
    orchestrator = CityOrchestratorAgent(registry)
    request = AgentRequest(request_id=uuid4(), query=query)

    response = await orchestrator.execute(request)

    assert response == AgentResponse(
        request_id=request.request_id,
        agent_name=name,
        success=True,
        answer=f"Handled by {name}",
    )
    assert agents[name].received == [request]
    assert orchestrator.name == "orchestrator"
    assert isinstance(orchestrator, BaseAgent)


@pytest.mark.asyncio
async def test_unsupported_query_returns_structured_failure(populated_registry) -> None:
    orchestrator = CityOrchestratorAgent(populated_registry[0])
    request = AgentRequest(query="Write me a poem about space")

    response = await orchestrator.execute(request)

    assert response.request_id == request.request_id
    assert response.agent_name == "orchestrator"
    assert response.success is False
    assert response.error.code == AgentErrorCode.UNSUPPORTED_REQUEST


@pytest.mark.asyncio
async def test_unregistered_routed_specialist_returns_agent_not_found() -> None:
    orchestrator = CityOrchestratorAgent(AgentRegistry())
    request = AgentRequest(query="Where can I park?")

    response = await orchestrator.execute(request)

    assert response.request_id == request.request_id
    assert response.error.code == AgentErrorCode.AGENT_NOT_FOUND


@pytest.mark.asyncio
async def test_specialist_exception_returns_safe_execution_failure() -> None:
    registry = AgentRegistry()
    registry.register(FakeAgent("mobility", fail=True))
    orchestrator = CityOrchestratorAgent(registry)
    request = AgentRequest(query="Where can I park?")

    response = await orchestrator.execute(request)

    assert response.request_id == request.request_id
    assert response.agent_name == "orchestrator"
    assert response.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert "private provider details" not in response.error.message


@pytest.mark.asyncio
async def test_agents_are_routed_independently(populated_registry) -> None:
    registry, agents = populated_registry
    orchestrator = CityOrchestratorAgent(registry)

    for query in ("bus route", "rain forecast", "police station"):
        await orchestrator.execute(AgentRequest(query=query))

    assert [len(agent.received) for agent in agents.values()] == [1, 1, 1]
