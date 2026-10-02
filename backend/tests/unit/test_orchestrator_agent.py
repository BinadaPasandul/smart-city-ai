from uuid import uuid4

import pytest

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.router import DeterministicQueryRouter
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


class FakeFailingWithOwnErrorAgent(BaseAgent):
    """Returns a normal (non-exception) failure with its own specific AgentError,
    mirroring EnvironmentAgent's ambiguous-location response shape."""

    def __init__(self, name: str, *, code: AgentErrorCode, message: str) -> None:
        super().__init__(name)
        self._code = code
        self._message = message

    async def execute(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=False,
            error=AgentError(code=self._code, message=self._message),
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
    orchestrator = CityOrchestratorAgent(registry, DeterministicQueryRouter())
    request = AgentRequest(request_id=uuid4(), query=query)

    response = await orchestrator.execute(request)

    assert response.request_id == request.request_id
    assert response.agent_name == name
    assert response.success is True
    assert response.answer == f"Handled by {name}"
    assert response.metadata["routing_method"] == "deterministic_fallback"
    assert agents[name].received == [request]
    assert orchestrator.name == "orchestrator"
    assert isinstance(orchestrator, BaseAgent)


@pytest.mark.asyncio
async def test_unsupported_query_returns_structured_failure(populated_registry) -> None:
    orchestrator = CityOrchestratorAgent(populated_registry[0], DeterministicQueryRouter())
    request = AgentRequest(query="Write me a poem about space")

    response = await orchestrator.execute(request)

    assert response.request_id == request.request_id
    assert response.agent_name == "orchestrator"
    assert response.success is False
    assert response.error.code == AgentErrorCode.UNSUPPORTED_REQUEST


@pytest.mark.asyncio
async def test_unregistered_routed_specialist_returns_agent_not_found() -> None:
    orchestrator = CityOrchestratorAgent(AgentRegistry(), DeterministicQueryRouter())
    request = AgentRequest(query="Where can I park?")

    response = await orchestrator.execute(request)

    assert response.request_id == request.request_id
    assert response.error.code == AgentErrorCode.AGENT_NOT_FOUND


@pytest.mark.asyncio
async def test_specialist_exception_returns_safe_execution_failure() -> None:
    registry = AgentRegistry()
    registry.register(FakeAgent("mobility", fail=True))
    orchestrator = CityOrchestratorAgent(registry, DeterministicQueryRouter())
    request = AgentRequest(query="Where can I park?")

    response = await orchestrator.execute(request)

    assert response.request_id == request.request_id
    assert response.agent_name == "orchestrator"
    assert response.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert "private provider details" not in response.error.message


@pytest.mark.asyncio
async def test_agents_are_routed_independently(populated_registry) -> None:
    registry, agents = populated_registry
    orchestrator = CityOrchestratorAgent(registry, DeterministicQueryRouter())

    for query in ("bus route", "rain forecast", "police station"):
        await orchestrator.execute(AgentRequest(query=query))

    assert [len(agent.received) for agent in agents.values()] == [1, 1, 1]


@pytest.mark.asyncio
async def test_single_specialist_failure_preserves_its_own_error() -> None:
    """A single selected specialist's specific AgentError (e.g. the
    Environment agent's "ambiguous city" INVALID_REQUEST) must reach the
    caller directly, not the generic multi-agent aggregation message."""
    registry = AgentRegistry()
    registry.register(
        FakeFailingWithOwnErrorAgent(
            "environment",
            code=AgentErrorCode.INVALID_REQUEST,
            message="The city name is ambiguous. Specify a country or provide latitude and longitude.",
        )
    )
    orchestrator = CityOrchestratorAgent(registry, DeterministicQueryRouter())
    request = AgentRequest(query="What's the weather in Colombo?")

    response = await orchestrator.execute(request)

    assert response.success is False
    assert response.error.code == AgentErrorCode.INVALID_REQUEST
    assert response.error.message == (
        "The city name is ambiguous. Specify a country or provide latitude and longitude."
    )
    assert response.error.message != "No selected specialist completed the request."


@pytest.mark.asyncio
async def test_multi_specialist_failure_aggregation_is_unchanged() -> None:
    """Fix 3 must not alter behavior when more than one specialist fails --
    the existing generic aggregation message/code selection still applies."""
    registry = AgentRegistry()
    registry.register(
        FakeFailingWithOwnErrorAgent(
            "mobility", code=AgentErrorCode.TIMEOUT, message="mobility timed out"
        )
    )
    registry.register(
        FakeFailingWithOwnErrorAgent(
            "environment", code=AgentErrorCode.TIMEOUT, message="environment timed out"
        )
    )

    class TwoAgentRouter:
        async def route(self, query: str, *, request_id: str | None = None):
            from app.agents.orchestrator.router import RoutingDecision, RoutingResult, SpecialistAgentName

            return RoutingResult(
                decision=RoutingDecision(
                    agent_names=[SpecialistAgentName.MOBILITY, SpecialistAgentName.ENVIRONMENT],
                    confidence=1.0,
                    reason="test",
                    needs_clarification=False,
                ),
                routing_method="deterministic_fallback",
            )

    orchestrator = CityOrchestratorAgent(registry, TwoAgentRouter())
    request = AgentRequest(query="traffic and weather")

    response = await orchestrator.execute(request)

    assert response.success is False
    assert response.error.code == AgentErrorCode.TIMEOUT
    assert response.error.message == "No selected specialist completed the request."
