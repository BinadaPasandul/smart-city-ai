import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.base import BaseAgent
from app.agents.contracts import AgentErrorCode, AgentRequest, AgentResponse, AgentSource
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.execution import ExecutionStatus
from app.agents.orchestrator.gemini_router import FallbackQueryRouter, GeminiQueryRouter
from app.agents.orchestrator.router import (
    DeterministicQueryRouter,
    RoutingDecision,
    RoutingResult,
    SpecialistAgentName,
)
from app.agents.registry import AgentRegistry


class MockRouter:
    def __init__(self, agent_names: list[str], *, clarify: bool = False) -> None:
        self.names = agent_names
        self.clarify = clarify

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        return RoutingResult(
            decision=RoutingDecision(
                agent_names=self.names,
                confidence=0.99,
                reason="Mocked routing plan.",
                needs_clarification=self.clarify,
            ),
            routing_method="gemini",
        )


class FakeSpecialist(BaseAgent):
    def __init__(
        self,
        name: str,
        *,
        delay: float = 0,
        failure: Exception | None = None,
    ) -> None:
        super().__init__(name)
        self.delay = delay
        self.failure = failure
        self.calls: list[AgentRequest] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.calls.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure:
            raise self.failure
        source = AgentSource(
            name=f"{self.name} source",
            source_type="test",
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=f"FAKE_{self.name.upper()}_RESULT",
            sources=[source],
        )


class FakeMobilityAgent(FakeSpecialist):
    def __init__(self, **kwargs) -> None:
        super().__init__("mobility", **kwargs)


class FakeEnvironmentAgent(FakeSpecialist):
    def __init__(self, **kwargs) -> None:
        super().__init__("environment", **kwargs)


class FakePublicServicesAgent(FakeSpecialist):
    def __init__(self, **kwargs) -> None:
        super().__init__("public_services", **kwargs)


class MockGeminiModels:
    def __init__(self, agent_names: list[str]) -> None:
        self.generate_content = AsyncMock(
            return_value=SimpleNamespace(
                text=RoutingDecision(
                    agent_names=agent_names,
                    confidence=0.96,
                    reason="The request requires these specialists.",
                    needs_clarification=False,
                ).model_dump_json()
            )
        )


class MockGeminiClient:
    def __init__(self, names: list[str]) -> None:
        self.models = MockGeminiModels(names)


def make_registry(*agents: FakeSpecialist) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    return registry


def gemini_router(names: list[str]) -> FallbackQueryRouter:
    primary = GeminiQueryRouter("mock-key", client=MockGeminiClient(names))
    return FallbackQueryRouter(primary, DeterministicQueryRouter())


@pytest.mark.parametrize(
    "names",
    [
        ["mobility", "environment"],
        ["mobility", "public_services"],
        ["environment", "public_services"],
        ["mobility", "environment", "public_services"],
    ],
)
def test_routing_decision_accepts_one_to_three_specialists(names: list[str]) -> None:
    plan = RoutingDecision(
        agent_names=names,
        confidence=0.9,
        reason="Valid multi-agent plan.",
        needs_clarification=False,
    )
    assert [name.value for name in plan.agent_names] == names


@pytest.mark.parametrize(
    "names",
    [
        ["mobility", "mobility"],
        ["mobility", "environment", "public_services", "mobility"],
        ["invented_agent"],
    ],
)
def test_routing_decision_rejects_duplicate_unknown_or_excess_agents(names: list[str]) -> None:
    with pytest.raises(ValueError):
        RoutingDecision(
            agent_names=names,
            confidence=0.9,
            reason="Invalid plan.",
            needs_clarification=False,
        )


def test_clarification_cannot_select_agents() -> None:
    with pytest.raises(ValueError):
        RoutingDecision(
            agent_names=["mobility"],
            confidence=0.9,
            reason="Contradictory plan.",
            needs_clarification=True,
        )


@pytest.mark.asyncio
async def test_deterministic_router_selects_all_matching_domains_in_stable_order() -> None:
    router = DeterministicQueryRouter()

    result = await router.route("How is traffic and air quality today?")
    assert [name.value for name in result.decision.agent_names] == ["mobility", "environment"]
    result = await router.route("Where is a hospital and what is traffic like?")
    assert [name.value for name in result.decision.agent_names] == ["public_services", "mobility"]
    result = await router.route("Write a poem about space")
    assert result.decision.agent_names == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("names", "agents"),
    [
        (["mobility"], [FakeMobilityAgent()]),
        (["mobility", "environment"], [FakeMobilityAgent(), FakeEnvironmentAgent()]),
        (
            ["mobility", "environment", "public_services"],
            [FakeMobilityAgent(), FakeEnvironmentAgent(), FakePublicServicesAgent()],
        ),
    ],
)
async def test_selected_specialists_execute_and_results_are_collected(names, agents) -> None:
    request_id = uuid4()
    request = AgentRequest(request_id=request_id, query="multi-domain request")
    orchestrator = CityOrchestratorAgent(
        make_registry(*agents), gemini_router(names), execution_timeout_seconds=1
    )

    response = await orchestrator.execute(request)

    assert response.request_id == request_id
    assert response.metadata["execution_status"] == ExecutionStatus.COMPLETE.value
    assert response.metadata["selected_agents"] == names
    summary = response.metadata["execution_summary"]
    assert summary["successful_agents"] == names
    assert len(summary["results"]) == len(names)
    assert all(agent.calls == [request] for agent in agents)
    assert all(result["response"]["request_id"] == str(request_id) for result in summary["results"])
    assert len(response.sources) == len(agents)
    if len(agents) > 1:
        assert response.agent_name == "orchestrator"
        assert {result["response"]["answer"] for result in summary["results"]} == {
            f"FAKE_{name.upper()}_RESULT" for name in names
        }
    else:
        assert response.agent_name == names[0]


@pytest.mark.asyncio
async def test_one_success_and_one_exception_yields_partial_success_without_exception_text() -> None:
    mobility = FakeMobilityAgent()
    environment = FakeEnvironmentAgent(failure=RuntimeError("secret exception detail"))
    request = AgentRequest(query="traffic and weather")
    response = await CityOrchestratorAgent(
        make_registry(mobility, environment),
        gemini_router(["mobility", "environment"]),
    ).execute(request)

    assert response.metadata["execution_status"] == ExecutionStatus.PARTIAL_SUCCESS.value
    summary = response.metadata["execution_summary"]
    assert summary["successful_agents"] == ["mobility"]
    assert summary["failed_agents"] == ["environment"]
    assert "secret exception detail" not in response.model_dump_json()
    assert [source.name for source in response.sources] == ["mobility source"]


@pytest.mark.asyncio
async def test_one_success_and_one_missing_agent_yields_partial_success() -> None:
    mobility = FakeMobilityAgent()
    response = await CityOrchestratorAgent(
        make_registry(mobility), gemini_router(["mobility", "environment"])
    ).execute(AgentRequest(query="traffic and weather"))

    summary = response.metadata["execution_summary"]
    assert response.metadata["execution_status"] == ExecutionStatus.PARTIAL_SUCCESS.value
    assert summary["failed_agents"] == ["environment"]
    assert summary["results"][1]["error"]["code"] == AgentErrorCode.AGENT_NOT_FOUND.value


@pytest.mark.asyncio
async def test_one_success_and_one_timeout_yields_partial_success() -> None:
    mobility = FakeMobilityAgent()
    environment = FakeEnvironmentAgent(delay=0.1)
    response = await CityOrchestratorAgent(
        make_registry(mobility, environment),
        gemini_router(["mobility", "environment"]),
        execution_timeout_seconds=0.02,
    ).execute(AgentRequest(query="traffic and weather"))

    summary = response.metadata["execution_summary"]
    environment_result = summary["results"][1]
    assert response.metadata["execution_status"] == ExecutionStatus.PARTIAL_SUCCESS.value
    assert summary["successful_agents"] == ["mobility"]
    assert environment_result["timed_out"] is True
    assert environment_result["error"]["code"] == AgentErrorCode.TIMEOUT.value


@pytest.mark.asyncio
async def test_all_specialist_exceptions_yield_failed_status() -> None:
    agents = [
        FakeMobilityAgent(failure=RuntimeError("internal one")),
        FakeEnvironmentAgent(failure=RuntimeError("internal two")),
    ]
    response = await CityOrchestratorAgent(
        make_registry(*agents), gemini_router(["mobility", "environment"])
    ).execute(AgentRequest(query="two domains"))

    assert response.metadata["execution_status"] == ExecutionStatus.FAILED.value
    assert response.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert "internal" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_all_missing_specialists_yield_failed_status() -> None:
    response = await CityOrchestratorAgent(
        AgentRegistry(), gemini_router(["mobility", "environment"])
    ).execute(AgentRequest(query="two domains"))

    assert response.metadata["execution_status"] == ExecutionStatus.FAILED.value
    assert response.error.code == AgentErrorCode.AGENT_NOT_FOUND
    assert response.metadata["failed_agents"] == ["mobility", "environment"]


@pytest.mark.asyncio
async def test_only_timed_out_specialist_sets_overall_timeout_error() -> None:
    environment = FakeEnvironmentAgent(delay=0.1)
    response = await CityOrchestratorAgent(
        make_registry(environment),
        gemini_router(["environment"]),
        execution_timeout_seconds=0.01,
    ).execute(AgentRequest(query="environment request"))

    assert response.error.code == AgentErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_clarification_and_unsupported_queries_invoke_no_specialists() -> None:
    mobility = FakeMobilityAgent()
    registry = make_registry(mobility)
    clarify = CityOrchestratorAgent(registry, MockRouter([], clarify=True))
    clarified = await clarify.execute(AgentRequest(query="Is it okay there?"))
    unsupported = await CityOrchestratorAgent(
        registry, DeterministicQueryRouter()
    ).execute(AgentRequest(query="Write a poem about space"))

    assert clarified.error.code == AgentErrorCode.NEEDS_CLARIFICATION
    assert unsupported.error.code == AgentErrorCode.UNSUPPORTED_REQUEST
    assert mobility.calls == []


@pytest.mark.asyncio
async def test_two_delayed_specialists_execute_concurrently() -> None:
    mobility = FakeMobilityAgent(delay=0.2)
    environment = FakeEnvironmentAgent(delay=0.2)
    orchestrator = CityOrchestratorAgent(
        make_registry(mobility, environment),
        gemini_router(["mobility", "environment"]),
        execution_timeout_seconds=2,
    )
    started = time.perf_counter()

    response = await orchestrator.execute(AgentRequest(query="two domains"))

    elapsed = time.perf_counter() - started
    assert response.metadata["execution_status"] == ExecutionStatus.COMPLETE.value
    assert elapsed < 0.36, f"Expected concurrent execution under 0.36s; took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_fallback_multi_domain_request_works_without_gemini_key() -> None:
    router = FallbackQueryRouter(GeminiQueryRouter(None), DeterministicQueryRouter())
    agents = [FakeMobilityAgent(), FakeEnvironmentAgent()]
    response = await CityOrchestratorAgent(make_registry(*agents), router).execute(
        AgentRequest(query="How are traffic and air quality?")
    )

    assert response.metadata["selected_agents"] == ["mobility", "environment"]
    assert response.metadata["routing_method"] == "deterministic_fallback"
    assert all(agent.calls for agent in agents)
