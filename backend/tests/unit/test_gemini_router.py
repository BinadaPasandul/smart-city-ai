from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agents.contracts import AgentErrorCode, AgentRequest
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.gemini_router import (
    FallbackQueryRouter,
    GeminiQueryRouter,
    GeminiRoutingError,
)
from app.agents.orchestrator.router import DeterministicQueryRouter, RoutingDecision
from app.agents.registry import AgentRegistry
from tests.unit.test_orchestrator_agent import FakeAgent


class MockGeminiClient:
    def __init__(self, text: str = "", *, exception: Exception | None = None) -> None:
        self.models = SimpleNamespace(generate_content=AsyncMock())
        if exception:
            self.models.generate_content.side_effect = exception
        else:
            self.models.generate_content.return_value = SimpleNamespace(text=text)


def decision_json(agent_names: list[str], *, confidence: float = 0.95, clarification: bool = False) -> str:
    import json

    return json.dumps(
        {
            "agent_names": agent_names,
            "confidence": confidence,
            "reason": "The request matches this city service.",
            "needs_clarification": clarification,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_names", [["mobility"], ["environment"], ["public_services"]])
async def test_gemini_router_validates_supported_specialist_decisions(agent_names: list[str]) -> None:
    client = MockGeminiClient(decision_json(agent_names))
    router = GeminiQueryRouter("test-key", client=client)

    result = await router.route("A natural-language request")

    assert [name.value for name in result.decision.agent_names] == agent_names
    assert result.decision.confidence == 0.95
    assert result.routing_method == "gemini"
    call = client.models.generate_content.await_args.kwargs
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["response_json_schema"] == RoutingDecision.model_json_schema()


@pytest.mark.asyncio
async def test_router_uses_configured_model_and_keeps_explicit_override() -> None:
    client = MockGeminiClient(decision_json(["mobility"]))
    router = GeminiQueryRouter("test-key", model="test-model-id", client=client)

    await router.route("parking")

    assert client.models.generate_content.await_args.kwargs["model"] == "test-model-id"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_names",
    [
        ["mobility", "environment"],
        ["mobility", "public_services"],
        ["environment", "public_services"],
        ["mobility", "environment", "public_services"],
    ],
)
async def test_gemini_router_accepts_multi_specialist_plans(agent_names: list[str]) -> None:
    result = await GeminiQueryRouter(
        "test-key", client=MockGeminiClient(decision_json(agent_names))
    ).route("A multi-domain request")

    assert [name.value for name in result.decision.agent_names] == agent_names


@pytest.mark.asyncio
async def test_gemini_router_normalizes_specialist_order() -> None:
    result = await GeminiQueryRouter(
        "test-key", client=MockGeminiClient(decision_json(["public_services", "mobility"]))
    ).route("A multi-domain request")

    assert [name.value for name in result.decision.agent_names] == ["mobility", "public_services"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "agent_name"),
    [
        ("I need somewhere to leave my car before catching the train.", "mobility"),
        ("Is the air safe to breathe today?", "environment"),
        ("I need medical help nearby.", "public_services"),
    ],
)
async def test_gemini_classifies_indirect_natural_language(query: str, agent_name: str) -> None:
    router = GeminiQueryRouter("test-key", client=MockGeminiClient(decision_json([agent_name])))

    result = await router.route(query)

    assert [name.value for name in result.decision.agent_names] == [agent_name]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "not json",
        decision_json(["made_up_agent"]),
        decision_json(["mobility"], confidence=1.5),
        decision_json(["mobility", "mobility"]),
        decision_json(["mobility", "environment", "public_services", "mobility"]),
        decision_json(["mobility"], clarification=True),
        '{"agent_names":["mobility"],"confidence":0.8}',
    ],
)
async def test_gemini_rejects_malformed_or_invalid_decisions(output: str) -> None:
    router = GeminiQueryRouter("test-key", client=MockGeminiClient(output))

    with pytest.raises(GeminiRoutingError):
        await router.route("question")


@pytest.mark.asyncio
async def test_gemini_represents_clarification_without_agent() -> None:
    router = GeminiQueryRouter(
        "test-key", client=MockGeminiClient(decision_json([], clarification=True))
    )

    result = await router.route("Is it okay there?")

    assert result.decision.agent_names == []
    assert result.decision.needs_clarification is True


@pytest.mark.asyncio
async def test_missing_api_key_uses_deterministic_fallback() -> None:
    router = FallbackQueryRouter(GeminiQueryRouter(None), DeterministicQueryRouter())

    result = await router.route("Where can I park?")

    assert [name.value for name in result.decision.agent_names] == ["mobility"]
    assert result.routing_method == "deterministic_fallback"


@pytest.mark.asyncio
async def test_gemini_exception_uses_deterministic_fallback() -> None:
    primary = GeminiQueryRouter(
        "test-key", client=MockGeminiClient(exception=TimeoutError("sensitive details"))
    )
    result = await FallbackQueryRouter(primary, DeterministicQueryRouter()).route(
        "Where can I park?", request_id="request-123"
    )

    assert [name.value for name in result.decision.agent_names] == ["mobility"]
    assert result.routing_method == "deterministic_fallback"


@pytest.mark.asyncio
async def test_invalid_gemini_output_uses_deterministic_fallback() -> None:
    primary = GeminiQueryRouter("test-key", client=MockGeminiClient("invalid json"))
    result = await FallbackQueryRouter(primary, DeterministicQueryRouter()).route(
        "Where can I park?"
    )

    assert [name.value for name in result.decision.agent_names] == ["mobility"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "agent_name"),
    [
        ("Where can I park?", "mobility"),
        ("Is air pollution high?", "environment"),
        ("Where is a hospital?", "public_services"),
    ],
)
async def test_deterministic_fallback_routes_each_supported_category(query: str, agent_name: str) -> None:
    primary = GeminiQueryRouter(None)
    result = await FallbackQueryRouter(primary, DeterministicQueryRouter()).route(query)

    assert [name.value for name in result.decision.agent_names] == [agent_name]


@pytest.mark.asyncio
async def test_both_strategies_unable_to_route_returns_unsupported() -> None:
    registry = AgentRegistry()
    router = FallbackQueryRouter(GeminiQueryRouter(None), DeterministicQueryRouter())
    response = await CityOrchestratorAgent(registry, router).execute(
        AgentRequest(query="Write a poem about space.")
    )

    assert response.error.code == AgentErrorCode.UNSUPPORTED_REQUEST
    assert response.metadata["routing_method"] == "deterministic_fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "agent_name"),
    [
        ("I need somewhere to leave my car before catching the train.", "mobility"),
        ("Is the air safe to breathe today?", "environment"),
        ("I need medical help nearby.", "public_services"),
    ],
)
async def test_orchestrator_uses_gemini_decision_and_preserves_request_id(query, agent_name) -> None:
    registry = AgentRegistry()
    agent = FakeAgent(agent_name)
    registry.register(agent)
    primary = GeminiQueryRouter(
        "test-key", client=MockGeminiClient(decision_json([agent_name]))
    )
    orchestrator = CityOrchestratorAgent(
        registry, FallbackQueryRouter(primary, DeterministicQueryRouter())
    )
    request = AgentRequest(query=query)

    response = await orchestrator.execute(request)

    assert agent.received == [request]
    assert response.request_id == request.request_id
    assert response.metadata["routing_method"] == "gemini"


@pytest.mark.asyncio
async def test_clarification_does_not_invoke_specialist() -> None:
    registry = AgentRegistry()
    mobility = FakeAgent("mobility")
    registry.register(mobility)
    primary = GeminiQueryRouter(
        "test-key", client=MockGeminiClient(decision_json([], clarification=True))
    )
    response = await CityOrchestratorAgent(
        registry, FallbackQueryRouter(primary, DeterministicQueryRouter())
    ).execute(AgentRequest(query="Is it okay there?"))

    assert response.error.code == AgentErrorCode.NEEDS_CLARIFICATION
    assert response.metadata["routing_method"] == "gemini"
    assert mobility.received == []
