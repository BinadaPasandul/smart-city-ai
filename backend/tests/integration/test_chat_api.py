from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.agents.base import BaseAgent
from app.agents.contracts import (
    AgentError,
    AgentErrorCode,
    AgentRequest,
    AgentResponse,
    AgentSource,
)
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.router import DeterministicQueryRouter, RoutingDecision, RoutingResult
from app.agents.orchestrator.synthesizer import SynthesisResult
from app.agents.registry import AgentRegistry
from app.api.dependencies import get_orchestrator
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.core.config import settings
from app.main import app


class FakeRouter:
    def __init__(self, names, *, clarify=False, supported=True) -> None:
        self.names = names
        self.clarify = clarify
        self.supported = supported

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        return RoutingResult(
            decision=RoutingDecision(
                agent_names=self.names,
                confidence=1.0,
                reason="Test route",
                needs_clarification=self.clarify,
            ),
            routing_method="gemini" if self.supported else "deterministic_fallback",
        )


class FakeAgent(BaseAgent):
    def __init__(self, name: str, answer: str, *, failure: bool = False) -> None:
        super().__init__(name)
        self.answer = answer
        self.failure = failure
        self.requests: list[AgentRequest] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if self.failure:
            raise RuntimeError("private test traceback detail")
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=self.answer,
            sources=[AgentSource(
                name=f"{self.name} source",
                source_type="test",
                url="https://example.test/evidence",
                retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                metadata={"record": "abc"},
            )],
            metadata={"safe_internal": "not public"},
        )


class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls = []

    async def synthesize(self, query, summary) -> SynthesisResult:
        self.calls.append((query, summary))
        answers = [
            result.response.answer
            for result in summary.results
            if result.success and result.response is not None
        ]
        return SynthesisResult(
            answer=" ".join(answers),
            used_agents=[result.agent_name for result in summary.results if result.success],
        )


def make_orchestrator(names, agents=(), *, clarify=False, synthesizer=None) -> CityOrchestratorAgent:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    return CityOrchestratorAgent(
        registry,
        FakeRouter(names, clarify=clarify),
        synthesizer=synthesizer,
    )


@pytest.fixture
def client_with_orchestrator():
    original = app.dependency_overrides.get(get_orchestrator)

    def override(orchestrator):
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator

    try:
        with TestClient(app) as client:
            yield client, override
    finally:
        if original is None:
            app.dependency_overrides.pop(get_orchestrator, None)
        else:
            app.dependency_overrides[get_orchestrator] = original


def test_chat_request_trims_message_and_context_defaults() -> None:
    request = ChatRequest(message="  Find parking  ")
    assert request.message == "Find parking"
    assert request.context == {}
    assert ChatRequest(message="query", context={"locale": "en"}).context == {"locale": "en"}


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_chat_request_rejects_blank_message(message: str) -> None:
    with pytest.raises(ValueError):
        ChatRequest(message=message)


def test_chat_request_rejects_message_over_configured_limit() -> None:
    with pytest.raises(ValueError):
        ChatRequest(message="x" * (settings.chat_max_message_length + 1))


def test_chat_response_maps_sources_error_and_only_allowlisted_metadata() -> None:
    source = AgentSource(name="Transit feed", source_type="api", url="https://example.test")
    response = AgentResponse(
        request_id=uuid4(),
        agent_name="orchestrator",
        success=False,
        answer="Could you clarify?",
        sources=[source],
        metadata={
            "routing_method": "gemini",
            "execution_status": "partial_success",
            "selected_agents": ["mobility"],
            "execution_summary": {"private": "do not expose"},
            "prompt": "do not expose",
        },
        error=AgentError(code=AgentErrorCode.NEEDS_CLARIFICATION, message="Please clarify."),
    )
    public = ChatResponse.from_agent_response(response)
    dumped = public.model_dump(mode="json")
    assert dumped["request_id"] == str(response.request_id)
    assert dumped["sources"][0]["name"] == "Transit feed"
    assert dumped["error"]["code"] == "needs_clarification"
    assert dumped["metadata"]["routing_method"] == "gemini"
    assert "execution_summary" not in dumped["metadata"]
    assert "prompt" not in dumped["metadata"]


def test_chat_route_executes_single_specialist_and_preserves_request_id(client_with_orchestrator) -> None:
    agent = FakeAgent("mobility", "Parking is available.")
    orchestrator = make_orchestrator(["mobility"], [agent])
    client, override = client_with_orchestrator
    override(orchestrator)

    response = client.post("/api/v1/chat", json={"message": "Where can I park?", "context": {"area": "Fort"}})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Parking is available."
    assert body["request_id"] == str(agent.requests[0].request_id)
    assert agent.requests[0].context == {"area": "Fort"}
    assert body["metadata"]["synthesis_method"] == "single_specialist_passthrough"
    assert body["sources"][0]["name"] == "mobility source"


def test_chat_route_executes_and_synthesizes_two_specialists(client_with_orchestrator) -> None:
    mobility = FakeAgent("mobility", "Traffic is heavy.")
    environment = FakeAgent("environment", "Air quality is moderate.")
    fake_synthesis = FakeSynthesizer()
    orchestrator = make_orchestrator(["mobility", "environment"], [mobility, environment], synthesizer=fake_synthesis)
    client, override = client_with_orchestrator
    override(orchestrator)

    response = client.post("/api/v1/chat", json={"message": "traffic and air quality"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Traffic is heavy. Air quality is moderate."
    assert body["metadata"]["selected_agents"] == ["mobility", "environment"]
    assert body["metadata"]["execution_status"] == "complete"
    assert body["metadata"]["synthesis_method"] == "gemini"
    assert len(fake_synthesis.calls) == 1
    assert len(mobility.requests) == len(environment.requests) == 1
    assert str(mobility.requests[0].request_id) == str(environment.requests[0].request_id) == body["request_id"]
    assert len(body["sources"]) == 2
    assert "execution_summary" not in body["metadata"]


def test_chat_route_maps_clarification_and_unsupported_responses(client_with_orchestrator) -> None:
    client, override = client_with_orchestrator
    override(make_orchestrator([], clarify=True))
    clarification = client.post("/api/v1/chat", json={"message": "Is it okay there?"})
    assert clarification.status_code == 200
    assert clarification.json()["success"] is False
    assert clarification.json()["error"]["code"] == "needs_clarification"

    override(CityOrchestratorAgent(AgentRegistry(), FakeRouter([], supported=False)))
    unsupported = client.post("/api/v1/chat", json={"message": "Write a space poem"})
    assert unsupported.status_code == 200
    assert unsupported.json()["error"]["code"] == "unsupported_request"


def test_chat_route_returns_partial_success_without_stack_trace(client_with_orchestrator) -> None:
    mobility = FakeAgent("mobility", "Traffic is heavy.")
    environment = FakeAgent("environment", "", failure=True)
    orchestrator = make_orchestrator(
        ["mobility", "environment"], [mobility, environment], synthesizer=FakeSynthesizer()
    )
    client, override = client_with_orchestrator
    override(orchestrator)

    response = client.post("/api/v1/chat", json={"message": "traffic and air quality"})

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["execution_status"] == "partial_success"
    assert body["metadata"]["failed_agents"] == ["environment"]
    assert "Traffic is heavy." in body["answer"]
    assert "environment could not be retrieved" in body["answer"]
    assert "private test traceback detail" not in response.text


def test_empty_production_style_registry_returns_agent_not_found(client_with_orchestrator) -> None:
    client, override = client_with_orchestrator
    override(CityOrchestratorAgent(AgentRegistry(), DeterministicQueryRouter()))

    response = client.post("/api/v1/chat", json={"message": "Where can I park?"})

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["error"]["code"] == "agent_not_found"


def test_invalid_request_is_422_and_health_still_works(client_with_orchestrator) -> None:
    client, _ = client_with_orchestrator
    assert client.post("/api/v1/chat", json={"message": "   "}).status_code == 422
    assert client.post("/api/v1/chat", json={"message": "x" * (settings.chat_max_message_length + 1)}).status_code == 422
    assert client.post("/api/v1/chat", json={"other": "not a message"}).status_code == 422
    assert client.get("/api/v1/health").status_code == 200


def test_unexpected_api_exception_is_safe_500(client_with_orchestrator) -> None:
    class ExplodingOrchestrator:
        name = "orchestrator"

        async def execute(self, request):
            raise RuntimeError("private stack trace and credential")

    client, override = client_with_orchestrator
    override(ExplodingOrchestrator())  # type: ignore[arg-type]

    response = client.post("/api/v1/chat", json={"message": "parking"})

    assert response.status_code == 500
    assert "private stack trace" not in response.text
    assert "internal_error" in response.text
