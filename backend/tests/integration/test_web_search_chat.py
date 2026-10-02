"""Offline authenticated HTTP walkthroughs for controlled search fallback."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.security.auth as auth
from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse, AgentSource
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.router import RoutingDecision, RoutingResult
from app.agents.orchestrator.synthesizer import SynthesisResult
from app.agents.orchestrator.web_search import WebSearchService
from app.agents.registry import AgentRegistry
from app.api.dependencies import get_orchestrator
from app.core.config import Settings
from app.ir.web_search import WebSearchProviderError, WebSearchResult
from app.main import app
from app.security.rate_limit import InMemoryRateLimiter

SECRET = "test-only-search-integration-secret-32-plus-characters"
TAVILY_TEST_KEY = "test-only-tavily-secret"
GEMINI_TEST_KEY = "test-only-gemini-secret"


def settings_for_auth():
    return Settings(
        _env_file=None, auth_enabled=True, jwt_secret=SecretStr(SECRET),
        jwt_issuer="search-integration", jwt_audience="search-api",
    )


def token():
    return jwt.encode({
        "sub": "test-user", "iss": "search-integration", "aud": "search-api",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }, SECRET, algorithm="HS256")


class Router:
    def __init__(self, names, *, clarify=False):
        self.names = names
        self.clarify = clarify

    async def route(self, query, *, request_id=None):
        return RoutingResult(
            decision=RoutingDecision(
                agent_names=self.names, confidence=1.0, reason="test routing",
                needs_clarification=self.clarify,
            ), routing_method="deterministic_fallback",
        )


class Specialist(BaseAgent):
    def __init__(self, name, *, fail=False, answer="Traffic is heavy."):
        super().__init__(name)
        self.fail = fail
        self.answer = answer
        self.requests: list[AgentRequest] = []

    async def execute(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("private specialist failure")
        return AgentResponse(
            request_id=request.request_id, agent_name=self.name, success=True, answer=self.answer,
            sources=[AgentSource(name="specialist feed", source_type="api", url="https://example.org/feed")],
        )


class Provider:
    provider_name = "tavily"

    def __init__(self, *, failure=None):
        self.failure = failure
        self.calls = []

    async def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self.failure:
            raise self.failure
        return [WebSearchResult(
            title="City notice", url="https://example.org/notice",
            snippet="Air quality is moderate.", provider="tavily",
        )]


class Synthesizer:
    def __init__(self):
        self.calls = []

    async def synthesize(self, query, summary, web_evidence=None):
        self.calls.append((query, summary, web_evidence))
        facts = [
            item.response.answer for item in summary.results
            if item.success and item.response is not None
        ]
        facts += [item.snippet for item in web_evidence or []]
        return SynthesisResult(
            answer=" ".join(facts),
            used_agents=[item.agent_name for item in summary.results if item.success],
        )


def build(names, agents, provider, *, clarify=False):
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    synthesizer = Synthesizer()
    service = WebSearchService(provider, enabled=True, max_results=5, query_max_length=500)
    orchestrator = CityOrchestratorAgent(
        registry, Router(names, clarify=clarify), synthesizer=synthesizer, web_search=service,
    )
    return orchestrator, synthesizer


@pytest.fixture
def chat_client(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", settings_for_auth)
    previous = app.dependency_overrides.get(get_orchestrator)
    old_limiter = app.state.rate_limiter
    app.state.rate_limiter = InMemoryRateLimiter(20, 60)

    def override(orchestrator):
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator

    try:
        with TestClient(app) as client:
            yield client, override
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_orchestrator, None)
        else:
            app.dependency_overrides[get_orchestrator] = previous
        app.state.rate_limiter = old_limiter


def authorized(client, message, *, context=None):
    return client.post(
        "/api/v1/chat", json={"message": message, "context": context or {}},
        headers={"Authorization": f"Bearer {token()}"},
    )


def test_jwt_protects_search_and_rate_limit_precedes_provider(chat_client):
    client, override = chat_client
    provider = Provider()
    orchestrator, _ = build(["mobility"], [Specialist("mobility", fail=True)], provider)
    override(orchestrator)
    assert client.post("/api/v1/chat", json={"message": "traffic"}).status_code == 401
    assert provider.calls == []
    old = app.state.rate_limiter
    app.state.rate_limiter = InMemoryRateLimiter(1, 60)
    try:
        assert authorized(client, "traffic").status_code == 200
        assert authorized(client, "traffic").status_code == 429
    finally:
        app.state.rate_limiter = old
    assert len(provider.calls) == 1


def test_complete_specialist_success_skips_search(chat_client):
    client, override = chat_client
    provider = Provider()
    orchestrator, _ = build(["mobility"], [Specialist("mobility")], provider)
    override(orchestrator)
    response = authorized(client, "traffic")
    assert response.status_code == 200
    assert response.json()["metadata"]["web_search_used"] is False
    assert response.json()["metadata"]["web_search_status"] == "not_needed"
    assert provider.calls == []


def test_total_failure_http_fallback_is_grounded_and_preserves_request_id_sources(chat_client, caplog):
    client, override = chat_client
    provider = Provider()
    agent = Specialist("mobility", fail=True)
    orchestrator, synthesis = build(["mobility"], [agent], provider)
    override(orchestrator)
    response = authorized(client, "traffic in Colombo", context={"subject": "never-forward", "token": "private-bearer"})
    body = response.json()
    assert response.status_code == 200 and body["success"] is True
    assert body["request_id"] == response.headers["x-request-id"] == str(agent.requests[0].request_id)
    assert body["metadata"]["execution_status"] == "failed"
    assert body["metadata"]["web_search_used"] is True
    assert body["metadata"]["web_search_status"] == "success"
    assert body["metadata"]["web_search_provider"] == "tavily"
    assert body["metadata"]["web_search_result_count"] == 1
    assert body["metadata"]["answer_basis"] == "web_fallback"
    assert body["answer"] == "Air quality is moderate."
    assert body["sources"][0]["url"] == "https://example.org/notice"
    assert body["sources"][0]["source_type"] == "web_search"
    assert provider.calls == [("traffic in Colombo", 5)] and len(synthesis.calls) == 1
    assert synthesis.calls[0][2][0].snippet == "Air quality is moderate."
    for secret in (SECRET, TAVILY_TEST_KEY, GEMINI_TEST_KEY, "private-bearer"):
        assert secret not in response.text and secret not in caplog.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_partial_failure_web_supplement_preserves_specialist_source(chat_client):
    client, override = chat_client
    provider = Provider()
    mobility = Specialist("mobility")
    environment = Specialist("environment", fail=True)
    orchestrator, synthesis = build(["mobility", "environment"], [mobility, environment], provider)
    override(orchestrator)
    response = authorized(client, "traffic and air quality")
    body = response.json()
    assert response.status_code == 200 and body["success"] is True
    assert body["metadata"]["execution_status"] == "partial_success"
    assert body["metadata"]["failed_agents"] == ["environment"]
    assert body["metadata"]["answer_basis"] == "specialist_plus_web"
    assert [source["source_type"] for source in body["sources"]] == ["api", "web_search"]
    assert "Traffic is heavy." in body["answer"]
    assert "Air quality is moderate." in body["answer"]
    assert "environment could not be retrieved" in body["answer"]
    assert synthesis.calls[0][1].status.value == "partial_success"
    assert len(provider.calls) == 1


def test_unsupported_and_clarification_do_not_use_provider(chat_client):
    client, override = chat_client
    provider = Provider()
    for clarify, expected in ((False, "unsupported_request"), (True, "needs_clarification")):
        orchestrator, _ = build([], [], provider, clarify=clarify)
        override(orchestrator)
        response = authorized(client, "Is it okay there?")
        assert response.status_code == 200
        assert response.json()["error"]["code"] == expected
    assert provider.calls == []


def test_provider_failure_is_safe_http_failure(chat_client, caplog):
    client, override = chat_client
    provider = Provider(failure=WebSearchProviderError("http_5xx"))
    orchestrator, synthesis = build(["mobility"], [], provider)
    override(orchestrator)
    response = authorized(client, "traffic")
    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["metadata"]["web_search_status"] == "failed"
    assert synthesis.calls == [] and len(provider.calls) == 1
    assert "http_5xx" not in response.text
    assert SECRET not in response.text and SECRET not in caplog.text
