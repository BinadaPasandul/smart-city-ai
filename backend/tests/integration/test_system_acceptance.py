"""Cross-component offline acceptance coverage using registered specialists."""

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from types import MethodType
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse, AgentSource
from app.agents.environment import AIR_QUALITY_API_URL, FORECAST_API_URL, GEOCODING_API_URL
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.execution import ExecutionStatus
from app.agents.orchestrator.gemini_router import FallbackQueryRouter, GeminiRoutingError
from app.agents.orchestrator.router import (
    DeterministicQueryRouter,
    RoutingDecision,
    RoutingResult,
    SpecialistAgentName,
)
from app.agents.orchestrator.synthesizer import (
    DeterministicResultSynthesizer,
    SynthesisResult,
)
from app.agents.orchestrator.web_search import WebSearchService
from app.agents.registry import AgentRegistry
from app.api.schemas.chat import ChatResponse
from app.core.bootstrap import create_agent_registry
from app.core.config import Settings
from app.ir.web_search import WebSearchProviderError, WebSearchResult
from app.nlp.models import UnderstandingDecision
from app.nlp.pipeline import LocalNlpAnalyzer, RequestUnderstandingPipeline


NAMES = ("mobility", "environment", "public_services")
COMBINATIONS = [
    ("mobility",),
    ("environment",),
    ("public_services",),
    ("mobility", "environment"),
    ("mobility", "public_services"),
    ("environment", "public_services"),
    NAMES,
]
QUERY = (
    "How is traffic near Colombo Fort, what is the air quality in Colombo today, "
    "and where is a hospital in Colombo?"
)


class FixedRouter:
    def __init__(self, names: tuple[str, ...], *, locations: list[str] | None = None) -> None:
        self.names = list(names)
        self.locations = locations or ["Colombo"]
        self.calls: list[tuple[str, str | None]] = []

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        self.calls.append((query, request_id))
        return RoutingResult(
            decision=RoutingDecision(
                agent_names=self.names,
                confidence=0.99,
                reason="Acceptance test route.",
                needs_clarification=False,
                locations=self.locations,
                temporal_expressions=["today"] if "today" in query.casefold() else [],
            ),
            routing_method="deterministic_fallback",
            understanding_method="deterministic_fallback",
            local_nlp_confidence=0.99,
            locations=self.locations,
            temporal_expressions=["today"] if "today" in query.casefold() else [],
        )


class CapturingSynthesizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    async def synthesize(self, query, summary, web_evidence=None):
        self.calls.append((query, summary, list(web_evidence or [])))
        if self.fail:
            raise RuntimeError("controlled synthesis failure")
        good = [item for item in summary.results if item.success and item.response]
        answer = " ".join(item.response.answer for item in good)
        answer += " " + " ".join(item.snippet for item in web_evidence or [])
        return SynthesisResult(
            answer=answer.strip() or "No specialist evidence was available.",
            used_agents=[item.agent_name for item in good],
            limitations=[item.agent_name.value for item in summary.results if not item.success],
        )


class MockResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class MockOpenMeteo:
    def __init__(self, responses):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, *, params):
        self.calls.append((url, params))
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        return value


def open_meteo_responses():
    today = datetime.now(ZoneInfo("Asia/Colombo")).date()
    tomorrow = today + timedelta(days=1)
    times = [
        datetime.combine(today, time(23, 0)).isoformat(timespec="minutes"),
        datetime.combine(tomorrow, time(12, 0)).isoformat(timespec="minutes"),
    ]
    return {
        GEOCODING_API_URL: MockResponse({"results": []}),
        FORECAST_API_URL: MockResponse({
            "timezone": "Asia/Colombo",
            "hourly": {
                "time": times,
                "temperature_2m": [29.0, 30.0],
                "precipitation_probability": [10, 20],
                "weather_code": [1, 2],
            },
        }),
        AIR_QUALITY_API_URL: MockResponse({
            "timezone": "Asia/Colombo",
            "hourly": {
                "time": times,
                "pm2_5": [12.0, 13.0],
                "us_aqi": [40, 42],
                "european_aqi": [18, 20],
            },
        }),
    }


def observe(agent):
    """Capture requests while retaining the actual specialist implementation."""
    received: list[AgentRequest] = []
    original = agent.execute

    async def execute(_self, request):
        received.append(request)
        return await original(request)

    agent.execute = MethodType(execute, agent)
    return received


def make_real_system(names, monkeypatch, *, provider_responses=None):
    provider = MockOpenMeteo(provider_responses or open_meteo_responses())
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: provider)
    registry = create_agent_registry()
    captured = {name: observe(registry.get(name)) for name in names}
    router = FixedRouter(tuple(names))
    synthesizer = CapturingSynthesizer()
    orchestrator = CityOrchestratorAgent(
        registry,
        router,
        synthesizer=synthesizer,
        web_search=WebSearchService(None, enabled=False),
        execution_timeout_seconds=3,
    )
    return orchestrator, registry, captured, provider, router, synthesizer


def query_for(names: tuple[str, ...]) -> str:
    if names == ("mobility",):
        return "How is traffic near Colombo Fort?"
    if names == ("environment",):
        return "What is the weather today?"
    if names == ("public_services",):
        return "Find a hospital in Colombo."
    if names == ("mobility", "environment"):
        return "How is traffic near Colombo Fort and what is air quality in Colombo today?"
    if names == ("mobility", "public_services"):
        return "How is traffic near Colombo Fort and where is a hospital in Colombo?"
    if names == ("environment", "public_services"):
        return "What is the air quality in Colombo today and where is a hospital in Colombo?"
    return QUERY


@pytest.mark.parametrize("names", COMBINATIONS, ids=["mobility", "environment", "public-services", "mobility-environment", "mobility-public", "environment-public", "all-three"])
def test_all_seven_registered_specialist_combinations(names, monkeypatch):
    orchestrator, registry, captured, provider, router, synthesis = make_real_system(names, monkeypatch)
    request = AgentRequest(query=query_for(names), context={
        "user_context": {"latitude": 6.9271, "longitude": 79.8612},
        "opaque_user_value": {"keep": True},
    })

    response = asyncio.run(orchestrator.execute(request))
    dto = ChatResponse.from_agent_response(response)

    assert router.names == list(names)
    assert response.success is True
    assert response.request_id == request.request_id == dto.request_id
    assert response.metadata["selected_agents"] == list(names)
    assert response.metadata["execution_status"] == ExecutionStatus.COMPLETE.value
    assert response.metadata["successful_agents"] == list(names)
    assert all(type(registry.get(name)).__name__ in {
        "MobilityAgent", "EnvironmentAgent", "PublicServicesAgent"
    } for name in names)
    assert all(len(captured[name]) == 1 for name in names)
    assert all(captured[name][0].query == request.query for name in names)
    assert all(captured[name][0].request_id == request.request_id for name in names)
    assert all(captured[name][0].context["nlp"]["locations"] == ["Colombo"] for name in names)
    assert len(dto.sources) == len(response.sources) and dto.sources

    if "environment" in names:
        context = captured["environment"][0].context
        assert context["latitude"] == pytest.approx(6.9271)
        assert context["longitude"] == pytest.approx(79.8612)
        assert context["nlp"]["temporal_expressions"] == (
            ["today"] if "today" in request.query.casefold() else []
        )
        assert any(source.source_type == "api" for source in dto.sources)
        assert provider.calls
    else:
        assert provider.calls == []

    if "mobility" in names:
        assert any(source.metadata.get("category") == "traffic" for source in dto.sources)
    if "public_services" in names:
        assert any(source.metadata.get("category") == "hospitals" for source in dto.sources)

    if len(names) == 1:
        assert response.metadata["synthesis_method"] == "single_specialist_passthrough"
        assert synthesis.calls == []
    else:
        assert len(synthesis.calls) == 1
        synth_query, summary, _ = synthesis.calls[0]
        assert synth_query == request.query
        assert summary.status is ExecutionStatus.COMPLETE
        assert [item.agent_name.value for item in summary.results] == list(names)
        assert [item.value for item in summary.successful_agents] == list(names)
        assert response.metadata["synthesis_used_agents"] == list(names)


class ScenarioAgent(BaseAgent):
    def __init__(self, name, *, error=None, barrier=None):
        super().__init__(name)
        self.error = error
        self.barrier = barrier
        self.calls = []

    async def execute(self, request):
        self.calls.append(request)
        if self.barrier is not None:
            self.barrier["started"].add(self.name)
            if len(self.barrier["started"]) == 3:
                self.barrier["all_started"].set()
            await asyncio.wait_for(self.barrier["all_started"].wait(), timeout=1)
        if self.error is not None:
            if self.error is AgentErrorCode.TIMEOUT:
                await asyncio.sleep(1)
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                success=False,
                error=AgentError(code=self.error, message="controlled failure"),
            )
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=f"Evidence from {self.name}.",
            sources=[AgentSource(name=f"{self.name} test source", source_type="test")],
        )


def scenario_system(names, *, failures=None, search_provider=None, synth=None, barrier=None, timeout=0.2):
    registry = AgentRegistry()
    agents = {}
    failures = failures or {}
    for name in names:
        agent = ScenarioAgent(name, error=failures.get(name), barrier=barrier)
        registry.register(agent)
        agents[name] = agent
    web = WebSearchService(
        search_provider,
        enabled=search_provider is not None,
        timeout_seconds=1,
        max_results=5,
    )
    router = FixedRouter(tuple(names))
    synthesis = synth or CapturingSynthesizer()
    return CityOrchestratorAgent(
        registry, router, synthesizer=synthesis, web_search=web,
        execution_timeout_seconds=timeout,
    ), agents, synthesis


@pytest.mark.asyncio
async def test_three_agent_execution_is_concurrent_and_context_copies_are_isolated():
    barrier = {"started": set(), "all_started": asyncio.Event()}
    orchestrator, agents, synthesis = scenario_system(NAMES, barrier=barrier)
    request = AgentRequest(query=QUERY, context={"shared": {"value": "original"}})
    response = await orchestrator.execute(request)

    assert barrier["started"] == set(NAMES)
    assert response.metadata["execution_status"] == "complete"
    assert all(agent.calls[0].query == request.query for agent in agents.values())
    assert all(agent.calls[0].request_id == request.request_id for agent in agents.values())
    assert len(synthesis.calls) == 1
    mobility_context = agents["mobility"].calls[0].context
    environment_context = agents["environment"].calls[0].context
    public_context = agents["public_services"].calls[0].context
    environment_context["location"] = "test-only location"
    assert "location" not in mobility_context and "location" not in public_context
    assert "nlp" in mobility_context and "nlp" in environment_context and "nlp" in public_context


@pytest.mark.asyncio
async def test_three_real_specialists_all_start_before_any_continues(monkeypatch):
    orchestrator, registry, _captured, _provider, _router, _synthesis = make_real_system(
        NAMES, monkeypatch
    )
    started: set[str] = set()
    all_started = asyncio.Event()
    for name in NAMES:
        agent = registry.get(name)
        original = agent.execute

        async def gated(_self, request, *, _name=name, _original=original):
            started.add(_name)
            if len(started) == len(NAMES):
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return await _original(request)

        agent.execute = MethodType(gated, agent)

    response = await orchestrator.execute(AgentRequest(
        query=QUERY,
        context={"user_context": {"latitude": 6.9271, "longitude": 79.8612}},
    ))
    assert started == set(NAMES)
    assert response.metadata["execution_status"] == "complete"


@pytest.mark.asyncio
async def test_three_agent_partial_and_total_failure_contracts():
    partial, agents, synthesis = scenario_system(
        NAMES,
        failures={"environment": AgentErrorCode.TIMEOUT},
    )
    request = AgentRequest(query=QUERY)
    response = await partial.execute(request)
    summary = synthesis.calls[0][1]
    assert response.metadata["execution_status"] == "partial_success"
    assert response.metadata["successful_agents"] == ["mobility", "public_services"]
    assert response.metadata["failed_agents"] == ["environment"]
    assert {source.name for source in response.sources} == {
        "mobility test source", "public_services test source"
    }
    assert [x.value for x in summary.successful_agents] == ["mobility", "public_services"]
    assert "environment" in response.metadata["synthesis_limitations"]
    assert "Unavailable information" in response.answer

    two_failure, _, total_synth = scenario_system(
        NAMES,
        failures={
            "environment": AgentErrorCode.TIMEOUT,
            "public_services": AgentErrorCode.AGENT_EXECUTION_FAILED,
        },
    )
    partial2 = await two_failure.execute(AgentRequest(query=QUERY))
    assert partial2.metadata["execution_status"] == "partial_success"
    assert partial2.metadata["failed_agents"] == ["environment", "public_services"]
    assert [source.name for source in partial2.sources] == ["mobility test source"]

    total, _, no_synth = scenario_system(
        NAMES,
        failures={
            "mobility": AgentErrorCode.TIMEOUT,
            "environment": AgentErrorCode.AGENT_EXECUTION_FAILED,
            "public_services": AgentErrorCode.TIMEOUT,
        },
    )
    failed = await total.execute(AgentRequest(query=QUERY))
    assert failed.metadata["execution_status"] == "failed"
    assert failed.success is False
    assert failed.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert failed.sources == []
    assert no_synth.calls == []


class OfflineSearchProvider:
    provider_name = "tavily"

    def __init__(self, *, failure=None):
        self.failure = failure
        self.calls = []

    async def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self.failure:
            raise self.failure
        return [WebSearchResult(
            title="Local provider fixture",
            url="https://example.org/environment",
            snippet="Provider returned environmental evidence.",
            provider=self.provider_name,
        )]


@pytest.mark.asyncio
async def test_total_and_partial_failure_tavily_policy_runs_at_most_once():
    provider = OfflineSearchProvider()
    total, _, total_synth = scenario_system(
        ("mobility", "environment"),
        failures={"mobility": AgentErrorCode.TIMEOUT, "environment": AgentErrorCode.AGENT_EXECUTION_FAILED},
        search_provider=provider,
    )
    failed = await total.execute(AgentRequest(query="traffic and air quality"))
    assert len(provider.calls) == 1
    assert failed.metadata["web_search_used"] is True
    assert failed.metadata["web_search_status"] == "success"
    assert failed.metadata["execution_status"] == "failed"
    assert failed.metadata["answer_basis"] == "web_fallback"
    assert failed.success is True
    assert len(failed.sources) == 1 and failed.sources[0].source_type == "web_search"
    assert total_synth.calls[0][1].successful_agents == []

    provider2 = OfflineSearchProvider()
    partial, _, partial_synth = scenario_system(
        ("mobility", "environment", "public_services"),
        failures={"environment": AgentErrorCode.TIMEOUT},
        search_provider=provider2,
    )
    response = await partial.execute(AgentRequest(query=QUERY))
    assert len(provider2.calls) == 1
    assert response.metadata["execution_status"] == "partial_success"
    assert response.metadata["answer_basis"] == "specialist_plus_web"
    assert response.metadata["failed_agents"] == ["environment"]
    assert {source.source_type for source in response.sources} == {"test", "web_search"}
    assert partial_synth.calls[0][1].failed_agents == [SpecialistAgentName.ENVIRONMENT]


@pytest.mark.asyncio
async def test_tavily_failure_preserves_partial_evidence_and_error_codes_control_eligibility():
    provider = OfflineSearchProvider(failure=WebSearchProviderError("connection"))
    orchestrator, _, synthesis = scenario_system(
        ("mobility", "environment"),
        failures={"environment": AgentErrorCode.TIMEOUT},
        search_provider=provider,
    )
    response = await orchestrator.execute(AgentRequest(query="traffic and air quality"))
    assert response.metadata["execution_status"] == "partial_success"
    assert response.metadata["web_search_status"] == "failed"
    assert response.metadata["web_search_used"] is False
    assert [source.name for source in response.sources] == ["mobility test source"]
    assert len(provider.calls) == 1
    assert len(synthesis.calls) == 1

    ineligible = OfflineSearchProvider()
    no_search, _, _ = scenario_system(
        ("mobility",), failures={"mobility": AgentErrorCode.UNSUPPORTED_REQUEST},
        search_provider=ineligible,
    )
    unsupported = await no_search.execute(AgentRequest(query="unsupported mobility location"))
    assert ineligible.calls == []
    assert unsupported.metadata["web_search_status"] == "not_needed"


@pytest.mark.asyncio
async def test_environment_ambiguous_geocoding_is_specialist_failure_with_partial_success(monkeypatch):
    class ColomboMatches(MockOpenMeteo):
        async def get(self, url, *, params):
            self.calls.append((url, params))
            if url == GEOCODING_API_URL:
                return MockResponse({"results": [
                    {"name": "Colombo", "country": "Sri Lanka", "latitude": 6.9, "longitude": 79.8},
                    {"name": "Colombo", "country": "Another Country", "latitude": 7.0, "longitude": 80.0},
                ]})
            return self.responses[url]

    provider = ColomboMatches(open_meteo_responses())
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: provider)
    registry = create_agent_registry()
    router = FixedRouter(("mobility", "environment"))
    orchestrator = CityOrchestratorAgent(
        registry, router, synthesizer=CapturingSynthesizer(),
        web_search=WebSearchService(None, enabled=False),
    )
    response = await orchestrator.execute(AgentRequest(
        query="How is traffic near Colombo Fort and what is the weather in Colombo?",
        context={"user_context": {"location": "Colombo"}},
    ))
    result = response.metadata["execution_summary"]["results"][1]
    assert response.metadata["execution_status"] == "partial_success"
    assert result["success"] is False
    assert result["error"]["code"] == AgentErrorCode.INVALID_REQUEST.value
    assert response.metadata["successful_agents"] == ["mobility"]
    assert response.sources and all(source.source_type == "structured_dataset" for source in response.sources)


@pytest.mark.asyncio
async def test_mobility_unsupported_location_current_specialist_behavior():
    registry = create_agent_registry()
    router = FixedRouter(("mobility",), locations=[])
    orchestrator = CityOrchestratorAgent(
        registry, router, synthesizer=CapturingSynthesizer(),
        web_search=WebSearchService(None, enabled=False),
    )
    response = await orchestrator.execute(AgentRequest(query="How is traffic in Imaginaryville?"))
    assert response.success is True
    assert response.metadata["execution_status"] == "complete"
    assert response.metadata["successful_agents"] == ["mobility"]
    assert response.sources
    # Current IR falls back to category-wide traffic data on an unmatched place.
    assert any("Colombo" in str(source.metadata) or "Road" in str(source.metadata) for source in response.sources)


@pytest.mark.asyncio
async def test_public_services_unsupported_category_is_structured_and_not_web_eligible():
    provider = OfflineSearchProvider()
    registry = create_agent_registry()
    orchestrator = CityOrchestratorAgent(
        registry,
        FixedRouter(("public_services",)),
        synthesizer=CapturingSynthesizer(),
        web_search=WebSearchService(provider, enabled=True),
    )
    response = await orchestrator.execute(AgentRequest(
        query="Where is the nearest library in Colombo?"
    ))
    assert response.success is False
    result = response.metadata["execution_summary"]["results"][0]
    assert result["error"]["code"] == AgentErrorCode.UNSUPPORTED_REQUEST.value
    assert response.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert response.sources == []
    assert response.metadata["execution_status"] == "failed"
    assert provider.calls == []
    assert response.metadata["web_search_status"] == "not_needed"


@pytest.mark.asyncio
async def test_explicit_all_domain_local_routing_does_not_call_gemini():
    class ColomboEntities:
        def extract_locations(self, text):
            return ["Colombo"] if "colombo" in text.casefold() else []

    class ForbiddenGemini:
        async def understand(self, *_args, **_kwargs):
            raise AssertionError("clear local three-domain request must not use Gemini")

    settings = Settings(_env_file=None, gemini_api_key=None, nlp_gemini_fallback_enabled=True)
    pipeline = RequestUnderstandingPipeline(
        analyzer=LocalNlpAnalyzer(entity_extractor=ColomboEntities()),
        gemini_router=ForbiddenGemini(),
        settings=settings,
    )
    result = await pipeline.route("Check traffic, air quality, and hospitals in Colombo.")
    assert [name.value for name in result.decision.agent_names] == list(NAMES)
    assert result.routing_method == "local_nlp"
    assert result.gemini_understanding_fallback_used is False


@pytest.mark.asyncio
async def test_gemini_entity_fallback_and_clarification_contract_offline():
    class NoLocations:
        def extract_locations(self, _text):
            return []

    class FakeGemini:
        def __init__(self, decision):
            self.decision = decision
            self.calls = []

        async def understand(self, query, *, local_analysis=None, request_id=None):
            self.calls.append((query, local_analysis, request_id))
            return self.decision

    fallback_decision = UnderstandingDecision(
        agent_names=["mobility"], locations=["Marine Drive"], temporal_expressions=[],
        confidence=0.96, reason="Named traffic location", needs_clarification=False,
        missing_information=[],
    )
    gemini = FakeGemini(fallback_decision)
    settings = Settings(_env_file=None, gemini_api_key="fake-key")
    pipeline = RequestUnderstandingPipeline(
        analyzer=LocalNlpAnalyzer(entity_extractor=NoLocations()),
        gemini_router=gemini,
        settings=settings,
    )
    result = await pipeline.route("Is there traffic congestion on Marine Drive?")
    assert len(gemini.calls) == 1
    assert result.gemini_understanding_fallback_used is True
    assert result.decision.agent_names == [SpecialistAgentName.MOBILITY]
    assert result.locations == ["Marine Drive"]

    registry = create_agent_registry()
    live_orchestrator = CityOrchestratorAgent(
        registry, pipeline, synthesizer=CapturingSynthesizer(),
        web_search=WebSearchService(None, enabled=False),
    )
    routed = await live_orchestrator.execute(
        AgentRequest(query="Is there traffic congestion on Marine Drive?")
    )
    assert routed.success is True
    assert routed.metadata["selected_agents"] == ["mobility"]
    assert routed.metadata["execution_status"] == "complete"
    assert routed.sources

    clarification = UnderstandingDecision(
        agent_names=[], locations=[], temporal_expressions=[], confidence=1.0,
        reason="Location is missing", needs_clarification=True,
        missing_information=["location"],
    )
    clarify_gemini = FakeGemini(clarification)
    clarify_pipeline = RequestUnderstandingPipeline(
        analyzer=LocalNlpAnalyzer(entity_extractor=NoLocations()),
        gemini_router=clarify_gemini,
        settings=settings,
    )
    registry = create_agent_registry()
    orchestrator = CityOrchestratorAgent(
        registry, clarify_pipeline, synthesizer=CapturingSynthesizer(),
        web_search=WebSearchService(None, enabled=False),
    )
    response = await orchestrator.execute(AgentRequest(query="How is the traffic there?"))
    assert response.error.code == AgentErrorCode.NEEDS_CLARIFICATION
    assert clarify_gemini.calls and response.metadata["gemini_understanding_fallback_used"] is True
    assert response.metadata.get("selected_agents", []) == []


@pytest.mark.asyncio
async def test_gemini_unavailable_uses_deterministic_route_or_safe_clarification():
    class FailedGemini:
        async def understand(self, *_args, **_kwargs):
            raise GeminiRoutingError("ConnectError")

    class NoLocations:
        def extract_locations(self, _text):
            return []

    settings = Settings(_env_file=None, gemini_api_key="fake-key")
    pipeline = RequestUnderstandingPipeline(
        analyzer=LocalNlpAnalyzer(entity_extractor=NoLocations()),
        gemini_router=FailedGemini(), settings=settings,
    )
    clear = await pipeline.route("Show traffic information in Colombo.")
    ambiguous = await pipeline.route("How is the traffic there?")
    assert [x.value for x in clear.decision.agent_names] == ["mobility"]
    assert clear.understanding_method == "deterministic_fallback"
    assert ambiguous.decision.needs_clarification is True
    assert ambiguous.decision.agent_names == []


@pytest.mark.asyncio
async def test_fake_gemini_can_route_all_three_and_execute_registered_agents(monkeypatch):
    class NoLocations:
        def extract_locations(self, _text):
            return []

    class FakeGemini:
        def __init__(self):
            self.calls = []

        async def understand(self, query, *, local_analysis=None, request_id=None):
            self.calls.append((query, local_analysis, request_id))
            return UnderstandingDecision(
                agent_names=list(NAMES), locations=["Colombo"],
                temporal_expressions=["today"], confidence=0.98,
                reason="The request covers all three city domains.",
                needs_clarification=False, missing_information=[],
            )

    mock_payloads = open_meteo_responses()
    mock_payloads[GEOCODING_API_URL] = MockResponse({"results": [
        {
            "name": "Colombo", "country": "Sri Lanka",
            "latitude": 6.9271, "longitude": 79.8612,
            "timezone": "Asia/Colombo",
        },
    ]})
    provider = MockOpenMeteo(mock_payloads)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: provider)
    gemini = FakeGemini()
    pipeline = RequestUnderstandingPipeline(
        analyzer=LocalNlpAnalyzer(entity_extractor=NoLocations()),
        gemini_router=gemini,
        settings=Settings(_env_file=None, gemini_api_key="fake-key"),
    )
    orchestrator = CityOrchestratorAgent(
        create_agent_registry(), pipeline, synthesizer=CapturingSynthesizer(),
        web_search=WebSearchService(None, enabled=False),
    )
    query = "Traffic conditions and air quality, plus a hospital in Colombo today."
    response = await orchestrator.execute(AgentRequest(
        query=query,
        context={},
    ))
    assert len(gemini.calls) == 1
    assert response.metadata["gemini_understanding_fallback_used"] is True
    assert response.metadata["selected_agents"] == list(NAMES)
    assert response.metadata["execution_status"] == "complete"
    assert response.metadata["successful_agents"] == list(NAMES)
    assert len(response.sources) >= 4
