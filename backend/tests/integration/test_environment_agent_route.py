from datetime import datetime, time, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from fastapi.testclient import TestClient

import pytest

from app.agents.contracts import AgentRequest
from app.agents.environment import AIR_QUALITY_API_URL, FORECAST_API_URL, GEOCODING_API_URL, EnvironmentAgent
from app.agents.mobility.agent import MobilityAgent
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.router import DeterministicQueryRouter
from app.agents.orchestrator.synthesizer import DeterministicResultSynthesizer
from app.agents.orchestrator.web_search import WebSearchService
from app.agents.registry import AgentRegistry
from app.core.config import Settings
from app.nlp.pipeline import LocalNlpAnalyzer, RequestUnderstandingPipeline
from app.main import app


class MockResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class MockProviderClient:
    def __init__(self, responses: dict[str, MockResponse | Exception], *, timeout: float) -> None:
        self.responses = responses
        self.timeout = timeout
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def get(self, url: str, *, params: dict) -> MockResponse:
        self.calls.append((url, params))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def provider_responses() -> dict[str, MockResponse]:
    today = datetime.now(ZoneInfo("Asia/Colombo")).date()
    tomorrow = today + timedelta(days=1)
    return {
        GEOCODING_API_URL: MockResponse(
            {
                "results": [
                    {
                        "name": "Colombo",
                        "country": "Sri Lanka",
                        "timezone": "Asia/Colombo",
                        "latitude": 6.9271,
                        "longitude": 79.8612,
                    }
                ]
            }
        ),
        FORECAST_API_URL: MockResponse(
            {
                "timezone": "Asia/Colombo",
                "hourly": {
                    "time": [
                        datetime.combine(today, time(23, 0)).isoformat(timespec="minutes"),
                        datetime.combine(tomorrow, time(12, 0)).isoformat(timespec="minutes"),
                    ],
                    "temperature_2m": [29.5, 30.0],
                    "precipitation_probability": [20, 45],
                    "weather_code": [2, 61],
                },
            }
        ),
        AIR_QUALITY_API_URL: MockResponse(
            {
                "timezone": "Asia/Colombo",
                "hourly": {
                    "time": [
                        datetime.combine(today, time(12, 0)).isoformat(timespec="minutes"),
                        datetime.combine(today, time(13, 0)).isoformat(timespec="minutes"),
                    ],
                    "pm2_5": [12.0, 13.5],
                    "us_aqi": [48, 52],
                    "european_aqi": [19, 21],
                },
            }
        ),
    }


def test_chat_api_runs_registered_environment_agent(monkeypatch) -> None:
    class ColomboEntityExtractor:
        def extract_locations(self, text: str) -> list[str]:
            return ["Colombo"] if "colombo" in text.casefold() else []

    class OfflineGemini:
        async def understand(self, query, *, local_analysis=None, request_id=None):
            raise AssertionError("local intent and location are complete")

    monkeypatch.setattr(
        app.state.orchestrator,
        "_router",
        RequestUnderstandingPipeline(
            analyzer=LocalNlpAnalyzer(entity_extractor=ColomboEntityExtractor()),
            gemini_router=OfflineGemini(),
            settings=Settings(_env_file=None, gemini_api_key=None),
        ),
    )
    mock_client = MockProviderClient(provider_responses(), timeout=0)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: mock_client)

    with TestClient(app) as client:
        registered = client.app.state.agent_registry.get("environment")
        assert isinstance(registered, EnvironmentAgent)
        assert isinstance(client.app.state.orchestrator, CityOrchestratorAgent)

        response = client.post(
            "/api/v1/chat",
            json={
                "message": "What is the weather and air quality in Colombo?",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["metadata"]["selected_agents"] == ["environment"]
    assert body["metadata"]["execution_status"] == "complete"
    assert "29.5" in body["answer"]
    assert "PM2.5 12.0" in body["answer"]
    assert {source["name"] for source in body["sources"]} == {
        "Open-Meteo Geocoding API",
        "Open-Meteo Forecast API",
        "Open-Meteo Air Quality API",
    }
    assert [url for url, _ in mock_client.calls] == [
        GEOCODING_API_URL,
        FORECAST_API_URL,
        AIR_QUALITY_API_URL,
    ]
    for url, params in mock_client.calls[1:]:
        assert params["latitude"] == 6.9271
        assert params["longitude"] == 79.8612


@pytest.mark.asyncio
async def test_real_mobility_and_environment_execute_concurrently_with_adapted_context(monkeypatch) -> None:
    class ColomboEntityExtractor:
        def extract_locations(self, text: str) -> list[str]:
            return ["Colombo"] if "colombo" in text.casefold() else []

    class TodayExtractor:
        def extract(self, text: str) -> list[str]:
            return ["today"] if "today" in text.casefold() else []

    class NoGemini:
        async def understand(self, query, *, local_analysis=None, request_id=None):
            raise AssertionError("the sufficiently complete local request should not call Gemini")

    class NoLiveSynthesis:
        async def synthesize(self, query, summary, web_evidence=None):
            raise RuntimeError("live synthesis is disabled in this test")

    class CapturingMobility(MobilityAgent):
        received = None

        async def execute(self, request):
            self.received = request
            return await super().execute(request)

    class CapturingEnvironment(EnvironmentAgent):
        received = None

        async def execute(self, request):
            self.received = request
            return await super().execute(request)

    provider = MockProviderClient(provider_responses(), timeout=0)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: provider)

    mobility = CapturingMobility()
    environment = CapturingEnvironment()
    registry = AgentRegistry()
    registry.register(mobility)
    registry.register(environment)
    settings = Settings(_env_file=None, gemini_api_key=None, nlp_gemini_fallback_enabled=False)
    router = RequestUnderstandingPipeline(
        analyzer=LocalNlpAnalyzer(
            entity_extractor=ColomboEntityExtractor(),
            temporal_extractor=TodayExtractor(),
        ),
        gemini_router=NoGemini(),
        settings=settings,
    )
    orchestrator = CityOrchestratorAgent(
        registry,
        router,
        synthesizer=NoLiveSynthesis(),
        web_search=WebSearchService(None, enabled=False),
    )
    request = AgentRequest(
        request_id=uuid4(),
        query="How are traffic and air quality in Colombo today?",
    )

    response = await orchestrator.execute(request)

    assert response.success is True
    assert response.request_id == request.request_id
    assert response.metadata["selected_agents"] == ["mobility", "environment"]
    assert response.metadata["execution_status"] == "complete"
    assert response.metadata["synthesis_method"] == "deterministic_fallback"
    assert mobility.received is not None and environment.received is not None
    assert mobility.received.request_id == environment.received.request_id == request.request_id
    assert mobility.received.query == environment.received.query == request.query
    assert mobility.received.context["nlp"]["locations"] == ["Colombo"]
    assert "location" not in mobility.received.context
    assert environment.received.context["location"] == "Colombo"
    assert environment.received.context["nlp"]["locations"] == ["Colombo"]
    assert {source.name for source in response.sources} >= {
        "Open-Meteo Geocoding API",
        "Open-Meteo Air Quality API",
    }
    assert response.metadata["successful_agents"] == ["mobility", "environment"]
    assert response.metadata["synthesis_used_agents"] == ["mobility", "environment"]
    assert {url for url, _ in provider.calls} == {
        GEOCODING_API_URL,
        AIR_QUALITY_API_URL,
    }


def test_chat_api_returns_structured_geocoding_error(monkeypatch) -> None:
    monkeypatch.setattr(
        app.state.orchestrator, "_router", DeterministicQueryRouter()
    )
    mock_client = MockProviderClient(
        {GEOCODING_API_URL: httpx.ReadTimeout("private timeout detail")}, timeout=0
    )
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: mock_client)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "Weather in Colombo", "context": {"location": "Colombo"}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "timeout"
    assert "private timeout detail" not in response.text
    assert body["metadata"]["failed_agents"] == ["environment"]


def test_chat_api_validates_request() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"context": {"location": "Colombo"}})

    assert response.status_code == 422
