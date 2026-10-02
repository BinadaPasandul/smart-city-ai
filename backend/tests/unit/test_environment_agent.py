from datetime import date, datetime, time, timedelta, timezone

import httpx
import pytest

from app.agents.contracts import AgentErrorCode, AgentRequest
from app.agents.environment import AIR_QUALITY_API_URL, FORECAST_API_URL, GEOCODING_API_URL, EnvironmentAgent


def make_payload() -> dict:
    today = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=1)
    return {
        "timezone": "UTC",
        "hourly": {
            "time": [
                datetime.combine(today, time(23, 0)).isoformat(timespec="minutes"),
                datetime.combine(tomorrow, time(12, 0)).isoformat(timespec="minutes"),
            ],
            "temperature_2m": [24.5, 26.0],
            "precipitation_probability": [10, 35],
            "weather_code": [1, 61],
        },
    }


class FakeResponse:
    def __init__(self, payload: object, *, status_code: int | None = None, invalid_json: bool = False) -> None:
        self.payload = payload
        self.status_code = status_code
        self.invalid_json = invalid_json

    def raise_for_status(self) -> None:
        if self.status_code is not None and self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("provider error", request=request, response=response)

    def json(self) -> object:
        if self.invalid_json:
            raise ValueError("invalid JSON")
        return self.payload


class FakeClient:
    def __init__(self, response: FakeResponse | Exception | dict[str, FakeResponse | Exception], *, timeout: float) -> None:
        self.response = response
        self.timeout = timeout
        self.params = None
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def get(self, url: str, *, params: dict) -> FakeResponse:
        self.params = params
        self.calls.append((url, params))
        selected = self.response.get(url) if isinstance(self.response, dict) else self.response
        if isinstance(selected, Exception):
            raise selected
        return selected


def request(*, query: str = "Weather and air quality forecast", **context) -> AgentRequest:
    return AgentRequest(query=query, context=context)


def make_air_quality_payload() -> dict:
    return {
        "timezone": "UTC",
        "hourly": {
            "time": ["2026-09-29T12:00", "2026-09-29T13:00"],
            "pm2_5": [12.4, 14.1],
            "us_aqi": [53, 61],
            "european_aqi": [22, 26],
        },
        "hourly_units": {
            "pm2_5": "μg/m³",
            "us_aqi": "USAQI",
            "european_aqi": "European AQI",
        },
    }


def make_geocoding_payload(*, timezone_name: str | None = "Asia/Colombo", name: str = "Colombo") -> dict:
    location = {"name": name, "country": "Sri Lanka", "latitude": 6.9271, "longitude": 79.8612}
    if timezone_name is not None:
        location["timezone"] = timezone_name
    return {"results": [location]}


@pytest.mark.asyncio
async def test_retrieves_today_and_tomorrow_forecasts(monkeypatch) -> None:
    fake_client = FakeClient(FakeResponse(make_payload()), timeout=0)
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: fake_client,
    )
    agent = EnvironmentAgent()
    result = await agent.execute(request(latitude=6.9271, longitude=79.8612, location="Ignored City"))

    assert agent.name == "environment"
    assert result.success is True
    assert result.agent_name == "environment"
    assert result.metadata["timezone"] == "UTC"
    forecasts = result.metadata["forecasts"]
    assert len(forecasts) == 2
    today = datetime.now(timezone.utc).date()
    assert forecasts[0]["date"] == today.isoformat()
    assert forecasts[0]["temperature_c"] == 24.5
    assert forecasts[0]["precipitation_probability_percent"] == 10
    assert forecasts[0]["weather_code"] == 1
    assert forecasts[1]["date"] == (today + timedelta(days=1)).isoformat()
    assert "Open-Meteo Forecast API" == result.sources[0].name
    assert [url for url, _ in fake_client.calls] == [FORECAST_API_URL, AIR_QUALITY_API_URL]


@pytest.mark.asyncio
async def test_successfully_resolves_city_and_uses_result_for_both_forecasts(monkeypatch) -> None:
    responses = {
        GEOCODING_API_URL: FakeResponse(make_geocoding_payload()),
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse(make_air_quality_payload()),
    }
    fake_client = FakeClient(responses, timeout=0)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: fake_client)
    result = await EnvironmentAgent().execute(request(location="Colombo"))

    assert result.success is True
    assert result.metadata["location"] == {
        "name": "Colombo",
        "country": "Sri Lanka",
        "timezone": "Asia/Colombo",
        "latitude": 6.9271,
        "longitude": 79.8612,
    }
    requests_by_url = {url: params for url, params in fake_client.calls}
    # "Colombo" is a bare known Sri Lankan city name, so it's biased toward
    # Sri Lanka in the geocoding request itself -- see _augment_with_known_country.
    assert requests_by_url[GEOCODING_API_URL]["name"] == "Colombo, Sri Lanka"
    for url in (FORECAST_API_URL, AIR_QUALITY_API_URL):
        assert requests_by_url[url]["latitude"] == 6.9271
        assert requests_by_url[url]["longitude"] == 79.8612
    assert GEOCODING_API_URL == result.sources[0].url


@pytest.mark.asyncio
async def test_city_resolution_without_timezone_still_returns_forecasts(monkeypatch) -> None:
    responses = {
        GEOCODING_API_URL: FakeResponse(make_geocoding_payload(timezone_name=None)),
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse(make_air_quality_payload()),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(location="Colombo"))

    assert result.success is True
    assert result.metadata["location"]["timezone"] is None
    assert result.metadata["weather"]["forecasts"]
    assert result.metadata["air_quality"]["forecasts"]


@pytest.mark.asyncio
async def test_city_with_no_geocoding_results_returns_not_found(monkeypatch) -> None:
    fake_client = FakeClient({GEOCODING_API_URL: FakeResponse({"results": []})}, timeout=0)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: fake_client)
    result = await EnvironmentAgent().execute(request(location="Unknownville"))

    assert result.success is False
    assert result.error.code == AgentErrorCode.INVALID_REQUEST
    assert "No city matched" in result.error.message
    assert [url for url, _ in fake_client.calls] == [GEOCODING_API_URL]


@pytest.mark.asyncio
async def test_ambiguous_exact_city_matches_request_clarification(monkeypatch) -> None:
    ambiguous = {
        "results": [
            {"name": "Springfield", "country": "United States", "latitude": 39.78, "longitude": -89.64},
            {"name": "Springfield", "country": "Canada", "latitude": 45.61, "longitude": -62.0},
        ]
    }
    fake_client = FakeClient({GEOCODING_API_URL: FakeResponse(ambiguous)}, timeout=0)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: fake_client)
    result = await EnvironmentAgent().execute(request(location="Springfield"))

    assert result.success is False
    assert result.error.code == AgentErrorCode.INVALID_REQUEST
    assert "ambiguous" in result.error.message
    assert "United States" in result.error.message
    assert "Canada" in result.error.message
    assert [url for url, _ in fake_client.calls] == [GEOCODING_API_URL]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (httpx.ReadTimeout("timeout"), AgentErrorCode.TIMEOUT),
        (FakeResponse({}, status_code=503), AgentErrorCode.AGENT_EXECUTION_FAILED),
    ],
)
async def test_geocoding_timeout_or_http_error_is_structured(monkeypatch, response, expected_code) -> None:
    fake_client = FakeClient({GEOCODING_API_URL: response}, timeout=0)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: fake_client)
    result = await EnvironmentAgent().execute(request(location="Colombo"))

    assert result.success is False
    assert result.error.code == expected_code
    assert "timeout" not in result.error.message.lower() or expected_code == AgentErrorCode.TIMEOUT
    assert [url for url, _ in fake_client.calls] == [GEOCODING_API_URL]


@pytest.mark.asyncio
async def test_invalid_geocoding_response_returns_structured_error(monkeypatch) -> None:
    fake_client = FakeClient({GEOCODING_API_URL: FakeResponse({"unexpected": []})}, timeout=0)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: fake_client)
    result = await EnvironmentAgent().execute(request(location="Colombo"))

    assert result.success is False
    assert result.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert "invalid response" in result.error.message
    assert [url for url, _ in fake_client.calls] == [GEOCODING_API_URL]


@pytest.mark.asyncio
async def test_successful_air_quality_retrieval_includes_values_units_and_scale_labels(monkeypatch) -> None:
    responses = {
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse(make_air_quality_payload()),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9271, longitude=79.8612))

    assert result.success is True
    air = result.metadata["air_quality"]
    assert air["forecasts"][0] == {
        "time": "2026-09-29T12:00",
        "pm2_5_ug_m3": 12.4,
        "us_aqi": 53,
        "european_aqi": 22,
    }
    assert air["units"] == {
        "pm2_5": "μg/m³",
        "us_aqi": "US AQI index",
        "european_aqi": "European AQI index",
    }
    assert air["aqi_scales"] == {"us_aqi": "United States AQI", "european_aqi": "European AQI"}
    assert air["data_kind"] == "model_forecast"
    assert air["observations"] is False
    assert "PM2.5 12.4 μg/m³" in result.answer
    assert "US AQI 53" in result.answer
    assert "European AQI 22" in result.answer
    air_source = next(source for source in result.sources if source.url == AIR_QUALITY_API_URL)
    assert air_source.retrieved_at is not None
    assert air_source.metadata["observations"] is False


@pytest.mark.asyncio
async def test_missing_air_quality_fields_are_reported_as_unavailable(monkeypatch) -> None:
    payload = make_air_quality_payload()
    payload["hourly"].pop("us_aqi")
    payload["hourly"].pop("european_aqi")
    responses = {
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse(payload),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9, longitude=79.8))

    assert result.success is True
    forecast = result.metadata["air_quality"]["forecasts"][0]
    assert forecast["pm2_5_ug_m3"] == 12.4
    assert forecast["us_aqi"] is None
    assert forecast["european_aqi"] is None


@pytest.mark.asyncio
async def test_air_quality_timeout_preserves_weather_results(monkeypatch) -> None:
    responses = {
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: httpx.ReadTimeout("private timeout details"),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9, longitude=79.8))

    assert result.success is True
    assert "weather" in result.metadata
    assert "air_quality" not in result.metadata
    assert result.metadata["partial_errors"] == [
        {"code": AgentErrorCode.TIMEOUT.value, "message": "Air quality forecast request timed out."}
    ]
    assert "private timeout details" not in str(result.metadata)


@pytest.mark.asyncio
async def test_air_quality_http_error_preserves_weather_results(monkeypatch) -> None:
    responses = {
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse({}, status_code=429),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9, longitude=79.8))

    assert result.success is True
    assert "weather" in result.metadata
    assert result.metadata["partial_errors"][0]["message"] == "Air quality forecast provider rate limit reached."


@pytest.mark.asyncio
async def test_invalid_air_quality_response_preserves_weather_results(monkeypatch) -> None:
    responses = {
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse({"unexpected": "shape"}),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9, longitude=79.8))

    assert result.success is True
    assert "weather" in result.metadata
    assert result.metadata["partial_errors"][0]["message"] == "Air quality provider returned an invalid response."


@pytest.mark.asyncio
async def test_invalid_air_quality_json_preserves_weather_results(monkeypatch) -> None:
    responses = {
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse({}, invalid_json=True),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9, longitude=79.8))

    assert result.success is True
    assert "weather" in result.metadata
    assert result.metadata["partial_errors"][0]["message"] == "Air quality forecast provider returned invalid JSON."


@pytest.mark.asyncio
async def test_missing_coordinates_returns_invalid_request(monkeypatch) -> None:
    def unexpected_client(**kwargs):
        pytest.fail("Provider must not be called without coordinates")

    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", unexpected_client)
    result = await EnvironmentAgent().execute(AgentRequest(query="Weather"))

    assert result.success is False
    assert result.error.code == AgentErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_invalid_coordinates_return_invalid_request(monkeypatch) -> None:
    def unexpected_client(**kwargs):
        pytest.fail("Provider must not be called with invalid coordinates")

    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", unexpected_client)
    result = await EnvironmentAgent().execute(request(latitude=91, longitude=79))

    assert result.success is False
    assert result.error.code == AgentErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_api_timeout_returns_timeout_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(httpx.ReadTimeout("timed out"), timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9, longitude=79.8))

    assert result.success is False
    assert result.error.code == AgentErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_invalid_provider_response_returns_execution_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(FakeResponse({"timezone": "UTC", "hourly": {}}), timeout=timeout),
    )
    result = await EnvironmentAgent().execute(request(latitude=6.9, longitude=79.8))

    assert result.success is False
    assert result.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_urls", "weather_expected", "air_quality_expected"),
    [
        ("weather today", [FORECAST_API_URL], True, False),
        ("air quality today", [AIR_QUALITY_API_URL], False, True),
        ("weather and air quality", [FORECAST_API_URL, AIR_QUALITY_API_URL], True, True),
        # Unknown environmental wording retains legacy retrieval of both feeds.
        ("environmental conditions in Colombo", [FORECAST_API_URL, AIR_QUALITY_API_URL], True, True),
    ],
)
async def test_query_intent_selects_environment_retrieval(
    monkeypatch, query, expected_urls, weather_expected, air_quality_expected
) -> None:
    responses = {
        FORECAST_API_URL: FakeResponse(make_payload()),
        AIR_QUALITY_API_URL: FakeResponse(make_air_quality_payload()),
    }
    fake_client = FakeClient(responses, timeout=0)
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: fake_client,
    )

    result = await EnvironmentAgent().execute(request(
        latitude=6.9, longitude=79.8, query=query,
    ))

    assert result.success is True
    assert [url for url, _ in fake_client.calls] == expected_urls
    assert ("weather" in result.metadata) is weather_expected
    assert ("air_quality" in result.metadata) is air_quality_expected
    assert all(params["latitude"] == 6.9 and params["longitude"] == 79.8 for _, params in fake_client.calls)
