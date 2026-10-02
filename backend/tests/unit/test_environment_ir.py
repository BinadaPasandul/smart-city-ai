from datetime import datetime, time, timedelta, timezone

import httpx
import pytest

from app.agents.contracts import AgentErrorCode
from app.ir.environment_ir import (
    AIR_QUALITY_API_URL,
    FORECAST_API_URL,
    GEOCODING_API_URL,
    EnvironmentIR,
)


def weather_payload():
    today = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=1)
    return {
        "timezone": "UTC",
        "hourly": {
            "time": [
                datetime.combine(today, time(23)).isoformat(timespec="minutes"),
                datetime.combine(tomorrow, time(12)).isoformat(timespec="minutes"),
            ],
            "temperature_2m": [24.5, 26.0],
            "precipitation_probability": [10, 35],
            "weather_code": [1, 61],
        },
    }


def air_payload():
    return {
        "timezone": "UTC",
        "hourly": {
            "time": ["2026-09-29T12:00"],
            "pm2_5": [12.4],
            "us_aqi": [53],
            "european_aqi": [22],
        },
    }


class FakeResponse:
    def __init__(self, payload, *, status_code=None, invalid_json=False):
        self.payload = payload
        self.status_code = status_code
        self.invalid_json = invalid_json

    def raise_for_status(self):
        if self.status_code is not None and self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            raise httpx.HTTPStatusError(
                "provider failure", request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self):
        if self.invalid_json:
            raise ValueError("bad JSON")
        return self.payload


class FakeClient:
    def __init__(self, responses, *, timeout):
        self.responses = responses
        self.timeout = timeout
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, *, params):
        self.calls.append((url, params))
        selected = self.responses[url]
        if isinstance(selected, Exception):
            raise selected
        return selected


@pytest.mark.asyncio
async def test_ir_resolves_location_and_normalizes_both_provider_responses(monkeypatch):
    responses = {
        GEOCODING_API_URL: FakeResponse({"results": [{
            "name": "Colombo", "country": "Sri Lanka", "timezone": "Asia/Colombo",
            "latitude": 6.9271, "longitude": 79.8612,
        }]}),
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }
    client = FakeClient(responses, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"location": "Colombo"})

    assert result.resolved_location.name == "Colombo"
    assert result.resolved_location.timezone == "Asia/Colombo"
    assert result.latitude == 6.9271 and result.longitude == 79.8612
    assert result.weather.forecasts[0].temperature_c == 24.5
    assert result.air_quality.forecasts[0].pm2_5_ug_m3 == 12.4
    assert result.air_quality.forecasts[0].us_aqi == 53
    assert len(result.sources) == 3
    weather_params = next(params for url, params in client.calls if url == FORECAST_API_URL)
    assert weather_params["latitude"] == 6.9271
    assert weather_params["longitude"] == 79.8612


@pytest.mark.asyncio
async def test_valid_coordinates_take_priority_over_city_and_geocoding(monkeypatch):
    responses = {
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }
    client = FakeClient(responses, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({
        "latitude": 6.9, "longitude": 79.8, "location": "Ignored City",
    })

    assert result.weather is not None and result.air_quality is not None
    assert all(url != GEOCODING_API_URL for url, _ in client.calls)
    assert all(params["latitude"] == 6.9 and params["longitude"] == 79.8 for _, params in client.calls)


@pytest.mark.asyncio
async def test_missing_geocoding_timezone_is_preserved_as_unavailable(monkeypatch):
    responses = {
        GEOCODING_API_URL: FakeResponse({"results": [{
            "name": "Colombo", "latitude": 6.9271, "longitude": 79.8612,
        }]}),
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }
    monkeypatch.setattr(
        "app.ir.environment_ir.httpx.AsyncClient",
        lambda *, timeout: FakeClient(responses, timeout=timeout),
    )
    result = await EnvironmentIR().retrieve({"location": "Colombo"})
    assert result.resolved_location.timezone is None
    assert result.weather is not None and result.air_quality is not None


@pytest.mark.asyncio
async def test_ambiguous_city_returns_clarification_error_without_forecast_calls(monkeypatch):
    client = FakeClient({GEOCODING_API_URL: FakeResponse({"results": [
        {"name": "Springfield", "country": "United States", "latitude": 39.78, "longitude": -89.64},
        {"name": "Springfield", "country": "Canada", "latitude": 45.61, "longitude": -62.0},
    ]})}, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"location": "Springfield"})

    assert result.errors[0].code == AgentErrorCode.INVALID_REQUEST
    assert "ambiguous" in result.errors[0].message
    assert [url for url, _ in client.calls] == [GEOCODING_API_URL]


@pytest.mark.asyncio
async def test_bare_known_sri_lankan_city_sends_country_augmented_geocoding_query(monkeypatch):
    """A bare gazetteer city name (no comma/country already present) is
    biased toward Sri Lanka in the geocoding request itself."""
    client = FakeClient({
        GEOCODING_API_URL: FakeResponse({"results": [{
            "name": "Kandy", "country": "Sri Lanka", "timezone": "Asia/Colombo",
            "latitude": 7.2906, "longitude": 80.6336,
        }]}),
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"location": "Kandy"})

    assert result.errors == []
    assert result.resolved_location is not None and result.resolved_location.country == "Sri Lanka"
    geocoding_params = next(params for url, params in client.calls if url == GEOCODING_API_URL)
    assert geocoding_params["name"] == "Kandy, Sri Lanka"


@pytest.mark.asyncio
async def test_bare_city_name_resolves_via_sri_lanka_bias_among_worldwide_matches(monkeypatch):
    """Regression test for the live bug this fix addresses: a bare "Kandy"
    or "Colombo" used to fail with an ambiguous-city error because
    Open-Meteo's geocoding is global and several unrelated places share the
    exact same name. Exactly one Sri Lanka-tagged match among several
    same-named worldwide candidates now resolves automatically instead of
    asking the user to disambiguate."""
    client = FakeClient({
        GEOCODING_API_URL: FakeResponse({"results": [
            {"name": "Kandy", "country": "Sri Lanka", "timezone": "Asia/Colombo", "latitude": 7.2906, "longitude": 80.6336},
            {"name": "Kandy", "country": "Uzbekistan", "latitude": 40.0, "longitude": 65.0},
            {"name": "Kandy", "country": "Kyrgyzstan", "latitude": 41.0, "longitude": 73.0},
        ]}),
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"location": "Kandy"})

    assert result.errors == []
    assert result.resolved_location is not None
    assert result.resolved_location.country == "Sri Lanka"
    assert result.latitude == 7.2906 and result.longitude == 80.6336


@pytest.mark.asyncio
async def test_ambiguous_bare_city_name_still_ambiguous_with_two_sri_lankan_matches(monkeypatch):
    """The Sri Lanka bias is a safety net, not a guess: if more than one
    candidate is itself tagged Sri Lanka, it's still genuinely ambiguous."""
    client = FakeClient({GEOCODING_API_URL: FakeResponse({"results": [
        {"name": "Kandy", "country": "Sri Lanka", "latitude": 7.29, "longitude": 80.63},
        {"name": "Kandy", "country": "Sri Lanka", "latitude": 7.30, "longitude": 80.64},
    ]})}, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"location": "Kandy"})

    assert result.errors[0].code == AgentErrorCode.INVALID_REQUEST
    assert "ambiguous" in result.errors[0].message


@pytest.mark.asyncio
async def test_location_with_explicit_country_is_not_augmented(monkeypatch):
    """A location that already names a country (e.g. composed by the NLP
    pipeline as "<city>, <country>") is sent through unchanged."""
    client = FakeClient({
        GEOCODING_API_URL: FakeResponse({"results": [{
            "name": "Kandy", "country": "Sri Lanka", "timezone": "Asia/Colombo",
            "latitude": 7.2906, "longitude": 80.6336,
        }]}),
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    await EnvironmentIR().retrieve({"location": "Kandy, Sri Lanka"})

    geocoding_params = next(params for url, params in client.calls if url == GEOCODING_API_URL)
    assert geocoding_params["name"] == "Kandy, Sri Lanka"


@pytest.mark.asyncio
async def test_weather_forecast_time_is_time_of_day_not_full_datetime(monkeypatch):
    """forecast.time must be just the time-of-day (e.g. "23:00"), not the
    provider's full ISO datetime -- `date` already carries the date half,
    so a full datetime here duplicated it when formatted for display."""
    client = FakeClient({
        GEOCODING_API_URL: FakeResponse({"results": [{
            "name": "Colombo", "country": "Sri Lanka", "timezone": "UTC",
            "latitude": 6.9271, "longitude": 79.8612,
        }]}),
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"location": "Colombo"})

    assert result.weather is not None
    times_of_day = [forecast.time for forecast in result.weather.forecasts]
    assert times_of_day == ["23:00", "12:00"]
    assert all("T" not in t for t in times_of_day)


@pytest.mark.asyncio
async def test_air_quality_forecasts_are_windowed_not_a_full_two_day_dump(monkeypatch):
    """Air quality used to return every hour of the full 2-day forecast
    (up to 48 rows); it's now windowed to a small number of upcoming hours,
    matching weather's current-forecast framing."""
    hours = 48
    times = [f"2026-09-29T{h % 24:02d}:00" for h in range(hours)]
    payload = {
        "timezone": "UTC",
        "hourly": {
            "time": times,
            "pm2_5": [10.0] * hours,
            "us_aqi": [50] * hours,
            "european_aqi": [20] * hours,
        },
    }
    client = FakeClient({
        GEOCODING_API_URL: FakeResponse({"results": [{
            "name": "Colombo", "country": "Sri Lanka", "timezone": "UTC",
            "latitude": 6.9271, "longitude": 79.8612,
        }]}),
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: FakeResponse(payload),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"location": "Colombo"})

    assert result.air_quality is not None
    assert 0 < len(result.air_quality.forecasts) <= 6


@pytest.mark.asyncio
async def test_air_quality_timeout_preserves_weather_as_partial_success(monkeypatch):
    client = FakeClient({
        FORECAST_API_URL: FakeResponse(weather_payload()),
        AIR_QUALITY_API_URL: httpx.ReadTimeout("private detail"),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"latitude": 6.9, "longitude": 79.8})

    assert result.weather is not None
    assert result.air_quality is None
    assert result.errors[0].code == AgentErrorCode.TIMEOUT
    assert "private detail" not in result.errors[0].message


@pytest.mark.asyncio
async def test_invalid_provider_data_is_reported_without_fabricating_values(monkeypatch):
    client = FakeClient({
        FORECAST_API_URL: FakeResponse({"timezone": "UTC", "hourly": {}}),
        AIR_QUALITY_API_URL: FakeResponse({"timezone": "UTC", "hourly": {"time": ["2026-09-29T12:00"]}}),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"latitude": 6.9, "longitude": 79.8})

    assert result.weather is None and result.air_quality is None
    assert len(result.errors) == 2
    assert "invalid response" in result.errors[0].message
    assert "unavailable" in result.errors[1].message


@pytest.mark.asyncio
async def test_http_error_from_weather_does_not_prevent_air_quality_retrieval(monkeypatch):
    client = FakeClient({
        FORECAST_API_URL: FakeResponse({}, status_code=503),
        AIR_QUALITY_API_URL: FakeResponse(air_payload()),
    }, timeout=10)
    monkeypatch.setattr("app.ir.environment_ir.httpx.AsyncClient", lambda *, timeout: client)

    result = await EnvironmentIR().retrieve({"latitude": 6.9, "longitude": 79.8})

    assert result.weather is None
    assert result.air_quality is not None
    assert result.errors[0].code == AgentErrorCode.AGENT_EXECUTION_FAILED
