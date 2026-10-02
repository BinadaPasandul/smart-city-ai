"""Open-Meteo retrieval and normalization for Environment Agent data."""

from datetime import date, datetime, timezone
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, Field

from app.agents.contracts import AgentErrorCode, AgentSource
from app.nlp.sri_lanka_locations import KNOWN_LOCATIONS, is_known_country

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"

# This project is scoped to Sri Lanka (see sri_lanka_locations.py), but
# Open-Meteo's geocoding is global: a bare "Kandy" or "Colombo" matches
# several same-named places worldwide and fails with an ambiguous-city
# error. A bare gazetteer city name is biased toward Sri Lanka before the
# request is made (see `_augment_with_known_country`), and candidate
# selection is biased toward Sri Lanka as a safety net for names outside
# the gazetteer (see `_resolve_location`).
_KNOWN_LOCATIONS_CASEFOLDED = frozenset(name.casefold() for name in KNOWN_LOCATIONS)
_AIR_QUALITY_WINDOW_HOURS = 6


class EnvironmentProviderError(BaseModel):
    """Safe, structured provider or location-resolution error."""

    code: AgentErrorCode
    message: str


class ResolvedLocation(BaseModel):
    """A geocoding result; timezone may legitimately be unavailable."""

    name: str
    country: str | None = None
    timezone: str | None = None
    latitude: float
    longitude: float


class WeatherForecast(BaseModel):
    date: str
    time: str
    temperature_c: float | None
    precipitation_probability_percent: float | None
    weather_code: int | None


class WeatherData(BaseModel):
    timezone: str
    forecasts: list[WeatherForecast]


class AirQualityForecast(BaseModel):
    time: str
    pm2_5_ug_m3: float | None
    us_aqi: float | None
    european_aqi: float | None


class AirQualityData(BaseModel):
    timezone: str
    data_kind: str = "model_forecast"
    observations: bool = False
    source_description: str = (
        "Open-Meteo air quality values are model-based forecasts, not ground-station observations."
    )
    units: dict[str, str] = Field(default_factory=lambda: {
        "pm2_5": "μg/m³",
        "us_aqi": "US AQI index",
        "european_aqi": "European AQI index",
    })
    aqi_scales: dict[str, str] = Field(default_factory=lambda: {
        "us_aqi": "United States AQI",
        "european_aqi": "European AQI",
    })
    forecasts: list[AirQualityForecast]


class EnvironmentRetrievalResult(BaseModel):
    """Normalized retrieval outcome; weather and air quality can be partial."""

    latitude: float | None = None
    longitude: float | None = None
    resolved_location: ResolvedLocation | None = None
    weather: WeatherData | None = None
    air_quality: AirQualityData | None = None
    sources: list[AgentSource] = Field(default_factory=list)
    errors: list[EnvironmentProviderError] = Field(default_factory=list)


class EnvironmentIR:
    """Asynchronously resolve a location and retrieve Open-Meteo forecasts."""

    def __init__(self, *, timeout: float = 10.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    async def retrieve(
        self,
        context: dict[str, Any],
        *,
        include_weather: bool = True,
        include_air_quality: bool = True,
    ) -> EnvironmentRetrievalResult:
        """Resolve coordinates from context, then request selected forecast feeds.

        Valid explicit coordinates take precedence over `context["location"]`.
        Invalid or incomplete coordinates may fall back to a non-empty city name,
        matching the Environment Agent's established request-context behavior.
        Both feeds are requested by default to preserve existing callers.
        """
        if not include_weather and not include_air_quality:
            return EnvironmentRetrievalResult(errors=[EnvironmentProviderError(
                code=AgentErrorCode.INVALID_REQUEST,
                message="At least one environment forecast type must be requested.",
            )])
        # The chat orchestrator keeps caller-supplied values under user_context
        # when it adds trusted NLP facts. Direct agent/API callers may still
        # provide location and coordinates at the top level.
        user_context = context.get("user_context")
        if isinstance(user_context, dict):
            context = {**user_context, **context}
        coordinates = _read_coordinates(context)
        location_query = _read_location(context)
        if coordinates is None and location_query is None:
            return EnvironmentRetrievalResult(errors=[EnvironmentProviderError(
                code=AgentErrorCode.INVALID_REQUEST,
                message="Provide valid latitude and longitude coordinates or a city name in request context.",
            )])

        sources: list[AgentSource] = []
        resolved_location = None
        errors: list[EnvironmentProviderError] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if coordinates is None:
                coordinates, resolved_location, error = await _resolve_location(client, location_query)
                if error:
                    return EnvironmentRetrievalResult(errors=[error])
                sources.append(AgentSource(
                    name="Open-Meteo Geocoding API",
                    source_type="api",
                    url=GEOCODING_API_URL,
                    retrieved_at=datetime.now(timezone.utc),
                    metadata={"location": resolved_location.model_dump()},
                ))

            latitude, longitude = coordinates
            weather = None
            if include_weather:
                weather_payload, error = await _request_json(
                    client,
                    FORECAST_API_URL,
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "hourly": "temperature_2m,precipitation_probability,weather_code",
                        "forecast_days": 2,
                        "timezone": "auto",
                    },
                    "Weather forecast",
                )
                if error:
                    errors.append(error)
                else:
                    weather, error = _build_weather_data(weather_payload)
                    if error:
                        errors.append(error)
                    else:
                        sources.append(AgentSource(
                            name="Open-Meteo Forecast API",
                            source_type="api",
                            url=FORECAST_API_URL,
                            retrieved_at=datetime.now(timezone.utc),
                            metadata={"timezone": weather.timezone, "latitude": latitude, "longitude": longitude},
                        ))

            air_quality = None
            if include_air_quality:
                air_payload, error = await _request_json(
                    client,
                    AIR_QUALITY_API_URL,
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "hourly": "pm2_5,us_aqi,european_aqi",
                        "forecast_days": 2,
                        "timezone": "auto",
                    },
                    "Air quality forecast",
                )
                if error:
                    errors.append(error)
                else:
                    air_quality, error = _build_air_quality_data(air_payload)
                    if error:
                        errors.append(error)
                    else:
                        sources.append(AgentSource(
                            name="Open-Meteo Air Quality API",
                            source_type="api",
                            url=AIR_QUALITY_API_URL,
                            retrieved_at=datetime.now(timezone.utc),
                            metadata={
                                "timezone": air_quality.timezone,
                                "latitude": latitude,
                                "longitude": longitude,
                                "data_kind": air_quality.data_kind,
                                "observations": air_quality.observations,
                            },
                        ))

        return EnvironmentRetrievalResult(
            latitude=coordinates[0],
            longitude=coordinates[1],
            resolved_location=resolved_location,
            weather=weather,
            air_quality=air_quality,
            sources=sources,
            errors=errors,
        )


def _read_coordinates(context: dict[str, Any]) -> tuple[float, float] | None:
    try:
        raw_latitude, raw_longitude = context["latitude"], context["longitude"]
        if isinstance(raw_latitude, bool) or isinstance(raw_longitude, bool):
            return None
        latitude, longitude = float(raw_latitude), float(raw_longitude)
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            return None
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return None
        return latitude, longitude
    except (KeyError, TypeError, ValueError):
        return None


def _read_location(context: dict[str, Any]) -> str | None:
    value = context.get("location")
    return value.strip() or None if isinstance(value, str) else None


async def _resolve_location(
    client: httpx.AsyncClient, query: str
) -> tuple[tuple[float, float] | None, ResolvedLocation | None, EnvironmentProviderError | None]:
    payload, error = await _request_json(
        client, GEOCODING_API_URL,
        {"name": _augment_with_known_country(query), "count": 10, "language": "en", "format": "json"},
        "Location search",
    )
    if error:
        return None, None, error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return None, None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Geocoding provider returned an invalid response.")
    results = payload["results"]
    if not results:
        return None, None, _error(
            AgentErrorCode.INVALID_REQUEST,
            f"No city matched '{query}'. Check the spelling or provide valid coordinates.",
        )

    candidates: list[ResolvedLocation] = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("name"), str) or not result["name"].strip():
            return None, None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Geocoding provider returned an invalid location result.")
        coordinates = _read_coordinates(result)
        if coordinates is None:
            return None, None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Geocoding provider returned invalid coordinates.")
        candidates.append(ResolvedLocation(
            name=result["name"].strip(),
            country=result.get("country") if isinstance(result.get("country"), str) else None,
            timezone=result.get("timezone") if isinstance(result.get("timezone"), str) else None,
            latitude=coordinates[0], longitude=coordinates[1],
        ))

    exact = [item for item in candidates if item.name.casefold() == query.casefold()]
    if len(exact) == 1:
        selected = exact[0]
    elif len(exact) > 1 or len(candidates) > 1:
        # Still ambiguous after the query-side bias above (e.g. a name
        # outside the gazetteer, or Open-Meteo returning more than one
        # Sri Lanka-tagged result alongside others). If exactly one
        # candidate in the relevant pool is tagged Sri Lanka, prefer it
        # rather than asking the user to disambiguate a Sri-Lanka-scoped
        # assistant's own country.
        pool = exact or candidates
        sri_lankan = [item for item in pool if is_known_country(item.country or "")]
        if len(sri_lankan) == 1:
            selected = sri_lankan[0]
        else:
            choices = [", ".join(part for part in (item.name, item.country) if part) for item in candidates]
            return None, None, _error(
                AgentErrorCode.INVALID_REQUEST,
                "The city name is ambiguous. Specify a country or provide latitude and longitude. "
                f"Matches: {'; '.join(choices)}",
            )
    else:
        selected = candidates[0]
    return (selected.latitude, selected.longitude), selected, None


def _augment_with_known_country(query: str) -> str:
    """Append ", Sri Lanka" to a bare gazetteer city name before searching.

    Only applies when the query is *exactly* a known Sri Lankan city with no
    qualifier already present (a query containing a comma is left alone --
    the caller, e.g. the NLP pipeline's own "<location>, <country>"
    composition, already said which place they mean).
    """
    stripped = query.strip()
    if "," in stripped:
        return query
    if stripped.casefold() in _KNOWN_LOCATIONS_CASEFOLDED:
        return f"{stripped}, Sri Lanka"
    return query


async def _request_json(
    client: httpx.AsyncClient, url: str, params: dict[str, Any], service: str
) -> tuple[Any | None, EnvironmentProviderError | None]:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json(), None
    except httpx.TimeoutException:
        return None, _error(AgentErrorCode.TIMEOUT, f"{service} request timed out.")
    except httpx.HTTPStatusError as exc:
        message = f"{service} provider rate limit reached." if exc.response.status_code == 429 else f"{service} provider returned an HTTP error."
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, message)
    except httpx.HTTPError:
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, f"{service} provider request failed.")
    except (ValueError, TypeError):
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, f"{service} provider returned invalid JSON.")


def _build_weather_data(payload: Any) -> tuple[WeatherData | None, EnvironmentProviderError | None]:
    parsed = _parse_forecast(payload)
    if parsed is None:
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Weather forecast provider returned an invalid response.")
    timezone_name, times, temperatures, probabilities, codes = parsed
    try:
        local_today = datetime.now(ZoneInfo(timezone_name)).date()
    except (ZoneInfoNotFoundError, ValueError):
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Weather forecast provider returned an invalid timezone.")

    forecasts = []
    for target_date in (local_today, date.fromordinal(local_today.toordinal() + 1)):
        candidates = [i for i, value in enumerate(times) if _forecast_date(value) == target_date]
        if target_date == local_today:
            current_local = datetime.now(ZoneInfo(timezone_name)).replace(tzinfo=None)
            upcoming = [i for i in candidates if _forecast_datetime(times[i]) >= current_local]
            if upcoming:
                candidates = upcoming
        if candidates:
            i = candidates[0]
            # `times[i]` is the provider's full ISO datetime (e.g.
            # "2026-10-02T17:00"); `date` above already carries the date
            # half, so only the time-of-day is kept here to avoid the date
            # appearing twice when this is formatted for display.
            forecast_dt = _forecast_datetime(times[i])
            forecasts.append(WeatherForecast(
                date=target_date.isoformat(),
                time=forecast_dt.strftime("%H:%M") if forecast_dt is not None else times[i],
                temperature_c=temperatures[i],
                precipitation_probability_percent=probabilities[i], weather_code=codes[i],
            ))
    if not forecasts:
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Weather forecast provider returned no usable forecasts for today or tomorrow.")
    return WeatherData(timezone=timezone_name, forecasts=forecasts), None


def _build_air_quality_data(payload: Any) -> tuple[AirQualityData | None, EnvironmentProviderError | None]:
    if not isinstance(payload, dict):
        return None, _invalid_air_quality_response()
    timezone_name, hourly = payload.get("timezone"), payload.get("hourly")
    if not isinstance(timezone_name, str) or not timezone_name or not isinstance(hourly, dict):
        return None, _invalid_air_quality_response()
    try:
        tzinfo = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Air quality provider returned an invalid timezone.")
    times = hourly.get("time")
    if not isinstance(times, list) or not times or any(not isinstance(v, str) or _forecast_datetime(v) is None for v in times):
        return None, _invalid_air_quality_response()
    arrays: dict[str, list[Any]] = {}
    for field in ("pm2_5", "us_aqi", "european_aqi"):
        values = hourly.get(field)
        if values is None:
            arrays[field] = [None] * len(times)
        elif not isinstance(values, list) or len(values) != len(times):
            return None, _invalid_air_quality_response()
        else:
            arrays[field] = values
            if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)) for v in values):
                return None, _invalid_air_quality_response()
    if not any(v is not None for values in arrays.values() for v in values):
        return None, _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Air quality forecast data is unavailable for this location.")

    # The provider returns a full multi-day hourly series; keep only a
    # small window (current hour onward) so the answer stays readable
    # instead of dumping every hour of both forecast days. Mirrors
    # weather's current-forecast framing above.
    current_local = datetime.now(tzinfo).replace(tzinfo=None)
    upcoming = [i for i, value in enumerate(times) if _forecast_datetime(value) >= current_local]
    window_indices = (
        upcoming[:_AIR_QUALITY_WINDOW_HOURS]
        if upcoming
        else list(range(min(_AIR_QUALITY_WINDOW_HOURS, len(times))))
    )

    forecasts = [AirQualityForecast(
        time=times[i], pm2_5_ug_m3=arrays["pm2_5"][i], us_aqi=arrays["us_aqi"][i],
        european_aqi=arrays["european_aqi"][i],
    ) for i in window_indices]
    return AirQualityData(timezone=timezone_name, forecasts=forecasts), None


def _invalid_air_quality_response() -> EnvironmentProviderError:
    return _error(AgentErrorCode.AGENT_EXECUTION_FAILED, "Air quality provider returned an invalid response.")


def _parse_forecast(payload: Any) -> tuple[str, list[str], list[Any], list[Any], list[Any]] | None:
    if not isinstance(payload, dict):
        return None
    timezone_name, hourly = payload.get("timezone"), payload.get("hourly")
    if not isinstance(timezone_name, str) or not timezone_name or not isinstance(hourly, dict):
        return None
    times, temperatures = hourly.get("time"), hourly.get("temperature_2m")
    probabilities, codes = hourly.get("precipitation_probability"), hourly.get("weather_code")
    arrays = (times, temperatures, probabilities, codes)
    if not all(isinstance(values, list) for values in arrays):
        return None
    if not times or len({len(values) for values in arrays}) != 1:
        return None
    if any(not isinstance(v, str) or _forecast_datetime(v) is None for v in times):
        return None
    for value in temperatures:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            return None
    for value in probabilities:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100):
            return None
    for value in codes:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            return None
    return timezone_name, times, temperatures, probabilities, codes


def _forecast_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _forecast_date(value: str) -> date | None:
    parsed = _forecast_datetime(value)
    return parsed.date() if parsed is not None else None


def _error(code: AgentErrorCode, message: str) -> EnvironmentProviderError:
    return EnvironmentProviderError(code=code, message=message)


__all__ = [
    "AIR_QUALITY_API_URL", "FORECAST_API_URL", "GEOCODING_API_URL",
    "AirQualityData", "AirQualityForecast", "EnvironmentIR",
    "EnvironmentProviderError", "EnvironmentRetrievalResult", "ResolvedLocation",
    "WeatherData", "WeatherForecast",
]
