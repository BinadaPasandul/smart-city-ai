"""Environment agent weather retrieval using the Open-Meteo Forecast API."""

from datetime import date, datetime, timezone
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse, AgentSource


FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"


class EnvironmentAgent(BaseAgent):
    """Retrieve near-term weather forecasts for a latitude/longitude pair."""

    def __init__(self, *, timeout: float = 10.0) -> None:
        super().__init__(
            "environment",
            description="Provides weather forecasts for supplied coordinates.",
            capabilities=("weather_forecast", "air_quality_forecast"),
        )
        self.timeout = timeout

    async def execute(self, request: AgentRequest) -> AgentResponse:
        coordinates = _read_coordinates(request.context)
        location_query = _read_location(request.context)
        if coordinates is None and location_query is None:
            return _failure(
                request,
                AgentErrorCode.INVALID_REQUEST,
                "Provide valid latitude and longitude coordinates or a city name in request context.",
            )

        resolved_location = None
        weather_data = None
        air_quality_data = None
        sources = []
        errors = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if coordinates is None:
                coordinates, resolved_location, geocoding_error = await _resolve_location(
                    client, location_query
                )
                if geocoding_error:
                    return _failure(
                        request,
                        AgentErrorCode(geocoding_error["code"]),
                        geocoding_error["message"],
                    )
                sources.append(
                    AgentSource(
                        name="Open-Meteo Geocoding API",
                        source_type="api",
                        url=GEOCODING_API_URL,
                        retrieved_at=datetime.now(timezone.utc),
                        metadata={"location": resolved_location},
                    )
                )

            latitude, longitude = coordinates
            weather_params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,precipitation_probability,weather_code",
                "forecast_days": 2,
                "timezone": "auto",
            }
            air_quality_params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "pm2_5,us_aqi,european_aqi",
                "forecast_days": 2,
                "timezone": "auto",
            }

            weather_payload, error = await _request_json(
                client, FORECAST_API_URL, weather_params, "Weather forecast"
            )
            if error:
                errors.append(error)
            else:
                weather_data, error = _build_weather_data(weather_payload)
                if error:
                    errors.append(error)
                else:
                    sources.append(
                        AgentSource(
                            name="Open-Meteo Forecast API",
                            source_type="api",
                            url=FORECAST_API_URL,
                            retrieved_at=datetime.now(timezone.utc),
                            metadata={"timezone": weather_data["timezone"], "latitude": latitude, "longitude": longitude},
                        )
                    )

            air_payload, error = await _request_json(
                client, AIR_QUALITY_API_URL, air_quality_params, "Air quality forecast"
            )
            if error:
                errors.append(error)
            else:
                air_quality_data, error = _build_air_quality_data(air_payload)
                if error:
                    errors.append(error)
                else:
                    sources.append(
                        AgentSource(
                            name="Open-Meteo Air Quality API",
                            source_type="api",
                            url=AIR_QUALITY_API_URL,
                            retrieved_at=datetime.now(timezone.utc),
                            metadata={
                                "timezone": air_quality_data["timezone"],
                                "latitude": latitude,
                                "longitude": longitude,
                                "data_kind": "model_forecast",
                                "observations": False,
                            },
                        )
                    )

        if weather_data is None and air_quality_data is None:
            first_error = errors[0] if errors else {
                "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
                "message": "Weather and air quality providers returned no usable data.",
            }
            return _failure(request, AgentErrorCode(first_error["code"]), first_error["message"])

        sections = []
        metadata = {}
        if resolved_location is not None:
            metadata["location"] = resolved_location
        if weather_data is not None:
            metadata["weather"] = weather_data
            metadata["timezone"] = weather_data["timezone"]
            # Preserve the original weather metadata keys for existing callers.
            metadata["forecasts"] = weather_data["forecasts"]
            sections.append(_format_answer(weather_data["forecasts"], weather_data["timezone"]))
        if air_quality_data is not None:
            metadata["air_quality"] = air_quality_data
            metadata.setdefault("timezone", air_quality_data["timezone"])
            sections.append(_format_air_quality_answer(air_quality_data))
        if errors:
            metadata["partial_errors"] = errors

        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer="\n\n".join(sections),
            sources=sources,
            metadata=metadata,
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
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


async def _resolve_location(
    client: httpx.AsyncClient, query: str
) -> tuple[tuple[float, float] | None, dict[str, Any] | None, dict[str, str] | None]:
    payload, error = await _request_json(
        client,
        GEOCODING_API_URL,
        {"name": query, "count": 10, "language": "en", "format": "json"},
        "Location search",
    )
    if error:
        return None, None, error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return None, None, {
            "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
            "message": "Geocoding provider returned an invalid response.",
        }

    results = payload["results"]
    if not results:
        return None, None, {
            "code": AgentErrorCode.INVALID_REQUEST.value,
            "message": f"No city matched '{query}'. Check the spelling or provide valid coordinates.",
        }

    candidates = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("name"), str) or not result["name"].strip():
            return None, None, {
                "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
                "message": "Geocoding provider returned an invalid location result.",
            }
        coordinates = _read_coordinates(result)
        if coordinates is None:
            return None, None, {
                "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
                "message": "Geocoding provider returned invalid coordinates.",
            }
        candidates.append(
            {
                "name": result["name"].strip(),
                "country": result.get("country") if isinstance(result.get("country"), str) else None,
                "timezone": result.get("timezone") if isinstance(result.get("timezone"), str) else None,
                "latitude": coordinates[0],
                "longitude": coordinates[1],
            }
        )

    exact_matches = [candidate for candidate in candidates if candidate["name"].casefold() == query.casefold()]
    if len(exact_matches) == 1:
        selected = exact_matches[0]
    elif len(exact_matches) > 1 or len(candidates) > 1:
        choices = [
            ", ".join(part for part in (candidate["name"], candidate["country"]) if part)
            for candidate in candidates
        ]
        return None, None, {
            "code": AgentErrorCode.INVALID_REQUEST.value,
            "message": "The city name is ambiguous. Specify a country or provide latitude and longitude. "
            f"Matches: {'; '.join(choices)}",
        }
    else:
        selected = candidates[0]

    return (selected["latitude"], selected["longitude"]), selected, None


async def _request_json(
    client: httpx.AsyncClient, url: str, params: dict[str, Any], service: str
) -> tuple[Any | None, dict[str, str] | None]:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json(), None
    except httpx.TimeoutException:
        return None, {"code": AgentErrorCode.TIMEOUT.value, "message": f"{service} request timed out."}
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            message = f"{service} provider rate limit reached."
        else:
            message = f"{service} provider returned an HTTP error."
        return None, {"code": AgentErrorCode.AGENT_EXECUTION_FAILED.value, "message": message}
    except httpx.HTTPError:
        return None, {"code": AgentErrorCode.AGENT_EXECUTION_FAILED.value, "message": f"{service} provider request failed."}
    except (ValueError, TypeError):
        return None, {"code": AgentErrorCode.AGENT_EXECUTION_FAILED.value, "message": f"{service} provider returned invalid JSON."}


def _build_weather_data(payload: Any) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    parsed = _parse_forecast(payload)
    if parsed is None:
        return None, {
            "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
            "message": "Weather forecast provider returned an invalid response.",
        }

    timezone_name, times, temperatures, probabilities, codes = parsed
    try:
        local_today = datetime.now(ZoneInfo(timezone_name)).date()
    except (ZoneInfoNotFoundError, ValueError):
        return None, {
            "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
            "message": "Weather forecast provider returned an invalid timezone.",
        }

    forecasts = []
    for target_date in (local_today, date.fromordinal(local_today.toordinal() + 1)):
        candidates = [i for i, value in enumerate(times) if _forecast_date(value) == target_date]
        if target_date == local_today:
            current_local = datetime.now(ZoneInfo(timezone_name)).replace(tzinfo=None)
            upcoming = [i for i in candidates if _forecast_datetime(times[i]) >= current_local]
            if upcoming:
                candidates = upcoming
        if candidates:
            index = candidates[0]
            forecasts.append(
                {
                    "date": target_date.isoformat(),
                    "time": times[index],
                    "temperature_c": temperatures[index],
                    "precipitation_probability_percent": probabilities[index],
                    "weather_code": codes[index],
                }
            )

    if not forecasts:
        return None, {
            "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
            "message": "Weather forecast provider returned no usable forecasts for today or tomorrow.",
        }
    return {"timezone": timezone_name, "forecasts": forecasts}, None


def _build_air_quality_data(payload: Any) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not isinstance(payload, dict):
        return None, _invalid_air_quality_response()
    timezone_name = payload.get("timezone")
    hourly = payload.get("hourly")
    if not isinstance(timezone_name, str) or not timezone_name or not isinstance(hourly, dict):
        return None, _invalid_air_quality_response()
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None, {
            "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
            "message": "Air quality provider returned an invalid timezone.",
        }

    times = hourly.get("time")
    if not isinstance(times, list) or not times or any(
        not isinstance(value, str) or _forecast_datetime(value) is None for value in times
    ):
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
            for value in values:
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    return None, _invalid_air_quality_response()

    if not any(value is not None for values in arrays.values() for value in values):
        return None, {
            "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
            "message": "Air quality forecast data is unavailable for this location.",
        }

    forecasts = [
        {
            "time": timestamp,
            "pm2_5_ug_m3": arrays["pm2_5"][index],
            "us_aqi": arrays["us_aqi"][index],
            "european_aqi": arrays["european_aqi"][index],
        }
        for index, timestamp in enumerate(times)
    ]
    return {
        "timezone": timezone_name,
        "data_kind": "model_forecast",
        "observations": False,
        "source_description": "Open-Meteo air quality values are model-based forecasts, not ground-station observations.",
        "units": {
            "pm2_5": "μg/m³",
            "us_aqi": "US AQI index",
            "european_aqi": "European AQI index",
        },
        "aqi_scales": {"us_aqi": "United States AQI", "european_aqi": "European AQI"},
        "forecasts": forecasts,
    }, None


def _invalid_air_quality_response() -> dict[str, str]:
    return {
        "code": AgentErrorCode.AGENT_EXECUTION_FAILED.value,
        "message": "Air quality provider returned an invalid response.",
    }


def _parse_forecast(payload: Any) -> tuple[str, list[str], list[Any], list[Any], list[Any]] | None:
    if not isinstance(payload, dict):
        return None
    timezone_name = payload.get("timezone")
    hourly = payload.get("hourly")
    if not isinstance(timezone_name, str) or not timezone_name or not isinstance(hourly, dict):
        return None
    times = hourly.get("time")
    temperatures = hourly.get("temperature_2m")
    probabilities = hourly.get("precipitation_probability")
    codes = hourly.get("weather_code")
    if not all(isinstance(values, list) for values in (times, temperatures, probabilities, codes)):
        return None
    if not times or not (len(times) == len(temperatures) == len(probabilities) == len(codes)):
        return None
    if any(not isinstance(value, str) or _forecast_datetime(value) is None for value in times):
        return None
    for value in temperatures:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            return None
    for value in probabilities:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
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


def _format_answer(forecasts: list[dict[str, Any]], timezone_name: str) -> str:
    lines = [f"Weather forecast (timezone: {timezone_name}):"]
    for forecast in forecasts:
        details = [f"{forecast['date']} at {forecast['time']}"]
        if forecast["temperature_c"] is not None:
            details.append(f"{forecast['temperature_c']} °C")
        if forecast["precipitation_probability_percent"] is not None:
            details.append(f"{forecast['precipitation_probability_percent']}% precipitation probability")
        if forecast["weather_code"] is not None:
            details.append(f"weather code {forecast['weather_code']}")
        lines.append(" - " + ", ".join(details))
    return "\n".join(lines)


def _format_air_quality_answer(data: dict[str, Any]) -> str:
    lines = [
        "Air quality model forecast (not ground-station observations)",
        f"Timezone: {data['timezone']}",
    ]
    for forecast in data["forecasts"]:
        details = [forecast["time"]]
        if forecast["pm2_5_ug_m3"] is not None:
            details.append(f"PM2.5 {forecast['pm2_5_ug_m3']} μg/m³")
        if forecast["us_aqi"] is not None:
            details.append(f"US AQI {forecast['us_aqi']}")
        if forecast["european_aqi"] is not None:
            details.append(f"European AQI {forecast['european_aqi']}")
        lines.append(" - " + ", ".join(details))
    return "\n".join(lines)


def _failure(request: AgentRequest, code: AgentErrorCode, message: str) -> AgentResponse:
    return AgentResponse(
        request_id=request.request_id,
        agent_name="environment",
        success=False,
        error=AgentError(code=code, message=message),
    )


__all__ = ["EnvironmentAgent"]
