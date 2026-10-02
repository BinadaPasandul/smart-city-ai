"""Environment specialist: coordinate retrieval and format agent responses."""

from typing import Any

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentRequest, AgentResponse
from app.ir.environment_ir import (
    AIR_QUALITY_API_URL,
    FORECAST_API_URL,
    GEOCODING_API_URL,
    EnvironmentIR,
)
from app.nlp.environment_nlp import EnvironmentIntent, EnvironmentNLP


class EnvironmentAgent(BaseAgent):
    """Coordinate Environment retrieval and map it to shared agent contracts."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        ir: EnvironmentIR | None = None,
        nlp: EnvironmentNLP | None = None,
    ) -> None:
        super().__init__(
            "environment",
            description="Provides weather and air-quality forecasts for supplied locations.",
            capabilities=("weather_forecast", "air_quality_forecast"),
        )
        self.ir = ir or EnvironmentIR(timeout=timeout)
        self.nlp = nlp or EnvironmentNLP()

    async def execute(self, request: AgentRequest) -> AgentResponse:
        parsed = self.nlp.parse(request.query)
        include_weather, include_air_quality = _retrieval_selection(parsed.intent)
        result = await self.ir.retrieve(
            request.context,
            include_weather=include_weather,
            include_air_quality=include_air_quality,
        )
        if result.weather is None and result.air_quality is None:
            first_error = result.errors[0] if result.errors else None
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                success=False,
                error=AgentError(
                    code=first_error.code if first_error else "agent_execution_failed",
                    message=first_error.message if first_error else "Weather and air quality providers returned no usable data.",
                ),
            )

        metadata: dict[str, Any] = {}
        sections: list[str] = []
        if result.resolved_location is not None:
            metadata["location"] = result.resolved_location.model_dump()
        if result.weather is not None:
            weather = result.weather.model_dump()
            metadata["weather"] = weather
            metadata["timezone"] = result.weather.timezone
            # Preserve the original weather metadata key for existing callers.
            metadata["forecasts"] = weather["forecasts"]
            sections.append(_format_weather(result.weather))
        if result.air_quality is not None:
            air_quality = result.air_quality.model_dump()
            metadata["air_quality"] = air_quality
            metadata.setdefault("timezone", result.air_quality.timezone)
            sections.append(_format_air_quality(result.air_quality.model_dump()))
        if result.errors:
            metadata["partial_errors"] = [error.model_dump(mode="json") for error in result.errors]

        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer="\n\n".join(sections),
            sources=result.sources,
            metadata=metadata,
        )


def _retrieval_selection(intent: EnvironmentIntent) -> tuple[bool, bool]:
    """Map parsed request intent to feeds; unknown retains legacy both-feed behavior."""
    if intent is EnvironmentIntent.WEATHER:
        return True, False
    if intent is EnvironmentIntent.AIR_QUALITY:
        return False, True
    # BOTH and UNKNOWN retain the established all-environment-data behavior.
    return True, True


def _format_weather(data: Any) -> str:
    lines = [f"Weather forecast (timezone: {data.timezone}):"]
    for forecast in data.forecasts:
        details = [f"{forecast.date} at {forecast.time}"]
        if forecast.temperature_c is not None:
            details.append(f"{forecast.temperature_c} °C")
        if forecast.precipitation_probability_percent is not None:
            details.append(f"{forecast.precipitation_probability_percent}% precipitation probability")
        if forecast.weather_code is not None:
            details.append(f"weather code {forecast.weather_code}")
        lines.append(" - " + ", ".join(details))
    return "\n".join(lines)


def _format_air_quality(data: dict[str, Any]) -> str:
    lines = [
        "Air quality model forecast (not ground-station observations)",
        f"Timezone: {data['timezone']}",
    ]
    for forecast in data["forecasts"]:
        details = [forecast["time"]]
        if forecast["pm2_5_ug_m3"] is not None:
            details.append(f"PM2.5 {forecast['pm2_5_ug_m3']} {data['units']['pm2_5']}")
        if forecast["us_aqi"] is not None:
            details.append(f"US AQI {forecast['us_aqi']}")
        if forecast["european_aqi"] is not None:
            details.append(f"European AQI {forecast['european_aqi']}")
        lines.append(" - " + ", ".join(details))
    return "\n".join(lines)


__all__ = [
    "AIR_QUALITY_API_URL", "FORECAST_API_URL", "GEOCODING_API_URL", "EnvironmentAgent",
]
