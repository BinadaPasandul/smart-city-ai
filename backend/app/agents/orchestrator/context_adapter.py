"""Build safe, specialist-compatible copies of a shared agent request."""

import math
from typing import Any

from app.agents.contracts import AgentRequest


class AmbiguousSpecialistContext(ValueError):
    """A selected specialist needs one location, but the request has several."""


class SpecialistContextAdapter:
    """Adapt trusted shared context without mutating the caller's request."""

    def adapt(self, agent_name: str, request: AgentRequest) -> AgentRequest:
        context = request.model_copy(deep=True).context
        if agent_name != "environment":
            return request.model_copy(update={"context": context}, deep=True)

        locations = self._trusted_locations(context)
        if len(locations) > 1:
            raise AmbiguousSpecialistContext(
                "Environment requests currently support one location at a time."
            )

        safe_root_data = self._validated_location_data(context)
        safe_user_data = self._validated_location_data(context.get("user_context"))
        safe_coordinates = (
            safe_root_data
            if "latitude" in safe_root_data and "longitude" in safe_root_data
            else safe_user_data
            if "latitude" in safe_user_data and "longitude" in safe_user_data
            else {}
        )
        # EnvironmentIR gives top-level coordinates precedence over location.
        if safe_coordinates:
            context.update(safe_coordinates)
        elif locations:
            context["location"] = locations[0]
        elif "location" in safe_user_data:
            context["location"] = safe_user_data["location"]
        elif "location" in safe_root_data:
            context["location"] = safe_root_data["location"]
        elif any(
            isinstance(source, dict)
            and any(key in source for key in ("latitude", "longitude", "location"))
            for source in (context, context.get("user_context"))
        ):
            # EnvironmentIR merges user_context into the root. An explicit empty
            # value prevents it from consuming an unvalidated nested location.
            context["location"] = ""

        return request.model_copy(update={"context": context}, deep=True)

    @classmethod
    def has_valid_user_location(cls, context: dict[str, Any] | None) -> bool:
        """Check client context without trusting or copying arbitrary values."""
        if not isinstance(context, dict):
            return False
        return bool(
            cls._validated_location_data(context)
            or cls._validated_location_data(context.get("user_context"))
        )

    @staticmethod
    def _trusted_locations(context: dict[str, Any]) -> list[str]:
        nlp = context.get("nlp")
        if not isinstance(nlp, dict):
            return []
        values = nlp.get("locations")
        if not isinstance(values, list):
            return []
        locations: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            key = cleaned.casefold()
            if cleaned and len(cleaned) <= 120 and not any(ord(char) < 32 for char in cleaned):
                if key not in seen:
                    seen.add(key)
                    locations.append(cleaned)
        return locations

    @staticmethod
    def _validated_location_data(value: Any) -> dict[str, float | str]:
        if not isinstance(value, dict):
            return {}

        latitude = value.get("latitude")
        longitude = value.get("longitude")
        if (
            SpecialistContextAdapter._valid_number(latitude, -90.0, 90.0)
            and SpecialistContextAdapter._valid_number(longitude, -180.0, 180.0)
        ):
            return {"latitude": float(latitude), "longitude": float(longitude)}

        location = value.get("location")
        if (
            isinstance(location, str)
            and 0 < len(location.strip()) <= 120
            and not any(ord(char) < 32 for char in location)
        ):
            return {"location": location.strip()}
        return {}

    @staticmethod
    def _valid_number(value: Any, minimum: float, maximum: float) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and minimum <= value <= maximum
        )
