"""Information Retrieval service for city mobility data."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agents.contracts import AgentSource

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "mobility_data.json"


class MobilityIRService:
    """Retrieves structured mobility data for traffic, transit, parking, and EV charging."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self._data: dict[str, Any] = self._load_data()

    def _load_data(self) -> dict[str, Any]:
        if not self.data_path.exists():
            logger.warning("Mobility data file not found at %s. Using empty dataset.", self.data_path)
            return {"traffic": [], "public_transport": {"buses": [], "trains": []}, "parking": [], "ev_charging": []}
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to parse mobility data JSON: %s", exc)
            return {"traffic": [], "public_transport": {"buses": [], "trains": []}, "parking": [], "ev_charging": []}

    def search_traffic(self, location: str | None = None) -> list[dict[str, Any]]:
        """Search traffic records matching a location or corridor."""
        traffic_records = self._data.get("traffic", [])
        if not location:
            return traffic_records

        loc_lower = location.lower()
        results = [
            item for item in traffic_records
            if loc_lower in item.get("location", "").lower()
            or loc_lower in item.get("corridor", "").lower()
        ]
        return results if results else traffic_records

    def search_buses(self, query: str | None = None, route_number: str | None = None) -> list[dict[str, Any]]:
        """Search bus routes matching query or route number."""
        buses = self._data.get("public_transport", {}).get("buses", [])
        if route_number:
            return [b for b in buses if b.get("route_number") == route_number]

        if not query:
            return buses

        q_lower = query.lower()
        results = []
        for bus in buses:
            stops_match = any(q_lower in stop.lower() for stop in bus.get("key_stops", []))
            name_match = q_lower in bus.get("name", "").lower() or q_lower in bus.get("route_number", "")
            if name_match or stops_match:
                results.append(bus)
        return results if results else buses

    def search_trains(self, origin: str | None = None, destination: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        """Search train lines and schedules matching origin, destination, or query."""
        trains = self._data.get("public_transport", {}).get("trains", [])
        if not origin and not destination and not query:
            return trains

        results = []
        for train in trains:
            orig_match = not origin or (origin.lower() in train.get("origin", "").lower())
            dest_match = not destination or (
                destination.lower() in train.get("destination", "").lower()
                or any(destination.lower() in s.lower() for s in train.get("stops", []))
            )
            query_match = not query or (
                query.lower() in train.get("train_name", "").lower()
                or query.lower() in train.get("line", "").lower()
                or any(query.lower() in s.lower() for s in train.get("stops", []))
            )

            if orig_match and dest_match and query_match:
                results.append(train)

        return results if results else trains

    def search_parking(self, location: str | None = None) -> list[dict[str, Any]]:
        """Search parking facilities by location or name."""
        parking_records = self._data.get("parking", [])
        if not location:
            return parking_records

        loc_lower = location.lower()
        results = [
            item for item in parking_records
            if loc_lower in item.get("location", "").lower()
            or loc_lower in item.get("address", "").lower()
            or loc_lower in item.get("name", "").lower()
        ]
        return results if results else parking_records

    def search_ev_charging(self, location: str | None = None, plug_type: str | None = None) -> list[dict[str, Any]]:
        """Search EV charging stations by location and optional plug type."""
        ev_records = self._data.get("ev_charging", [])
        results = ev_records

        if location:
            loc_lower = location.lower()
            results = [
                item for item in results
                if loc_lower in item.get("location", "").lower()
                or loc_lower in item.get("address", "").lower()
                or loc_lower in item.get("name", "").lower()
            ]
            if not results:
                results = ev_records

        if plug_type:
            plug_lower = plug_type.lower()
            plug_results = [
                item for item in results
                if any(plug_lower in p.lower() for p in item.get("plug_types", []))
            ]
            if plug_results:
                results = plug_results

        return results

    def build_agent_sources(self, records: list[dict[str, Any]], category: str) -> list[AgentSource]:
        """Convert raw IR records into typed AgentSource objects."""
        sources: list[AgentSource] = []
        for item in records:
            source_name = item.get("source_name", f"Smart City {category.title()} Data")
            last_updated_str = item.get("last_updated")
            retrieved_at = None
            if last_updated_str:
                try:
                    retrieved_at = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
                except ValueError:
                    retrieved_at = datetime.utcnow()
            else:
                retrieved_at = datetime.utcnow()

            sources.append(
                AgentSource(
                    name=source_name,
                    source_type="structured_dataset",
                    retrieved_at=retrieved_at,
                    metadata={
                        "category": category,
                        "item_id": item.get("id") or item.get("route_number") or item.get("train_name"),
                        "location": item.get("location") or item.get("name"),
                        "status": item.get("status", "Available"),
                    },
                )
            )
        return sources
