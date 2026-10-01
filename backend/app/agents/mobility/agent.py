"""Mobility Specialist Agent for Smart City AI."""

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.contracts import (
    AgentError,
    AgentErrorCode,
    AgentRequest,
    AgentResponse,
    AgentSource,
)
from app.ir.mobility_ir import MobilityIRService
from app.nlp.mobility_nlp import MobilityIntent, MobilityNLPAnalyzer

logger = logging.getLogger(__name__)


class MobilityAgent(BaseAgent):
    """Specialist agent for city traffic, public transport, parking, and EV charging."""

    def __init__(
        self,
        ir_service: MobilityIRService | None = None,
        nlp_analyzer: MobilityNLPAnalyzer | None = None,
    ) -> None:
        super().__init__(
            "mobility",
            description="Handles city traffic conditions, transit schedules, parking, and EV charging.",
            capabilities=(
                "traffic_monitoring",
                "transit_schedules",
                "parking_search",
                "ev_charging_info",
                "route_recommendation",
            ),
        )
        self.ir_service = ir_service or MobilityIRService()
        self.nlp_analyzer = nlp_analyzer or MobilityNLPAnalyzer()

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Process a citizen query and return domain-specific mobility guidance."""
        if not request.query or not request.query.strip():
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                success=False,
                error=AgentError(
                    code=AgentErrorCode.INVALID_REQUEST,
                    message="Query must not be empty.",
                ),
            )

        intent, entities = self.nlp_analyzer.analyze(request.query)
        logger.info(
            "MobilityAgent executing request_id=%s intent=%s locations=%s mode=%s",
            request.request_id,
            intent.value,
            entities.locations,
            entities.transit_mode,
        )

        try:
            answer, sources, metadata = self._process_query(request.query, intent, entities)
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                success=True,
                answer=answer,
                sources=sources,
                metadata={
                    "intent": intent.value,
                    "extracted_entities": entities.model_dump(mode="json"),
                    "responsible_ai_notice": "Mobility data is retrieved from official municipal and public transit feeds. Real-time conditions may vary during peak hours or sudden weather events.",
                    **metadata,
                },
            )
        except Exception as exc:
            logger.error(
                "MobilityAgent failed request_id=%s exception_type=%s message=%s",
                request.request_id,
                type(exc).__name__,
                exc,
            )
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                success=False,
                error=AgentError(
                    code=AgentErrorCode.AGENT_EXECUTION_FAILED,
                    message="Failed to retrieve mobility information.",
                ),
            )

    def _process_query(
        self, query: str, intent: MobilityIntent, entities: Any
    ) -> tuple[str, list[AgentSource], dict[str, Any]]:
        target_location = entities.locations[0] if entities.locations else None

        if intent == MobilityIntent.TRAFFIC_CHECK:
            return self._handle_traffic(target_location)
        elif intent == MobilityIntent.PARKING_SEARCH:
            return self._handle_parking(target_location, entities.vehicle_type)
        elif intent == MobilityIntent.EV_CHARGING:
            return self._handle_ev_charging(target_location, entities.plug_type)
        elif intent == MobilityIntent.PUBLIC_TRANSIT:
            return self._handle_public_transit(query, entities)
        elif intent == MobilityIntent.ROUTE_PLANNING:
            return self._handle_route_planning(entities)
        else:
            return self._handle_general_mobility(query, target_location)

    def _handle_traffic(self, location: str | None) -> tuple[str, list[AgentSource], dict[str, Any]]:
        records = self.ir_service.search_traffic(location)
        sources = self.ir_service.build_agent_sources(records, "traffic")

        lines = ["### Traffic & Corridor Status Report"]
        if location:
            lines.append(f"Showing traffic conditions for **{location}**:")
        else:
            lines.append("Showing overall city traffic conditions:")

        for r in records:
            status_indicator = "[NORMAL]" if r["status"] in ["Normal Flow", "Clear"] else "[HEAVY]" if "Heavy" in r["status"] else "[MODERATE]"
            lines.append(f"\n* **{r['corridor']}** ({r['location']}):")
            lines.append(f"  - Status: {status_indicator} {r['status']}")
            lines.append(f"  - Average Speed: {r['average_speed_kmh']} km/h | Expected Delay: {r['delay_minutes']} mins")
            if r.get("incidents"):
                incidents_str = "; ".join(r["incidents"])
                lines.append(f"  - [Incident Advisory]: {incidents_str}")

        lines.append("\n*Source: Data provided by Colombo Municipal Traffic Control & Road Development Authority.*")
        return "\n".join(lines), sources, {"records_count": len(records), "category": "traffic"}

    def _handle_parking(
        self, location: str | None, vehicle_type: str | None
    ) -> tuple[str, list[AgentSource], dict[str, Any]]:
        records = self.ir_service.search_parking(location)
        sources = self.ir_service.build_agent_sources(records, "parking")

        lines = ["### Parking Facilities & Real-time Availability"]
        if location:
            lines.append(f"Available parking options near **{location}**:")
        else:
            lines.append("City parking facilities:")

        v_type = vehicle_type or "car"
        for r in records:
            avail = r["available_spots"]
            total = r["total_capacity"]
            rate = r.get("rates_lkr_per_hour", {}).get(v_type, 150)
            avail_indicator = "[FULL]" if avail == 0 else "[LIMITED]" if avail < 20 else "[AVAILABLE]"

            lines.append(f"\n* **{r['name']}** ({r['location']})")
            lines.append(f"  - Address: {r['address']}")
            lines.append(f"  - Availability: {avail_indicator} ({avail}/{total} spots free)")
            lines.append(f"  - Rate ({v_type.title()}): LKR {rate}/hour | Operating Hours: {r['operating_hours']}")

        lines.append("\n*Source: Municipal Council Parking Authority & UDA.*")
        return "\n".join(lines), sources, {"records_count": len(records), "category": "parking"}

    def _handle_ev_charging(
        self, location: str | None, plug_type: str | None
    ) -> tuple[str, list[AgentSource], dict[str, Any]]:
        records = self.ir_service.search_ev_charging(location, plug_type)
        sources = self.ir_service.build_agent_sources(records, "ev_charging")

        lines = ["### EV Charging Infrastructure"]
        filter_desc = []
        if location:
            filter_desc.append(f"near **{location}**")
        if plug_type:
            filter_desc.append(f"with **{plug_type}** connectors")

        desc_str = f" {' '.join(filter_desc)}" if filter_desc else ""
        lines.append(f"Showing EV charging stations{desc_str}:")

        for r in records:
            plugs = ", ".join(r["plug_types"])
            lines.append(f"\n* [EV Charger] **{r['name']}** ({r['provider']})")
            lines.append(f"  - Location: {r['address']}")
            lines.append(f"  - Ports Available: {r['available_ports']}/{r['total_ports']} ports free ({r['status']})")
            lines.append(f"  - Connectors: {plugs}")
            lines.append(f"  - Pricing: LKR {r['price_lkr_per_kwh']}/kWh | Operating Hours: {r['operating_hours']}")

        lines.append("\n*Source: National EV Charging Registry & ChargeNET Feeds.*")
        return "\n".join(lines), sources, {"records_count": len(records), "category": "ev_charging"}

    def _handle_public_transit(
        self, query: str, entities: Any
    ) -> tuple[str, list[AgentSource], dict[str, Any]]:
        mode = getattr(entities, "transit_mode", None)
        buses = [] if mode == "train" else self.ir_service.search_buses(query)
        trains = [] if mode == "bus" else self.ir_service.search_trains(query=query, origin=entities.origin, destination=entities.destination)

        sources = self.ir_service.build_agent_sources(buses, "bus_transit")
        sources.extend(self.ir_service.build_agent_sources(trains, "train_transit"))


        lines = ["### Public Transport Schedules & Route Options"]

        if buses:
            lines.append("\n#### Bus Services:")
            for b in buses:
                stops_str = " -> ".join(b["key_stops"][:4]) + " ..."
                lines.append(f"* **Route {b['route_number']}**: {b['name']}")
                lines.append(f"  - Key Stops: {stops_str}")
                lines.append(f"  - Frequency: Every {b['frequency_minutes']} mins | Fare: LKR {b['fare_lkr']}")
                lines.append(f"  - Service Status: {b['status']}")

        if trains:
            lines.append("\n#### Train Services:")
            for t in trains:
                classes_str = ", ".join(t["classes"])
                lines.append(f"* **{t['train_name']}** ({t['line']})")
                lines.append(f"  - Route: {t['origin']} ({t['departure_time']}) -> {t['destination']} ({t['arrival_time']})")
                lines.append(f"  - Classes: {classes_str} | Schedule: {t['frequency']}")
                lines.append(f"  - Service Status: {t['status']}")


        lines.append("\n*Source: Sri Lanka Railways & National Transport Commission (NTC).*")
        return "\n".join(lines), sources, {
            "bus_count": len(buses),
            "train_count": len(trains),
            "category": "public_transport",
        }

    def _handle_route_planning(self, entities: Any) -> tuple[str, list[AgentSource], dict[str, Any]]:
        origin = entities.origin or (entities.locations[0] if len(entities.locations) > 0 else "Current Location")
        destination = entities.destination or (entities.locations[1] if len(entities.locations) > 1 else entities.locations[0] if entities.locations else "City Center")

        buses = self.ir_service.search_buses(query=f"{origin} {destination}")
        trains = self.ir_service.search_trains(origin=origin, destination=destination)
        parking = self.ir_service.search_parking(location=destination)

        sources = self.ir_service.build_agent_sources(buses, "bus_route")
        sources.extend(self.ir_service.build_agent_sources(trains, "train_route"))

        lines = [f"### Recommended Travel Routes: {origin} to {destination}"]
        lines.append(f"Here are options for travelling from **{origin}** to **{destination}**:")

        if trains:
            t = trains[0]
            lines.append(f"\n1. **Train Option ({t['line']})**: Take {t['train_name']} departing {t['origin']} at {t['departure_time']}. Fast and cost-effective.")
        if buses:
            b = buses[0]
            lines.append(f"\n2. **Bus Option (Route {b['route_number']})**: {b['name']} running every {b['frequency_minutes']} mins (Fare: LKR {b['fare_lkr']}).")

        lines.append(f"\n3. **Driving / Private Transport**: Via primary arterial corridors. Estimated drive time 25-35 mins depending on traffic congestion.")

        if parking:
            p = parking[0]
            lines.append(f"\n[Parking at Destination]: {p['name']} ({p['available_spots']} spots available, LKR {p['rates_lkr_per_hour'].get('car', 150)}/hr).")

        return "\n".join(lines), sources, {"category": "route_planning", "origin": origin, "destination": destination}

    def _handle_general_mobility(
        self, query: str, location: str | None
    ) -> tuple[str, list[AgentSource], dict[str, Any]]:
        traffic = self.ir_service.search_traffic(location)
        buses = self.ir_service.search_buses(location)
        parking = self.ir_service.search_parking(location)

        sources = self.ir_service.build_agent_sources(traffic[:1], "traffic")
        sources.extend(self.ir_service.build_agent_sources(parking[:1], "parking"))

        lines = ["### City Mobility Overview"]
        if location:
            lines.append(f"Mobility information around **{location}**:")
        else:
            lines.append("General city transportation and mobility status:")

        if traffic:
            t = traffic[0]
            lines.append(f"\n* **Traffic**: {t['corridor']} is reporting {t['status']} (Avg speed: {t['average_speed_kmh']} km/h).")

        if parking:
            p = parking[0]
            lines.append(f"\n* **Parking**: {p['name']} has {p['available_spots']} spots available.")

        if buses:
            b = buses[0]
            lines.append(f"\n* **Public Transit**: Bus Route {b['route_number']} ({b['name']}) active.")

        lines.append("\nNeed specific details? Ask about traffic, bus/train schedules, parking spaces, or EV charging.")
        return "\n".join(lines), sources, {"category": "general_mobility"}
