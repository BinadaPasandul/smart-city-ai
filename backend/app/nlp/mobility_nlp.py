"""NLP Intent Classifier and Named Entity Recognition (NER) for Mobility Domain."""

import re
from enum import Enum
from pydantic import BaseModel, Field


class MobilityIntent(str, Enum):
    """Categorized user intent within the mobility domain."""

    TRAFFIC_CHECK = "traffic_check"
    ROUTE_PLANNING = "route_planning"
    PUBLIC_TRANSIT = "public_transit"
    PARKING_SEARCH = "parking_search"
    EV_CHARGING = "ev_charging"
    GENERAL_MOBILITY = "general_mobility"


class MobilityEntities(BaseModel):
    """Extracted entities relevant to mobility queries."""

    locations: list[str] = Field(default_factory=list)
    origin: str | None = None
    destination: str | None = None
    transit_mode: str | None = None
    plug_type: str | None = None
    vehicle_type: str | None = None


KNOWN_LOCATIONS = [
    "colombo fort", "fort", "pettah", "kollupitiya", "bambalapitiya",
    "galle face", "wellawatte", "dehiwala", "mount lavinia", "panadura",
    "galle road", "kandy road", "maharagama", "nugegoda", "rajagiriya",
    "kaduwela", "battaramulla", "borella", "town hall", "kandy", "matara",
    "kelaniya", "peliyagoda", "kottawa", "southern expressway"
]

TRAFFIC_KEYWORDS = ["traffic", "congestion", "jam", "delay", "highway", "expressway", "speed", "road block", "accident", "corridor"]
PARKING_KEYWORDS = ["park", "parking", "car park", "garage", "space", "vehicle spot", "capacity"]
EV_KEYWORDS = ["ev", "electric vehicle", "charging", "charger", "plug", "ccs2", "type 2", "chargenet", "kwh"]
TRANSIT_KEYWORDS = ["bus", "train", "transit", "route", "schedule", "timetable", "fare", "ticket", "station", "express line"]
ROUTE_KEYWORDS = ["how do i get to", "route to", "travel to", "best way to", "directions to", "navigate to", "from", "to"]


class MobilityNLPAnalyzer:
    """Analyze citizen queries to extract mobility intent and structured entities."""

    def analyze(self, query: str) -> tuple[MobilityIntent, MobilityEntities]:
        query_lower = query.lower()
        intent = self._classify_intent(query_lower)
        entities = self._extract_entities(query_lower)
        return intent, entities

    def _classify_intent(self, query_lower: str) -> MobilityIntent:
        # Check EV first for explicit EV terms
        if any(kw in query_lower for kw in EV_KEYWORDS):
            return MobilityIntent.EV_CHARGING
        # Check Parking
        if any(kw in query_lower for kw in PARKING_KEYWORDS):
            return MobilityIntent.PARKING_SEARCH
        # Check Traffic
        if any(kw in query_lower for kw in TRAFFIC_KEYWORDS):
            return MobilityIntent.TRAFFIC_CHECK
        # Check Route planning vs Public transit
        if any(kw in query_lower for kw in ROUTE_KEYWORDS) and ("from" in query_lower or "to" in query_lower):
            return MobilityIntent.ROUTE_PLANNING
        if any(kw in query_lower for kw in TRANSIT_KEYWORDS):
            return MobilityIntent.PUBLIC_TRANSIT

        return MobilityIntent.GENERAL_MOBILITY

    def _extract_entities(self, query_lower: str) -> MobilityEntities:
        extracted_locs = []
        for loc in KNOWN_LOCATIONS:
            if re.search(r'\b' + re.escape(loc) + r'\b', query_lower):
                # Format nicely (e.g. Title Case)
                extracted_locs.append(loc.title())

        origin = None
        destination = None
        # Extract origin/destination patterns like "from X to Y" or "to Y from X"
        from_to_match = re.search(r'from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)', query_lower)
        if from_to_match:
            origin = from_to_match.group(1).strip().title()
            destination = from_to_match.group(2).strip().title()
        else:
            to_match = re.search(r'to\s+([a-zA-Z\s]+)', query_lower)
            if to_match:
                destination = to_match.group(1).strip().title()

        transit_mode = None
        if "bus" in query_lower:
            transit_mode = "bus"
        elif "train" in query_lower:
            transit_mode = "train"
        elif "car" in query_lower:
            transit_mode = "car"
        elif "ev" in query_lower or "electric" in query_lower:
            transit_mode = "ev"

        plug_type = None
        if "ccs2" in query_lower or "ccs" in query_lower:
            plug_type = "CCS2"
        elif "type 2" in query_lower or "type2" in query_lower:
            plug_type = "Type 2"
        elif "chademo" in query_lower:
            plug_type = "CHAdeMO"

        vehicle_type = "car"
        if "bike" in query_lower or "motorcycle" in query_lower:
            vehicle_type = "motorcycle"
        elif "van" in query_lower:
            vehicle_type = "van"

        return MobilityEntities(
            locations=extracted_locs,
            origin=origin,
            destination=destination,
            transit_mode=transit_mode,
            plug_type=plug_type,
            vehicle_type=vehicle_type,
        )
