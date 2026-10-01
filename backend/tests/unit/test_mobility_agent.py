"""Unit tests for MobilityAgent, MobilityIRService, and MobilityNLPAnalyzer."""

import pytest
from app.agents.contracts import AgentErrorCode, AgentRequest
from app.agents.mobility.agent import MobilityAgent
from app.ir.mobility_ir import MobilityIRService
from app.nlp.mobility_nlp import MobilityIntent, MobilityNLPAnalyzer


def test_nlp_intent_classification():
    analyzer = MobilityNLPAnalyzer()

    # Traffic intent
    intent, entities = analyzer.analyze("What is the traffic like on Galle Road?")
    assert intent == MobilityIntent.TRAFFIC_CHECK
    assert "Galle Road" in entities.locations

    intent, entities = analyzer.analyze("Check traffic situation on Marine Drive")
    assert intent == MobilityIntent.TRAFFIC_CHECK
    assert "Marine Drive" in entities.locations

    # Parking intent
    intent, entities = analyzer.analyze("Where can I park near Colombo Fort?")
    assert intent == MobilityIntent.PARKING_SEARCH
    assert "Colombo Fort" in entities.locations

    intent, entities = analyzer.analyze("Is there parking available at Majestic City Bambalapitiya?")
    assert intent == MobilityIntent.PARKING_SEARCH
    assert "Bambalapitiya" in entities.locations

    # EV Charging intent
    intent, entities = analyzer.analyze("Are there CCS2 EV charging ports in Kollupitiya?")
    assert intent == MobilityIntent.EV_CHARGING
    assert "Kollupitiya" in entities.locations
    assert entities.plug_type == "CCS2"

    intent, entities = analyzer.analyze("Find an electric vehicle charging station near Katunayake Airport")
    assert intent == MobilityIntent.EV_CHARGING
    assert "Katunayake" in entities.locations or "Airport" in entities.locations

    # Public Transit intent (Bus & Train)
    intent, entities = analyzer.analyze("What is the bus schedule for route 138?")
    assert intent == MobilityIntent.PUBLIC_TRANSIT
    assert entities.transit_mode == "bus"

    intent, entities = analyzer.analyze("What time does Yal Devi train leave?")
    assert intent == MobilityIntent.PUBLIC_TRANSIT
    assert entities.transit_mode == "train"

    # Route Planning intent
    intent, entities = analyzer.analyze("How do I get from Colombo Fort to Kandy?")
    assert intent == MobilityIntent.ROUTE_PLANNING
    assert entities.origin == "Colombo Fort"
    assert entities.destination == "Kandy"


def test_ir_service_search_traffic():
    ir = MobilityIRService()

    # Search existing and newly added corridors
    galle_road = ir.search_traffic("Galle Road")
    assert len(galle_road) > 0
    assert "Galle Road" in galle_road[0]["corridor"]

    marine_drive = ir.search_traffic("Marine Drive")
    assert len(marine_drive) > 0
    assert "Marine Drive" in marine_drive[0]["corridor"]

    baseline_road = ir.search_traffic("Baseline Road")
    assert len(baseline_road) > 0
    assert "Baseline Road" in baseline_road[0]["corridor"]
    assert len(baseline_road[0]["incidents"]) > 0

    expressway = ir.search_traffic("Katunayake")
    assert len(expressway) > 0
    assert "Katunayake" in expressway[0]["corridor"]


def test_ir_service_search_buses():
    ir = MobilityIRService()

    # Search specific route numbers
    buses_138 = ir.search_buses(route_number="138")
    assert len(buses_138) == 1
    assert buses_138[0]["route_number"] == "138"

    # Search airport expressway bus
    buses_187 = ir.search_buses(route_number="187")
    assert len(buses_187) == 1
    assert "Airport" in buses_187[0]["name"]
    assert buses_187[0]["fare_lkr"] == 300

    # Search by key stop query
    buses_borella = ir.search_buses(query="Borella")
    assert len(buses_borella) >= 3


def test_ir_service_search_trains():
    ir = MobilityIRService()

    # Search Coastal Line trains
    coastal_trains = ir.search_trains(origin="Colombo Fort", destination="Matara")
    assert len(coastal_trains) > 0
    assert coastal_trains[0]["destination"] == "Matara"

    # Search Northern Line Yal Devi
    yal_devi = ir.search_trains(query="Yal Devi")
    assert len(yal_devi) == 1
    assert yal_devi[0]["train_name"] == "Yal Devi Express"
    assert "Jaffna" in yal_devi[0]["destination"]

    # Search Kelani Valley line commuter
    kv_commuter = ir.search_trains(query="KV Commuter")
    assert len(kv_commuter) >= 1
    assert any(t["train_name"] == "KV Commuter" for t in kv_commuter)
    assert any(t["destination"] == "Avissawella" for t in kv_commuter)



def test_ir_service_search_parking():
    ir = MobilityIRService()

    # Colombo Fort
    fort_parking = ir.search_parking("Colombo Fort")
    assert len(fort_parking) > 0
    assert "Colombo Fort" in fort_parking[0]["location"]

    # One Galle Face / Galle Face
    galle_face_parking = ir.search_parking("Galle Face")
    assert len(galle_face_parking) >= 2

    # Bambalapitiya Majestic City
    bamba_parking = ir.search_parking("Bambalapitiya")
    assert len(bamba_parking) > 0
    assert any("Majestic City" in p["name"] for p in bamba_parking)

    # Cinnamon Gardens Racecourse
    racecourse_parking = ir.search_parking("Cinnamon Gardens")
    assert len(racecourse_parking) > 0
    assert "Racecourse" in racecourse_parking[0]["name"]


def test_ir_service_search_ev_charging():
    ir = MobilityIRService()

    # Kollupitiya with CCS2
    kollupitiya_ev = ir.search_ev_charging("Kollupitiya", plug_type="CCS2")
    assert len(kollupitiya_ev) > 0
    assert any("CCS2" in p for p in kollupitiya_ev[0]["plug_types"])

    # Katunayake Airport Hub
    airport_ev = ir.search_ev_charging("Katunayake")
    assert len(airport_ev) > 0
    assert any("150 kW" in plug for plug in airport_ev[0]["plug_types"])

    # Dehiwala Fast Charger
    dehiwala_ev = ir.search_ev_charging("Dehiwala")
    assert len(dehiwala_ev) > 0
    assert "Vega Innovations" in dehiwala_ev[0]["provider"]


@pytest.mark.asyncio
async def test_mobility_agent_execution_parking():
    agent = MobilityAgent()
    request = AgentRequest(query="Where can I park near Colombo Fort?")

    response = await agent.execute(request)
    assert response.success is True
    assert response.agent_name == "mobility"
    assert "Parking Facilities" in response.answer
    assert len(response.sources) > 0
    assert response.metadata["intent"] == "parking_search"
    assert "responsible_ai_notice" in response.metadata


@pytest.mark.asyncio
async def test_mobility_agent_traffic_query():
    agent = MobilityAgent()
    request = AgentRequest(query="Is there traffic on Galle Road right now?")

    response = await agent.execute(request)
    assert response.success is True
    assert "Traffic & Corridor Status Report" in response.answer
    assert "Galle Road" in response.answer


@pytest.mark.asyncio
async def test_mobility_agent_marine_drive_traffic():
    agent = MobilityAgent()
    request = AgentRequest(query="What is the traffic situation on Marine Drive?")

    response = await agent.execute(request)
    assert response.success is True
    assert "Marine Drive" in response.answer
    assert "[NORMAL]" in response.answer


@pytest.mark.asyncio
async def test_mobility_agent_train_schedule():
    agent = MobilityAgent()
    request = AgentRequest(query="What time does Yal Devi train leave?")

    response = await agent.execute(request)
    assert response.success is True
    assert "Yal Devi Express" in response.answer
    assert "06:35" in response.answer
    assert "Jaffna" in response.answer


@pytest.mark.asyncio
async def test_mobility_agent_ev_query():
    agent = MobilityAgent()
    request = AgentRequest(query="I need an EV charger in Kollupitiya with Type 2 plug")

    response = await agent.execute(request)
    assert response.success is True
    assert "EV Charging" in response.answer
    assert "ChargeNET" in response.answer or "Kollupitiya" in response.answer


@pytest.mark.asyncio
async def test_mobility_agent_airport_ev():
    agent = MobilityAgent()
    request = AgentRequest(query="Are there fast EV chargers near Katunayake Airport?")

    response = await agent.execute(request)
    assert response.success is True
    assert "BIA Katunayake Airport EV Charging Hub" in response.answer
    assert "150 kW" in response.answer


@pytest.mark.asyncio
async def test_mobility_agent_empty_query():
    agent = MobilityAgent()
    request = AgentRequest(query="   ")

    response = await agent.execute(request)
    assert response.success is False
    assert response.error is not None
    assert response.error.code == AgentErrorCode.INVALID_REQUEST
