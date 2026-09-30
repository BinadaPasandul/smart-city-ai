"""Unit tests for MobilityAgent, MobilityIRService, and MobilityNLPAnalyzer."""

import pytest
from app.agents.contracts import AgentErrorCode, AgentRequest
from app.agents.mobility.agent import MobilityAgent
from app.ir.mobility_ir import MobilityIRService
from app.nlp.mobility_nlp import MobilityIntent, MobilityNLPAnalyzer


def test_nlp_intent_classification():
    analyzer = MobilityNLPAnalyzer()

    intent, entities = analyzer.analyze("What is the traffic like on Galle Road?")
    assert intent == MobilityIntent.TRAFFIC_CHECK
    assert "Galle Road" in entities.locations

    intent, entities = analyzer.analyze("Where can I park near Colombo Fort?")
    assert intent == MobilityIntent.PARKING_SEARCH
    assert "Colombo Fort" in entities.locations

    intent, entities = analyzer.analyze("Are there CCS2 EV charging ports in Kollupitiya?")
    assert intent == MobilityIntent.EV_CHARGING
    assert "Kollupitiya" in entities.locations
    assert entities.plug_type == "CCS2"

    intent, entities = analyzer.analyze("What is the bus schedule for route 138?")
    assert intent == MobilityIntent.PUBLIC_TRANSIT
    assert entities.transit_mode == "bus"

    intent, entities = analyzer.analyze("How do I get from Colombo Fort to Kandy?")
    assert intent == MobilityIntent.ROUTE_PLANNING
    assert entities.origin == "Colombo Fort"
    assert entities.destination == "Kandy"


def test_ir_service_search():
    ir = MobilityIRService()

    traffic = ir.search_traffic("Galle Road")
    assert len(traffic) > 0
    assert "Galle Road" in traffic[0]["corridor"]

    buses = ir.search_buses(route_number="138")
    assert len(buses) == 1
    assert buses[0]["route_number"] == "138"

    trains = ir.search_trains(origin="Colombo Fort", destination="Matara")
    assert len(trains) > 0
    assert trains[0]["destination"] == "Matara"

    parking = ir.search_parking("Colombo Fort")
    assert len(parking) > 0
    assert "Colombo Fort" in parking[0]["location"]

    ev = ir.search_ev_charging("Kollupitiya", plug_type="CCS2")
    assert len(ev) > 0
    assert any("CCS2" in p for p in ev[0]["plug_types"])


@pytest.mark.asyncio
async def test_mobility_agent_execution_success():
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
async def test_mobility_agent_ev_query():
    agent = MobilityAgent()
    request = AgentRequest(query="I need an EV charger in Kollupitiya with Type 2 plug")

    response = await agent.execute(request)
    assert response.success is True
    assert "EV Charging" in response.answer
    assert "ChargeNET" in response.answer or "Kollupitiya" in response.answer


@pytest.mark.asyncio
async def test_mobility_agent_empty_query():
    agent = MobilityAgent()
    request = AgentRequest(query="   ")

    response = await agent.execute(request)
    assert response.success is False
    assert response.error is not None
    assert response.error.code == AgentErrorCode.INVALID_REQUEST
