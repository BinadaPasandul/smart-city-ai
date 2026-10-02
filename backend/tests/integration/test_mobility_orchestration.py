"""Integration tests for MobilityAgent integrated into CityOrchestrator and FastAPI."""

import pytest
from fastapi.testclient import TestClient

from app.agents.orchestrator.router import DeterministicQueryRouter, SpecialistAgentName
from app.core.config import Settings
from app.main import app
from app.nlp.models import UnderstandingDecision
from app.nlp.pipeline import LocalNlpAnalyzer, RequestUnderstandingPipeline


@pytest.fixture
def client(monkeypatch):
    client = TestClient(app)
    # Keep integration coverage offline; the dedicated Marine Drive test below
    # replaces this with a fake structured Gemini fallback.
    monkeypatch.setattr(app.state.orchestrator, "_router", DeterministicQueryRouter())
    return client


def test_chat_endpoint_routes_to_mobility(client):
    response = client.post("/api/v1/chat", json={"message": "Where can I park near Colombo Fort?"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["metadata"]["selected_agents"] == ["mobility"]
    assert "Parking Facilities" in data["answer"]
    assert len(data["sources"]) > 0
    assert data["sources"][0]["metadata"]["category"] == "parking"


def test_chat_endpoint_traffic_query(client):
    response = client.post("/api/v1/chat", json={"message": "What is the traffic situation on Kandy Road?"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mobility" in data["metadata"]["selected_agents"]
    assert "Traffic & Corridor Status Report" in data["answer"]


def test_chat_endpoint_marine_drive_traffic_uses_offline_gemini_entity_fallback(client, monkeypatch):
    class NoLocations:
        def extract_locations(self, text):
            return []

    class NoTimes:
        def extract(self, text):
            return []

    class FakeGemini:
        calls = []

        async def understand(self, query, *, local_analysis=None, request_id=None):
            self.calls.append((query, local_analysis, request_id))
            return UnderstandingDecision(
                agent_names=[SpecialistAgentName.MOBILITY],
                locations=["Marine Drive"],
                temporal_expressions=[],
                confidence=0.96,
                reason="The request asks for traffic at a named road.",
                needs_clarification=False,
                missing_information=[],
            )

    gemini = FakeGemini()
    router = RequestUnderstandingPipeline(
        analyzer=LocalNlpAnalyzer(entity_extractor=NoLocations(), temporal_extractor=NoTimes()),
        gemini_router=gemini,
        settings=Settings(_env_file=None, gemini_api_key=None),
    )
    orchestrator = client.app.state.orchestrator
    monkeypatch.setattr(orchestrator, "_router", router)

    query = "Is there traffic congestion on Marine Drive?"
    response = client.post("/api/v1/chat", json={"message": query})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mobility" in data["metadata"]["selected_agents"]
    assert "Marine Drive" in data["answer"]
    assert data["metadata"]["routing_method"] == "gemini"
    assert data["metadata"]["gemini_understanding_fallback_used"] is True
    assert gemini.calls[0][0] == query
    assert gemini.calls[0][1]["reason_for_fallback"] == "location_specific_query_missing_location"
    assert gemini.calls[0][2] == data["request_id"]


def test_chat_endpoint_ev_query(client):
    response = client.post("/api/v1/chat", json={"message": "Find EV charging stations in Kollupitiya"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mobility" in data["metadata"]["selected_agents"]
    assert "EV Charging" in data["answer"]


def test_chat_endpoint_airport_ev_query(client):
    response = client.post("/api/v1/chat", json={"message": "Is there an EV charging station at Katunayake Airport?"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mobility" in data["metadata"]["selected_agents"]
    assert "Katunayake" in data["answer"]
    assert len(data["sources"]) > 0


def test_chat_endpoint_train_query(client):
    response = client.post("/api/v1/chat", json={"message": "What is the timetable for Yal Devi train to Jaffna?"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mobility" in data["metadata"]["selected_agents"]
    assert "Yal Devi Express" in data["answer"]
