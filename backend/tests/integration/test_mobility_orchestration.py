"""Integration tests for MobilityAgent integrated into CityOrchestrator and FastAPI."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


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


def test_chat_endpoint_ev_query(client):
    response = client.post("/api/v1/chat", json={"message": "Find EV charging stations in Kollupitiya"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mobility" in data["metadata"]["selected_agents"]
    assert "EV Charging" in data["answer"]
