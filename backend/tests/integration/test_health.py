from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_backend_status() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "smart-city-ai-backend",
    }


def test_application_imports() -> None:
    assert app.title == "Smart City AI Backend"
