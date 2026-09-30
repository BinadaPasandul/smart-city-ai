from datetime import datetime, timedelta, timezone

import pytest

jwt = pytest.importorskip("jwt", reason="PyJWT is installed from backend/requirements.txt")

from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.security.auth as auth
from app.agents.contracts import AgentRequest, AgentResponse
from app.api.dependencies import get_orchestrator
from app.core.config import Settings
from app.main import app
from app.security.rate_limit import InMemoryRateLimiter

TEST_SECRET = "integration-test-secret-with-at-least-32-characters"
FAKE_GEMINI_KEY = "fake-gemini-key-for-auth-test"


def _auth_settings() -> Settings:
    return Settings(
        _env_file=None,
        auth_enabled=True,
        jwt_secret=SecretStr(TEST_SECRET),
        jwt_algorithm="HS256",
        jwt_issuer="smart-city-integration-tests",
        jwt_audience="smart-city-api-tests",
        chat_rate_limit_requests=1,
        chat_rate_limit_window_seconds=60,
    )


def _token(subject: str = "citizen-1", *, expires_in: timedelta = timedelta(minutes=5)) -> str:
    return jwt.encode(
        {
            "sub": subject,
            "iss": "smart-city-integration-tests",
            "aud": "smart-city-api-tests",
            "exp": datetime.now(timezone.utc) + expires_in,
        },
        TEST_SECRET,
        algorithm="HS256",
    )


class CountingOrchestrator:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            request_id=request.request_id,
            agent_name="mobility",
            success=True,
            answer="Traffic is flowing.",
        )


def test_authenticated_chat_flow_and_public_health(monkeypatch, caplog) -> None:
    monkeypatch.setattr(auth, "get_settings", _auth_settings)
    orchestrator = CountingOrchestrator()
    previous = app.dependency_overrides.get(get_orchestrator)
    old_limiter = app.state.rate_limiter
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.state.rate_limiter = InMemoryRateLimiter(1, 60)
    test_token = _token("user-a")
    other_user_token = _token("user-b")
    expired_token = _token("expired-user", expires_in=timedelta(minutes=-5))
    try:
        with TestClient(app) as client:
            health = client.get("/api/v1/health")
            missing = client.post("/api/v1/chat", json={"message": "traffic"})
            invalid = client.post(
                "/api/v1/chat", json={"message": "traffic"},
                headers={"Authorization": "Bearer malformed"},
            )
            expired = client.post(
                "/api/v1/chat", json={"message": "traffic"},
                headers={"Authorization": f"Bearer {expired_token}"},
            )
            valid = client.post(
                "/api/v1/chat",
                json={"message": "traffic", "context": {"user_id": "spoofed-user"}},
                headers={"Authorization": f"Bearer {test_token}"},
            )
            same_user_limited = client.post(
                "/api/v1/chat", json={"message": "traffic"},
                headers={"Authorization": f"Bearer {test_token}"},
            )
            other_user = client.post(
                "/api/v1/chat", json={"message": "traffic"},
                headers={"Authorization": f"Bearer {other_user_token}"},
            )
        assert health.status_code == 200
        assert missing.status_code == invalid.status_code == expired.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"
        assert missing.headers["x-request-id"]
        assert valid.status_code == other_user.status_code == 200
        assert same_user_limited.status_code == 429
        assert len(orchestrator.requests) == 2
        internal_request = orchestrator.requests[0]
        assert valid.json()["request_id"] == str(internal_request.request_id)
        assert valid.headers["x-request-id"] == valid.json()["request_id"]
        assert internal_request.context == {"user_context": {"user_id": "spoofed-user"}}
        assert TEST_SECRET not in valid.text
        assert FAKE_GEMINI_KEY not in valid.text
        assert test_token not in valid.text
        assert expired_token not in expired.text
        assert test_token not in caplog.text
        assert expired_token not in caplog.text
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_orchestrator, None)
        else:
            app.dependency_overrides[get_orchestrator] = previous
        app.state.rate_limiter = old_limiter


def test_openapi_documents_bearer_for_chat_but_not_health() -> None:
    document = app.openapi()
    operation = document["paths"]["/api/v1/chat"]["post"]
    assert operation["security"]
    assert document["paths"]["/api/v1/health"]["get"].get("security") is None


def test_unknown_or_invalid_bearer_does_not_reveal_test_secrets(monkeypatch) -> None:
    monkeypatch.setattr(auth, "get_settings", _auth_settings)
    for credentials in (None, "Basic abc", "Bearer nope"):
        request_headers = {"Authorization": credentials} if credentials else {}
        with TestClient(app) as client:
            response = client.post("/api/v1/chat", json={"message": "traffic"}, headers=request_headers)
        assert response.status_code == 401
        assert TEST_SECRET not in response.text
        assert FAKE_GEMINI_KEY not in response.text
