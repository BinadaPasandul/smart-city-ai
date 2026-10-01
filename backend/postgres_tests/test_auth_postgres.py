"""Real disposable PostgreSQL verification; requires TEST_DATABASE_URL."""

import asyncio
import os
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete, inspect
from sqlalchemy.engine import make_url

import app.api.dependencies as api_dependencies
import app.security.auth as jwt_verifier
from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.router import DeterministicQueryRouter
from app.agents.registry import AgentRegistry
from app.api.dependencies import get_orchestrator
from app.core.config import Settings
from app.db.models.user import User
from app.db.session import DatabaseManager
from app.main import app
from app.repositories.user_repository import DuplicateAccountError, UserRepository
from app.security.password import PasswordHasher
from app.security.rate_limit import InMemoryRateLimiter

pytestmark = pytest.mark.postgres
TEST_JWT_SECRET = "postgres-auth-verification-secret-with-strong-length"
TEST_PASSWORD = "PostgreSQL test password café 42"


@pytest.fixture
def test_database_url():
    raw = os.environ.get("TEST_DATABASE_URL", "")
    if not raw:
        pytest.fail("Set TEST_DATABASE_URL to a disposable local PostgreSQL test database")
    parsed = make_url(raw)
    if (
        parsed.drivername != "postgresql+asyncpg"
        or parsed.database != "smart_city_ai_test"
        or parsed.host not in ("localhost", "127.0.0.1", "::1")
    ):
        pytest.fail("TEST_DATABASE_URL must target local smart_city_ai_test via asyncpg")
    return raw


def auth_settings():
    return Settings(
        _env_file=None, auth_enabled=True, jwt_secret=SecretStr(TEST_JWT_SECRET),
        jwt_issuer="postgres-auth-test", jwt_audience="postgres-auth-api",
        jwt_access_token_expire_minutes=60,
    )


@pytest.mark.asyncio
async def test_repository_persists_hash_and_database_rejects_duplicate(test_database_url):
    manager = DatabaseManager(test_database_url)
    user_id = None
    email = f"phase65-{uuid4().hex}@example.com"
    try:
        async with manager.engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        assert "users" in tables and "alembic_version" in tables
        password_hash = PasswordHasher().hash(TEST_PASSWORD)
        async with manager.session_factory() as session:
            repository = UserRepository(session)
            user = await repository.create(
                email=email.upper(), password_hash=password_hash, display_name="Disposable test user",
            )
            user_id = user.id
            assert user.email == email
            assert user.password_hash.startswith("$argon2id$")
            assert TEST_PASSWORD not in user.password_hash
            assert user.is_active is True
            assert user.created_at.tzinfo is not None
            assert user.updated_at.tzinfo is not None
            assert (await repository.get_by_email(email.upper())).id == user.id
            assert (await repository.get_by_id(user.id)).email == email
            with pytest.raises(DuplicateAccountError):
                await repository.create(
                    email=email, password_hash=password_hash, display_name=None,
                )
            assert (await repository.get_by_email(email)).id == user.id
    finally:
        if user_id is not None:
            async with manager.session_factory() as session:
                await session.execute(delete(User).where(User.id == user_id))
                await session.commit()
        await manager.aclose()


class FakeMobilityAgent(BaseAgent):
    def __init__(self):
        super().__init__("mobility")
        self.requests: list[AgentRequest] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            request_id=request.request_id, agent_name=self.name,
            success=True, answer="Test mobility response.",
        )


def test_register_login_issued_jwt_chat_and_rate_isolation(test_database_url, monkeypatch, caplog):
    config = auth_settings()
    monkeypatch.setattr(api_dependencies, "get_settings", lambda: config)
    monkeypatch.setattr(jwt_verifier, "get_settings", lambda: config)
    manager = DatabaseManager(test_database_url)
    old_manager = app.state.db_manager
    old_chat_limiter = app.state.rate_limiter
    old_auth_limiter = app.state.auth_rate_limiter
    old_override = app.dependency_overrides.get(get_orchestrator)
    app.state.db_manager = manager
    app.state.rate_limiter = InMemoryRateLimiter(1, 60)
    app.state.auth_rate_limiter = InMemoryRateLimiter(20, 60)
    mobility = FakeMobilityAgent()
    registry = AgentRegistry()
    registry.register(mobility)
    orchestrator = CityOrchestratorAgent(registry, DeterministicQueryRouter())
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    emails = [f"phase65-{uuid4().hex}@example.com" for _ in range(2)]
    user_ids = []
    try:
        with TestClient(app) as client:
            assert client.get("/api/v1/health").status_code == 200
            assert client.post("/api/v1/chat", json={"message": "traffic"}).status_code == 401
            tokens = []
            for email in emails:
                register = client.post("/api/v1/auth/register", json={
                    "email": email.upper(), "password": TEST_PASSWORD,
                    "display_name": "Disposable",
                })
                assert register.status_code == 201
                body = register.json()
                assert body["email"] == email
                user_ids.append(body["id"])
                assert "password_hash" not in register.text and TEST_PASSWORD not in register.text
                login = client.post("/api/v1/auth/login", json={
                    "email": email, "password": TEST_PASSWORD,
                })
                assert login.status_code == 200
                token = login.json()["access_token"]
                tokens.append(token)
                assert login.json()["token_type"] == "bearer"
                claims = jwt.decode(
                    token, TEST_JWT_SECRET, algorithms=["HS256"],
                    issuer=config.jwt_issuer, audience=config.jwt_audience,
                )
                assert claims["sub"] == body["id"]
                assert TEST_PASSWORD not in register.text + login.text
            duplicate = client.post("/api/v1/auth/register", json={
                "email": emails[0].upper(), "password": TEST_PASSWORD,
            })
            assert duplicate.status_code == 409
            wrong = client.post("/api/v1/auth/login", json={
                "email": emails[0], "password": "wrong password",
            })
            unknown = client.post("/api/v1/auth/login", json={
                "email": f"unknown-{uuid4().hex}@example.com", "password": "wrong password",
            })
            assert wrong.status_code == unknown.status_code == 401
            assert wrong.json() == unknown.json()
            first = client.post(
                "/api/v1/chat", json={"message": "traffic"},
                headers={"Authorization": f"Bearer {tokens[0]}"},
            )
            repeated = client.post(
                "/api/v1/chat", json={"message": "traffic"},
                headers={"Authorization": f"Bearer {tokens[0]}"},
            )
            second_user = client.post(
                "/api/v1/chat", json={"message": "traffic"},
                headers={"Authorization": f"Bearer {tokens[1]}"},
            )
            assert first.status_code == second_user.status_code == 200
            assert repeated.status_code == 429
            assert first.json()["answer"] == "Test mobility response."
            assert first.headers["x-request-id"] == first.json()["request_id"]
            assert first.json()["request_id"] == str(mobility.requests[0].request_id)
            assert len(mobility.requests) == 2
            assert first.headers["cache-control"] == "no-store"
            assert first.headers["x-content-type-options"] == "nosniff"
            assert first.headers["referrer-policy"] == "no-referrer"
            for secret in (TEST_JWT_SECRET, TEST_PASSWORD, *tokens):
                assert secret not in first.text + register.text + duplicate.text + wrong.text
                assert secret not in caplog.text
        # Reopen a fresh engine because TestClient lifespan disposes the owned one.
        verify_manager = DatabaseManager(test_database_url)
        try:
            async def check_rows():
                async with verify_manager.session_factory() as session:
                    for user_id in user_ids:
                        user = await UserRepository(session).get_by_id(UUID(user_id))
                        assert user is not None
                        assert user.password_hash.startswith("$argon2id$")
                        assert TEST_PASSWORD not in user.password_hash
            asyncio.run(check_rows())
        finally:
            asyncio.run(verify_manager.aclose())
    finally:
        async def cleanup():
            cleanup_manager = DatabaseManager(test_database_url)
            try:
                async with cleanup_manager.session_factory() as session:
                    for user_id in user_ids:
                        await session.execute(delete(User).where(User.id == UUID(user_id)))
                    await session.commit()
            finally:
                await cleanup_manager.aclose()
        asyncio.run(cleanup())
        app.state.db_manager = old_manager
        app.state.rate_limiter = old_chat_limiter
        app.state.auth_rate_limiter = old_auth_limiter
        if old_override is None:
            app.dependency_overrides.pop(get_orchestrator, None)
        else:
            app.dependency_overrides[get_orchestrator] = old_override
