"""Offline account, hashing, token, and public HTTP contract tests."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.exc import OperationalError

import app.security.auth as jwt_verifier
from app.api.dependencies import get_auth_service
from app.api.schemas.auth import LoginRequest, RegisterRequest, UserPublic
from app.core.config import Settings
from app.main import app
from app.repositories.user_repository import DuplicateAccountError, normalize_email
from app.security.password import PasswordHasher
from app.security.rate_limit import InMemoryRateLimiter
from app.security.token import JwtIssuer
from app.services.auth_service import AuthService, InvalidCredentialsError

TEST_SECRET = "phase-65-unit-test-jwt-secret-that-is-long-enough"
RAW_PASSWORD = "Correct horse battery café 42"


def auth_settings(**overrides):
    values = dict(
        auth_enabled=True, jwt_secret=SecretStr(TEST_SECRET), jwt_algorithm="HS256",
        jwt_issuer="phase-65-tests", jwt_audience="phase-65-api",
        jwt_access_token_expire_minutes=60,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakeRepository:
    def __init__(self):
        self.users = {}
        self.created = []

    async def create(self, *, email, password_hash, display_name):
        email = normalize_email(email)
        if email in self.users:
            raise DuplicateAccountError()
        user = SimpleNamespace(
            id=uuid4(), email=email, password_hash=password_hash,
            display_name=display_name, is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        self.users[email] = user
        self.created.append(user)
        return user

    async def get_by_email(self, email):
        return self.users.get(normalize_email(email))

    async def get_by_id(self, user_id):
        return next((user for user in self.users.values() if user.id == user_id), None)


def test_argon2id_hashing_is_salted_and_verifies():
    hasher = PasswordHasher()
    first = hasher.hash(RAW_PASSWORD)
    second = hasher.hash(RAW_PASSWORD)
    assert first != second != RAW_PASSWORD
    assert first.startswith("$argon2id$")
    assert hasher.verify(RAW_PASSWORD, first)
    assert not hasher.verify("wrong password", first)
    assert not hasher.verify(RAW_PASSWORD, "invalid stored hash")


def test_email_normalization_is_consistent():
    assert normalize_email("  User@Example.COM  ") == "user@example.com"
    assert RegisterRequest(email="User@Example.COM", password=RAW_PASSWORD).email == "user@example.com"
    assert LoginRequest(email="USER@example.com", password="x").email == "user@example.com"


@pytest.mark.parametrize("payload", [
    {"email": "bad-email", "password": RAW_PASSWORD},
    {"email": "user@example.com", "password": "too-short"},
    {"email": "user@example.com", "password": "x" * 129},
    {"email": "user@example.com", "password": RAW_PASSWORD, "unexpected": True},
    {"email": "user@example.com", "password": RAW_PASSWORD, "display_name": " "},
])
def test_registration_validation_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        RegisterRequest(**payload)


def test_public_user_model_never_contains_hash_or_raw_password():
    user = SimpleNamespace(
        id=uuid4(), email="user@example.com", display_name="Citizen",
        is_active=True, created_at=datetime.now(timezone.utc), password_hash="sensitive-hash",
    )
    public = UserPublic.model_validate(user).model_dump_json()
    assert "password_hash" not in public
    assert "sensitive-hash" not in public
    assert RAW_PASSWORD not in public


@pytest.mark.asyncio
async def test_register_hashes_before_repository_create_and_rejects_duplicate():
    repository = FakeRepository()
    service = AuthService(repository, PasswordHasher(), JwtIssuer(auth_settings()))
    user = await service.register(email=" USER@Example.com ", password=RAW_PASSWORD, display_name="Citizen")
    assert user.email == "user@example.com"
    assert user.password_hash != RAW_PASSWORD
    assert user.password_hash.startswith("$argon2id$")
    assert await repository.get_by_id(user.id) is user
    with pytest.raises(DuplicateAccountError):
        await service.register(email="user@example.COM", password=RAW_PASSWORD, display_name=None)


@pytest.mark.asyncio
async def test_login_uses_same_normalized_email_and_rejects_invalid_accounts():
    repository = FakeRepository()
    settings = auth_settings()
    service = AuthService(repository, PasswordHasher(), JwtIssuer(settings))
    user = await service.register(email="User@Example.COM", password=RAW_PASSWORD, display_name=None)
    token, expiry = await service.login(email=" USER@example.com ", password=RAW_PASSWORD)
    assert expiry == 3600
    claims = jwt.decode(token, TEST_SECRET, algorithms=["HS256"], issuer=settings.jwt_issuer, audience=settings.jwt_audience)
    assert claims["sub"] == str(user.id)
    for email, password in (("user@example.com", "wrong"), ("absent@example.com", RAW_PASSWORD)):
        with pytest.raises(InvalidCredentialsError):
            await service.login(email=email, password=password)
    user.is_active = False
    with pytest.raises(InvalidCredentialsError):
        await service.login(email="user@example.com", password=RAW_PASSWORD)


def test_issued_jwt_has_minimal_claims_and_existing_verifier_accepts_it(monkeypatch):
    settings = auth_settings(jwt_access_token_expire_minutes=3)
    issuer = JwtIssuer(settings)
    user_id = uuid4()
    token = issuer.issue(user_id)
    claims = jwt.decode(token, TEST_SECRET, algorithms=["HS256"], issuer=settings.jwt_issuer, audience=settings.jwt_audience)
    assert set(claims) == {"sub", "iss", "aud", "iat", "exp", "jti"}
    assert claims["sub"] == str(user_id)
    assert claims["exp"] - claims["iat"] == 180
    assert RAW_PASSWORD not in token
    monkeypatch.setattr(jwt_verifier, "get_settings", lambda: settings)
    principal = jwt_verifier.get_current_principal(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    )
    assert principal.subject == str(user_id)


def test_token_expiry_configuration_is_positive():
    with pytest.raises(ValueError):
        auth_settings(jwt_access_token_expire_minutes=0)
    with pytest.raises(ValueError):
        auth_settings(auth_rate_limit_requests=0)
    with pytest.raises(ValueError):
        auth_settings(auth_rate_limit_window_seconds=0)


def test_existing_verifier_rejects_expired_issued_claims(monkeypatch):
    settings = auth_settings()
    expired = jwt.encode(
        {
            "sub": str(uuid4()), "iss": settings.jwt_issuer, "aud": settings.jwt_audience,
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        },
        TEST_SECRET, algorithm="HS256",
    )
    monkeypatch.setattr(jwt_verifier, "get_settings", lambda: settings)
    with pytest.raises(HTTPException) as caught:
        jwt_verifier.get_current_principal(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=expired)
        )
    assert caught.value.status_code == 401


@pytest.fixture
def auth_client():
    original = app.dependency_overrides.get(get_auth_service)
    old_limiter = app.state.auth_rate_limiter
    repository = FakeRepository()
    service = AuthService(repository, PasswordHasher(), JwtIssuer(auth_settings()))
    app.dependency_overrides[get_auth_service] = lambda: service
    app.state.auth_rate_limiter = InMemoryRateLimiter(20, 60)
    try:
        with TestClient(app) as client:
            yield client, repository
    finally:
        if original is None:
            app.dependency_overrides.pop(get_auth_service, None)
        else:
            app.dependency_overrides[get_auth_service] = original
        app.state.auth_rate_limiter = old_limiter


def test_register_and_login_http_are_public_and_safe(auth_client, caplog):
    client, repository = auth_client
    registered = client.post("/api/v1/auth/register", json={
        "email": "  USER@Example.com ", "password": RAW_PASSWORD, "display_name": " Citizen ",
    })
    assert registered.status_code == 201
    assert registered.json()["email"] == "user@example.com"
    assert registered.json()["display_name"] == "Citizen"
    assert "password_hash" not in registered.text and RAW_PASSWORD not in registered.text
    assert repository.users["user@example.com"].password_hash != RAW_PASSWORD
    duplicate = client.post("/api/v1/auth/register", json={
        "email": "user@example.COM", "password": RAW_PASSWORD,
    })
    assert duplicate.status_code == 409
    good = client.post("/api/v1/auth/login", json={
        "email": "USER@example.com", "password": RAW_PASSWORD,
    })
    assert good.status_code == 200
    assert good.json()["token_type"] == "bearer" and good.json()["expires_in"] == 3600
    assert "password_hash" not in good.text and RAW_PASSWORD not in good.text
    wrong = client.post("/api/v1/auth/login", json={"email": "user@example.com", "password": "wrong"})
    unknown = client.post("/api/v1/auth/login", json={"email": "absent@example.com", "password": "wrong"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert wrong.headers["www-authenticate"] == "Bearer"
    assert registered.headers["x-request-id"] and good.headers["x-request-id"]
    assert registered.headers["x-content-type-options"] == "nosniff"
    assert registered.headers["referrer-policy"] == "no-referrer"
    assert good.headers["cache-control"] == "no-store"
    assert RAW_PASSWORD not in caplog.text and TEST_SECRET not in caplog.text


def test_auth_http_validation_openapi_and_rate_limit(auth_client):
    client, _ = auth_client
    for payload in (
        {"email": "invalid", "password": RAW_PASSWORD},
        {"email": "user@example.com", "password": "short"},
        {"email": "user@example.com", "password": "x" * 129},
        {"email": "user@example.com", "password": RAW_PASSWORD, "extra": 1},
    ):
        assert client.post("/api/v1/auth/register", json=payload).status_code == 422
    document = app.openapi()
    for path in ("/api/v1/auth/register", "/api/v1/auth/login"):
        assert document["paths"][path]["post"].get("security") is None
    old_limiter = app.state.auth_rate_limiter
    app.state.auth_rate_limiter = InMemoryRateLimiter(1, 60)
    try:
        first = client.post("/api/v1/auth/login", json={"email": "absent@example.com", "password": "wrong"})
        second = client.post("/api/v1/auth/login", json={"email": "absent@example.com", "password": "wrong"})
    finally:
        app.state.auth_rate_limiter = old_limiter
    assert first.status_code == 401 and second.status_code == 429


def test_missing_database_configuration_returns_safe_503():
    previous = app.state.db_manager
    app.state.db_manager = None
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/auth/register", json={
                "email": "user@example.com", "password": RAW_PASSWORD,
            })
    finally:
        app.state.db_manager = previous
    assert response.status_code == 503
    assert "DATABASE_URL" not in response.text


def test_injected_database_failure_is_safe_for_register_and_login(caplog):
    class FailingService:
        async def register(self, **kwargs):
            raise OperationalError("sensitive-db-url", None, Exception("db-password-test-value"))

        async def login(self, **kwargs):
            raise OperationalError("sensitive-db-url", None, Exception("db-password-test-value"))

    previous = app.dependency_overrides.get(get_auth_service)
    old_limiter = app.state.auth_rate_limiter
    app.dependency_overrides[get_auth_service] = lambda: FailingService()
    app.state.auth_rate_limiter = InMemoryRateLimiter(20, 60)
    try:
        with TestClient(app) as client:
            register = client.post("/api/v1/auth/register", json={
                "email": "user@example.com", "password": RAW_PASSWORD,
            })
            login = client.post("/api/v1/auth/login", json={
                "email": "user@example.com", "password": RAW_PASSWORD,
            })
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_auth_service, None)
        else:
            app.dependency_overrides[get_auth_service] = previous
        app.state.auth_rate_limiter = old_limiter
    assert register.status_code == login.status_code == 503
    for secret in ("sensitive-db-url", "db-password-test-value", RAW_PASSWORD):
        assert secret not in register.text + login.text + caplog.text
