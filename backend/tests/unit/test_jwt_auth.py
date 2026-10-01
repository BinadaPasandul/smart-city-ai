from datetime import datetime, timedelta, timezone

import pytest

jwt = pytest.importorskip("jwt", reason="PyJWT is installed from backend/requirements.txt")

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr

import app.security.auth as auth
from app.core.config import Settings

TEST_SECRET = "test-jwt-secret-value-that-is-at-least-32-chars"
FAKE_GEMINI_KEY = "fake-gemini-key-for-secret-leak-tests"


def _settings(**overrides) -> Settings:
    values = {
        "auth_enabled": True,
        "jwt_secret": SecretStr(TEST_SECRET),
        "jwt_algorithm": "HS256",
        "jwt_issuer": "smart-city-tests",
        "jwt_audience": "smart-city-tests-api",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _token(**claims) -> str:
    payload = {
        "sub": "citizen-123",
        "iss": "smart-city-tests",
        "aud": "smart-city-tests-api",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        "jti": "token-id-1",
    }
    payload.update(claims)
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


def _credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_valid_signed_jwt_returns_minimal_principal_without_raw_token(monkeypatch) -> None:
    token = _token()
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())

    principal = auth.get_current_principal(_credentials(token))

    assert principal.subject == "citizen-123"
    assert principal.issuer == "smart-city-tests"
    assert principal.token_id == "token-id-1"
    assert token not in str(principal.model_dump())
    assert "raw_token" not in auth.AuthenticatedPrincipal.model_fields


@pytest.mark.parametrize("token", ["not-a-jwt", "", "a.b.c"])
def test_malformed_token_is_safe_401(token: str, monkeypatch) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    with pytest.raises(HTTPException) as caught:
        auth.get_current_principal(_credentials(token))
    assert caught.value.status_code == 401
    assert caught.value.headers["WWW-Authenticate"] == "Bearer"
    if token:
        assert token not in str(caught.value.detail)


def test_missing_bearer_token_is_401(monkeypatch) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    with pytest.raises(HTTPException) as caught:
        auth.get_current_principal(None)
    assert caught.value.status_code == 401
    assert caught.value.headers["WWW-Authenticate"] == "Bearer"


def test_invalid_signature_is_rejected(monkeypatch) -> None:
    token = jwt.encode(
        {"sub": "citizen", "iss": "smart-city-tests", "aud": "smart-city-tests-api", "exp": datetime.now(timezone.utc) + timedelta(minutes=2)},
        "wrong-signature-key-that-is-also-long-enough",
        algorithm="HS256",
    )
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    with pytest.raises(HTTPException) as caught:
        auth.get_current_principal(_credentials(token))
    assert caught.value.status_code == 401


def test_expired_wrong_issuer_and_wrong_audience_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    expired = _token(exp=datetime.now(timezone.utc) - timedelta(minutes=1))
    wrong_issuer = _token(iss="other-issuer")
    wrong_audience = _token(aud="other-audience")
    for token in (expired, wrong_issuer, wrong_audience):
        with pytest.raises(HTTPException) as caught:
            auth.get_current_principal(_credentials(token))
        assert caught.value.status_code == 401


def test_missing_subject_and_unsigned_or_unsupported_algorithms_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    missing_sub = _token(sub=None)
    unsigned = jwt.encode(
        {"iss": "smart-city-tests", "aud": "smart-city-tests-api", "exp": datetime.now(timezone.utc) + timedelta(minutes=2)},
        key="",
        algorithm="none",
    )
    for token in (missing_sub, unsigned):
        with pytest.raises(HTTPException) as caught:
            auth.get_current_principal(_credentials(token))
        assert caught.value.status_code == 401


def test_auth_disabled_needs_no_jwt_secret(monkeypatch) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(_env_file=None, auth_enabled=False))
    assert auth.get_current_principal(None) is None
