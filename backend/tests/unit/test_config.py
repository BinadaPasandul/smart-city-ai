from app.core.config import Settings
from pydantic import SecretStr
import pytest


def test_optional_service_settings_have_safe_defaults(monkeypatch) -> None:
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "DATABASE_URL", "JWT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    config = Settings(_env_file=None)

    assert config.gemini_api_key is None
    assert config.database_url is None
    assert config.jwt_secret is None
    assert config.api_v1_prefix == "/api/v1"
    assert config.agent_execution_timeout_seconds == 10.0
    assert config.gemini_model == "gemini-3.5-flash-lite"
    assert config.chat_max_message_length == 5000
    assert config.auth_enabled is False
    assert config.jwt_algorithm == "HS256"
    assert config.chat_rate_limit_requests == 20
    assert config.chat_rate_limit_window_seconds == 60


def test_gemini_model_can_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "test-model-id")

    config = Settings(_env_file=None)

    assert config.gemini_model == "test-model-id"


def test_chat_message_limit_can_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CHAT_MAX_MESSAGE_LENGTH", "6000")

    config = Settings(_env_file=None)

    assert config.chat_max_message_length == 6000


def test_auth_disabled_allows_startup_without_jwt_secret() -> None:
    config = Settings(_env_file=None, auth_enabled=False, jwt_secret=None)
    assert config.jwt_secret is None


def test_auth_enabled_requires_strong_secret_issuer_and_audience() -> None:
    with pytest.raises(ValueError, match="32 characters"):
        Settings(_env_file=None, auth_enabled=True, jwt_secret="weak", jwt_issuer="issuer", jwt_audience="aud")
    with pytest.raises(ValueError, match="JWT_ISSUER"):
        Settings(_env_file=None, auth_enabled=True, jwt_secret=SecretStr("Strong-Test-Secret-Value-1234567890"), jwt_audience="aud")
    with pytest.raises(ValueError, match="JWT_AUDIENCE"):
        Settings(_env_file=None, auth_enabled=True, jwt_secret=SecretStr("Strong-Test-Secret-Value-1234567890"), jwt_issuer="issuer")


def test_unsupported_jwt_algorithm_and_wildcard_cors_are_rejected() -> None:
    with pytest.raises(ValueError, match="HS256"):
        Settings(_env_file=None, jwt_algorithm="none")
    with pytest.raises(ValueError, match="wildcard"):
        Settings(_env_file=None, cors_origins=["*"])


def test_rate_limit_settings_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, chat_rate_limit_requests=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, chat_rate_limit_window_seconds=0)
