from app.core.config import Settings


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


def test_gemini_model_can_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "test-model-id")

    config = Settings(_env_file=None)

    assert config.gemini_model == "test-model-id"
