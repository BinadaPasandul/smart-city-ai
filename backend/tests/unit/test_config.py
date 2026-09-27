from app.core.config import Settings


def test_optional_service_settings_have_safe_defaults() -> None:
    config = Settings(_env_file=None)

    assert config.gemini_api_key is None
    assert config.database_url is None
    assert config.jwt_secret is None
    assert config.api_v1_prefix == "/api/v1"
