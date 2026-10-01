"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Shared settings for the backend process."""

    app_name: str = "Smart City AI Backend"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    database_url: str | None = None
    test_database_url: str | None = None
    auth_enabled: bool = False
    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_access_token_expire_minutes: int = Field(default=60, ge=1, le=10080)
    auth_rate_limit_requests: int = Field(default=10, ge=1)
    auth_rate_limit_window_seconds: int = Field(default=60, ge=1)
    agent_execution_timeout_seconds: float = Field(default=10.0, gt=0)
    chat_max_message_length: int = Field(default=5000, ge=1, le=50000)
    chat_rate_limit_requests: int = Field(default=20, ge=1)
    chat_rate_limit_window_seconds: int = Field(default=60, ge=1)
    web_search_enabled: bool = False
    web_search_provider: str = "tavily"
    tavily_api_key: SecretStr | None = None
    web_search_timeout_seconds: float = Field(default=8.0, gt=0)
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    web_search_query_max_length: int = Field(default=500, ge=1, le=5000)
    web_search_on_partial_failure: bool = True

    nlp_enabled: bool = True
    nlp_spacy_model: str = "en_core_web_sm"
    nlp_local_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    nlp_gemini_fallback_enabled: bool = True

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @model_validator(mode="after")
    def validate_security_configuration(self) -> "Settings":
        if self.jwt_algorithm != "HS256":
            raise ValueError("JWT_ALGORITHM must be HS256")
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS must list explicit origins; wildcard is not allowed")
        if self.web_search_provider != "tavily":
            raise ValueError("WEB_SEARCH_PROVIDER must be tavily")
        if self.web_search_enabled and not (
            self.tavily_api_key and self.tavily_api_key.get_secret_value().strip()
        ):
            raise ValueError("TAVILY_API_KEY is required when WEB_SEARCH_ENABLED is true")
        if self.auth_enabled:
            secret = self.jwt_secret.get_secret_value() if self.jwt_secret else ""
            if len(secret) < 32:
                raise ValueError("JWT_SECRET must contain at least 32 characters when AUTH_ENABLED is true")
            normalized_secret = secret.lower().replace("-", "").replace("_", "")
            if len(set(secret)) < 6 or any(
                marker in normalized_secret
                for marker in ("changeme", "changeit", "yoursecret", "secretkey", "password")
            ):
                raise ValueError("JWT_SECRET must not be an obvious placeholder or repetitive value")
            if not self.jwt_issuer or not self.jwt_issuer.strip():
                raise ValueError("JWT_ISSUER is required when AUTH_ENABLED is true")
            if not self.jwt_audience or not self.jwt_audience.strip():
                raise ValueError("JWT_AUDIENCE is required when AUTH_ENABLED is true")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings object."""
    return Settings()


settings = get_settings()
