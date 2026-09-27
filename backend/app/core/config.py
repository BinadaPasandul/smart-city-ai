"""Application settings loaded from environment variables and .env."""

from functools import lru_cache

from pydantic import Field
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
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    agent_execution_timeout_seconds: float = Field(default=10.0, gt=0)

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

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
