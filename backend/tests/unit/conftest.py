import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def disable_live_gemini_for_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests offline even when a developer has a local API key."""
    monkeypatch.setattr(settings, "gemini_api_key", None)
