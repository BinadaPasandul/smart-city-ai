"""Validated web snippets and a provider-neutral asynchronous search contract."""

from datetime import datetime, timezone
import ipaddress
import re
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agents.contracts import AgentSource


class WebSearchResult(BaseModel):
    """One bounded, safe-to-display search snippet; result pages are never fetched."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    url: str
    snippet: str = Field(min_length=1, max_length=2000)
    source_domain: str = ""
    provider: str = Field(min_length=1, max_length=40)
    relevance_score: float | None = Field(default=None, ge=0, le=1)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_result(self) -> "WebSearchResult":
        self.title = self.title.strip()
        self.snippet = self.snippet.strip()
        if not self.title or not self.snippet:
            raise ValueError("search title and snippet must contain text")
        if len(self.url) > 2048 or any(char.isspace() or ord(char) < 32 for char in self.url):
            raise ValueError("invalid search URL")
        try:
            parsed = urlsplit(self.url)
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid search URL") from exc
        if parsed.scheme != "https" or not host or parsed.username or parsed.password:
            raise ValueError("search URL must be HTTPS without credentials")
        if "@" in parsed.netloc or "%" in host or "\\" in self.url:
            raise ValueError("invalid search URL host")
        host = host.lower().rstrip(".")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ValueError("local search URL is not allowed")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if all(label.isdecimal() for label in host.split(".")):
                raise ValueError("numeric search URL host is not allowed")
            try:
                host = host.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ValueError("invalid search URL host") from exc
            if "." not in host or any(
                not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in host.split(".")
            ):
                raise ValueError("invalid search URL host")
        else:
            if not address.is_global:
                raise ValueError("private or reserved search URL is not allowed")
            host = f"[{address.compressed}]" if address.version == 6 else str(address)
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("invalid search URL port")
        netloc = host if port in (None, 443) else f"{host}:{port}"
        self.source_domain = host.strip("[]")
        self.url = urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))
        return self


class WebEvidence(BaseModel):
    """Supplemental search evidence kept distinct from specialist responses."""

    title: str
    snippet: str
    url: str
    domain: str
    provider: str
    retrieved_at: datetime

    @classmethod
    def from_result(cls, result: WebSearchResult) -> "WebEvidence":
        return cls(
            title=result.title,
            snippet=result.snippet,
            url=result.url,
            domain=result.source_domain,
            provider=result.provider,
            retrieved_at=result.retrieved_at,
        )

    def to_agent_source(self) -> AgentSource:
        return AgentSource(
            name=self.title,
            source_type="web_search",
            url=self.url,
            retrieved_at=self.retrieved_at,
            metadata={"provider": self.provider, "domain": self.domain},
        )


class WebSearchProvider(Protocol):
    """Provider-neutral asynchronous snippet retrieval."""

    provider_name: str

    async def search(self, query: str, max_results: int) -> list[WebSearchResult]: ...


class WebSearchProviderError(Exception):
    """Safe provider failure category without response bodies or credentials."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class TavilyWebSearchProvider:
    """Call Tavily Basic Search and keep only validated result snippets."""

    provider_name = "tavily"
    SEARCH_URL = "https://api.tavily.com/search"

    def __init__(
        self, api_key: str, *, client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Tavily API key is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int) -> list[WebSearchResult]:
        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")
        if self._api_key in query:
            raise WebSearchProviderError("unsafe_query")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await self._client.post(
                self.SEARCH_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                    "auto_parameters": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            category = "http_4xx" if 400 <= exc.response.status_code < 500 else "http_5xx"
            raise WebSearchProviderError(category) from None
        except httpx.TimeoutException:
            raise WebSearchProviderError("timeout") from None
        except httpx.RequestError:
            raise WebSearchProviderError("connection") from None
        except ValueError:
            raise WebSearchProviderError("invalid_json") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise WebSearchProviderError("invalid_schema")
        validated: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for raw in payload["results"]:
            if len(validated) >= max_results:
                break
            if not isinstance(raw, dict):
                continue
            try:
                title = raw.get("title", "")
                snippet = raw.get("content", "")
                result = WebSearchResult(
                    title=title[:200] if isinstance(title, str) else title,
                    url=raw.get("url", ""),
                    snippet=snippet[:2000] if isinstance(snippet, str) else snippet,
                    provider=self.provider_name,
                    relevance_score=raw.get("score"),
                )
            except (ValidationError, ValueError, TypeError):
                continue
            if result.url not in seen_urls:
                validated.append(result)
                seen_urls.add(result.url)
        return validated

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
