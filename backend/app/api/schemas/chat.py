"""Stable public DTOs for the chat HTTP boundary."""

from datetime import datetime
import json
import math
from typing import Any, ClassVar
import unicodedata
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.contracts import AgentErrorCode, AgentResponse, AgentSource
from app.core.config import settings


class ChatRequest(BaseModel):
    """Client chat input, intentionally separate from the internal AgentRequest."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=settings.chat_max_message_length)
    context: dict[str, Any] = Field(default_factory=dict)

    MAX_CONTEXT_KEYS: ClassVar[int] = 32
    MAX_CONTEXT_KEY_LENGTH: ClassVar[int] = 64
    MAX_CONTEXT_STRING_LENGTH: ClassVar[int] = 1024
    MAX_CONTEXT_DEPTH: ClassVar[int] = 4
    MAX_CONTEXT_LIST_ITEMS: ClassVar[int] = 100
    MAX_CONTEXT_BYTES: ClassVar[int] = 8192

    @field_validator("message")
    @classmethod
    def trim_and_require_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        if any(
            unicodedata.category(char) == "Cc" and char not in "\t\r\n"
            for char in value
        ):
            raise ValueError("message contains unsupported control characters")
        return value

    @field_validator("context")
    @classmethod
    def validate_untrusted_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > cls.MAX_CONTEXT_KEYS:
            raise ValueError("context contains too many keys")

        def validate_value(item: Any, depth: int) -> None:
            if depth > cls.MAX_CONTEXT_DEPTH:
                raise ValueError("context is nested too deeply")
            if isinstance(item, str):
                if len(item) > cls.MAX_CONTEXT_STRING_LENGTH:
                    raise ValueError("context string value is too long")
            elif isinstance(item, dict):
                if len(item) > cls.MAX_CONTEXT_KEYS:
                    raise ValueError("nested context contains too many keys")
                for key, nested in item.items():
                    if not isinstance(key, str) or not key or len(key) > cls.MAX_CONTEXT_KEY_LENGTH:
                        raise ValueError("context keys must be nonempty strings of at most 64 characters")
                    validate_value(nested, depth + 1)
            elif isinstance(item, list):
                if len(item) > cls.MAX_CONTEXT_LIST_ITEMS:
                    raise ValueError("context list contains too many items")
                for nested in item:
                    validate_value(nested, depth + 1)
            elif isinstance(item, float) and not math.isfinite(item):
                raise ValueError("context numbers must be finite")
            elif item is None or isinstance(item, (bool, int, float)):
                return
            else:
                raise ValueError("context values must be JSON-compatible")

        for key, item in value.items():
            if not key or len(key) > cls.MAX_CONTEXT_KEY_LENGTH:
                raise ValueError("context keys must be nonempty strings of at most 64 characters")
            validate_value(item, 1)
        try:
            serialized = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("context must contain only JSON-compatible values") from exc
        if len(serialized) > cls.MAX_CONTEXT_BYTES:
            raise ValueError("context is too large")
        return value


class ChatSource(BaseModel):
    """Public citation fields mapped from specialist source metadata."""

    name: str
    source_type: str
    url: str | None = None
    retrieved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_agent_source(cls, source: AgentSource) -> "ChatSource":
        return cls(**source.model_dump())


class ChatError(BaseModel):
    """Safe structured API error; never contains internal exception objects."""

    code: AgentErrorCode
    message: str


class ChatMetadata(BaseModel):
    """Allowlisted orchestration metadata exposed to clients."""

    routing_method: str | None = None
    selected_agents: list[str] = Field(default_factory=list)
    execution_status: str | None = None
    successful_agents: list[str] = Field(default_factory=list)
    failed_agents: list[str] = Field(default_factory=list)
    synthesis_method: str | None = None
    synthesis_used_agents: list[str] = Field(default_factory=list)
    synthesis_limitations: list[str] = Field(default_factory=list)

    @classmethod
    def from_agent_metadata(cls, metadata: dict[str, Any]) -> "ChatMetadata":
        allowed = {
            "routing_method",
            "selected_agents",
            "execution_status",
            "successful_agents",
            "failed_agents",
            "synthesis_method",
            "synthesis_used_agents",
            "synthesis_limitations",
        }
        return cls(**{key: value for key, value in metadata.items() if key in allowed})


class ChatResponse(BaseModel):
    """Frontend-friendly response independent of internal orchestration models."""

    request_id: UUID
    success: bool
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    metadata: ChatMetadata = Field(default_factory=ChatMetadata)
    error: ChatError | None = None

    @classmethod
    def from_agent_response(cls, response: AgentResponse) -> "ChatResponse":
        return cls(
            request_id=response.request_id,
            success=response.success,
            answer=response.answer,
            sources=[ChatSource.from_agent_source(source) for source in response.sources],
            metadata=ChatMetadata.from_agent_metadata(response.metadata),
            error=(
                ChatError(code=response.error.code, message=response.error.message)
                if response.error
                else None
            ),
        )
