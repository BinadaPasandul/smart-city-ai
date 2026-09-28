"""Stable public DTOs for the chat HTTP boundary."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.contracts import AgentErrorCode, AgentResponse, AgentSource
from app.core.config import settings


class ChatRequest(BaseModel):
    """Client chat input, intentionally separate from the internal AgentRequest."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=settings.chat_max_message_length)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def trim_and_require_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
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
