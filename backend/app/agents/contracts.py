"""Typed in-process request and response contracts shared by all agents."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentErrorCode(str, Enum):
    """Stable error categories that callers can handle programmatically."""

    INVALID_REQUEST = "invalid_request"
    AGENT_NOT_FOUND = "agent_not_found"
    AGENT_EXECUTION_FAILED = "agent_execution_failed"
    TIMEOUT = "timeout"
    UNSUPPORTED_REQUEST = "unsupported_request"
    NEEDS_CLARIFICATION = "needs_clarification"


class AgentError(BaseModel):
    """Structured error information returned when an agent fails."""

    code: AgentErrorCode
    message: str


class AgentRequest(BaseModel):
    """Generic user request and shared context passed between agents."""

    request_id: UUID = Field(default_factory=uuid4)
    query: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)


class AgentSource(BaseModel):
    """Evidence reference that can represent web, API, or internal sources."""

    name: str
    source_type: str
    url: str | None = None
    retrieved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Standard result envelope returned by every agent."""

    model_config = ConfigDict(use_enum_values=True)

    request_id: UUID
    agent_name: str
    success: bool
    answer: str = ""
    sources: list[AgentSource] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: AgentError | None = None
