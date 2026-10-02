"""Minimal identity representation for successfully verified access tokens."""

from pydantic import BaseModel, ConfigDict, Field


class AuthenticatedPrincipal(BaseModel):
    """Validated identity claims only; raw JWTs and unrelated claims are omitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(min_length=1)
    issuer: str
    token_id: str | None = None
