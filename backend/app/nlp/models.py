"""Typed results for local and fallback request understanding."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.orchestrator.router import MissingInformation, SpecialistAgentName


class NlpAnalysisResult(BaseModel):
    """Local extraction output; confidence is a heuristic score, not a probability."""

    model_config = ConfigDict(extra="forbid")

    original_text: str
    normalized_text: str
    locations: list[str] = Field(default_factory=list, max_length=10)
    temporal_expressions: list[str] = Field(default_factory=list, max_length=10)
    candidate_agents: list[SpecialistAgentName] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_information: list[MissingInformation] = Field(default_factory=list, max_length=3)
    local_nlp_available: bool


class UnderstandingDecision(BaseModel):
    """Structured Gemini fallback output; it cannot contain an answer or tool call."""

    model_config = ConfigDict(extra="forbid")

    agent_names: list[SpecialistAgentName] = Field(max_length=3)
    locations: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(max_length=10)
    temporal_expressions: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(max_length=10)
    confidence: float = Field(strict=True, ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)
    needs_clarification: bool
    missing_information: list[MissingInformation] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_agent_selection(self) -> "UnderstandingDecision":
        if len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("A structured understanding cannot repeat a specialist")
        if self.needs_clarification and self.agent_names:
            raise ValueError("A clarification decision cannot select specialists")
        if any(not value.strip() for value in (*self.locations, *self.temporal_expressions)):
            raise ValueError("Extracted locations and time expressions must not be blank")
        return self
