"""Typed specialist execution outcomes collected by the Orchestrator."""

from enum import Enum

from pydantic import BaseModel, Field

from app.agents.contracts import AgentError, AgentResponse
from app.agents.orchestrator.router import SpecialistAgentName


class ExecutionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class SpecialistExecutionResult(BaseModel):
    """Outcome for one selected specialist."""

    agent_name: SpecialistAgentName
    success: bool
    response: AgentResponse | None = None
    error: AgentError | None = None
    timed_out: bool = False


class OrchestrationExecutionSummary(BaseModel):
    """Stable, structured summary for one or more specialist executions."""

    requested_agents: list[SpecialistAgentName] = Field(default_factory=list)
    successful_agents: list[SpecialistAgentName] = Field(default_factory=list)
    failed_agents: list[SpecialistAgentName] = Field(default_factory=list)
    status: ExecutionStatus
    results: list[SpecialistExecutionResult] = Field(default_factory=list)
