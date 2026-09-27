"""Simple in-process registry for BaseAgent implementations."""

from app.agents.base import BaseAgent


class DuplicateAgentError(ValueError):
    """Raised when an agent name is already registered."""


class AgentNotFoundError(LookupError):
    """Raised when a requested agent name is not registered."""


class AgentRegistry:
    """Store and retrieve agents by their stable machine-readable name."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent, rejecting duplicate names."""
        if agent.name in self._agents:
            raise DuplicateAgentError(f"Agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        """Return a registered agent or raise AgentNotFoundError."""
        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentNotFoundError(f"Agent '{name}' is not registered") from exc

    def list_agents(self) -> list[BaseAgent]:
        """Return a snapshot of currently registered agents."""
        return list(self._agents.values())
