"""Abstract asynchronous interface implemented by future city agents."""

from abc import ABC, abstractmethod

from app.agents.contracts import AgentRequest, AgentResponse


class BaseAgent(ABC):
    """Base contract; contains no domain or integration behavior."""

    def __init__(self, name: str, *, description: str = "", capabilities: tuple[str, ...] = ()) -> None:
        if not name or name.strip() != name:
            raise ValueError("Agent name must be non-empty and have no surrounding whitespace")
        self.name = name
        self.description = description
        self.capabilities = capabilities

    @abstractmethod
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Handle a shared request and return the standard response envelope."""
        raise NotImplementedError
