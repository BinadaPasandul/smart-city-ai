import pytest

from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse


def test_base_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgent("test")


@pytest.mark.asyncio
async def test_fake_agent_implements_async_contract() -> None:
    class FakeAgent(BaseAgent):
        async def execute(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                success=True,
                answer=request.query,
            )

    agent = FakeAgent("fake", description="Test agent", capabilities=("echo",))
    request = AgentRequest(query="Hello")
    response = await agent.execute(request)

    assert agent.name == "fake"
    assert agent.description == "Test agent"
    assert agent.capabilities == ("echo",)
    assert response.request_id == request.request_id
    assert response.agent_name == "fake"
    assert response.answer == "Hello"


def test_agent_name_must_be_nonempty_and_trimmed() -> None:
    class FakeAgent(BaseAgent):
        async def execute(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(request_id=request.request_id, agent_name=self.name, success=True)

    with pytest.raises(ValueError):
        FakeAgent("")
