"""Integration tests: /chat -> Orchestrator -> AgentRegistry -> PublicServicesAgent.

These exercise the REAL bootstrap registry (so the real, registered
PublicServicesAgent -> PublicServicesNLP -> PublicServicesIR -> seed data
pipeline runs), through the real orchestrator and the real /chat endpoint.

Routing is fixed via an injected router -- the same dependency-injection
pattern already used by `backend/tests/integration/test_chat_api.py` --
rather than depending on a live Gemini key. (Step 6's report flagged that
the real `DeterministicQueryRouter` did not yet recognize everyday
complaint phrasing like "streetlight"; Step 7 closed that gap in
`router.py`, and the tests at the bottom of this file exercise the real
router directly to prove it.)
"""

import pytest
from fastapi.testclient import TestClient

from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.router import (
    DeterministicQueryRouter,
    RoutingDecision,
    RoutingResult,
    SpecialistAgentName,
)
from app.agents.orchestrator.synthesizer import DeterministicResultSynthesizer
from app.api.dependencies import get_orchestrator
from app.core.bootstrap import create_agent_registry
from app.main import app


class _FixedRouter:
    """Deterministically selects a fixed set of specialists for one test.

    Bypasses both the live Gemini router and the deterministic keyword
    router's exact phrase list, without changing any production routing
    code -- the same approach `FakeRouter` already takes in
    `test_chat_api.py`.
    """

    def __init__(self, names: list[SpecialistAgentName]) -> None:
        self.names = names

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        return RoutingResult(
            decision=RoutingDecision(
                agent_names=self.names,
                confidence=1.0,
                reason="Fixed test routing.",
                needs_clarification=False,
            ),
            routing_method="deterministic_fallback",
        )


class _FakeMobilityAgent(BaseAgent):
    """Minimal stand-in specialist used only to prove multi-agent participation.

    The real Mobility agent lives on an unmerged sibling branch; this local
    stub exists solely so a test can show Public Services running alongside
    another specialist without itself being modified.
    """

    def __init__(self) -> None:
        super().__init__("mobility", description="Test-only mobility stub.")

    async def execute(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer="Mobility stub answer.",
        )


@pytest.fixture
def client_with_orchestrator():
    original = app.dependency_overrides.get(get_orchestrator)

    def override(orchestrator) -> None:
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator

    try:
        with TestClient(app) as client:
            yield client, override
    finally:
        if original is None:
            app.dependency_overrides.pop(get_orchestrator, None)
        else:
            app.dependency_overrides[get_orchestrator] = original


def _real_orchestrator_selecting(*names: SpecialistAgentName) -> CityOrchestratorAgent:
    """Build an orchestrator around the real bootstrap registry (real PublicServicesAgent)."""
    registry = create_agent_registry()
    return CityOrchestratorAgent(registry, _FixedRouter(list(names)))


def test_chat_hospital_query_executes_real_public_services_agent(client_with_orchestrator) -> None:
    client, override = client_with_orchestrator
    override(_real_orchestrator_selecting(SpecialistAgentName.PUBLIC_SERVICES))

    response = client.post("/api/v1/chat", json={"message": "I need an emergency hospital near Colombo"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["metadata"]["selected_agents"] == ["public_services"]
    assert body["metadata"]["successful_agents"] == ["public_services"]
    assert body["metadata"]["synthesis_method"] == "single_specialist_passthrough"
    assert "Colombo" in body["answer"]
    assert body["sources"]
    assert body["sources"][0]["name"] == "public_services_data.json"
    assert body["sources"][0]["source_type"] == "synthetic_demo"
    assert body["sources"][0]["metadata"]["category"] == "hospitals"


def test_chat_police_query_executes_real_public_services_agent(client_with_orchestrator) -> None:
    client, override = client_with_orchestrator
    override(_real_orchestrator_selecting(SpecialistAgentName.PUBLIC_SERVICES))

    response = client.post("/api/v1/chat", json={"message": "Where is the nearest police station in Kandy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["metadata"]["successful_agents"] == ["public_services"]
    assert "Kandy" in body["answer"]
    assert body["sources"][0]["metadata"]["category"] == "police_stations"


def test_chat_citizen_complaint_query_executes_real_public_services_agent(client_with_orchestrator) -> None:
    client, override = client_with_orchestrator
    override(_real_orchestrator_selecting(SpecialistAgentName.PUBLIC_SERVICES))

    response = client.post("/api/v1/chat", json={"message": "I want to report a broken streetlight"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["metadata"]["successful_agents"] == ["public_services"]
    assert "streetlight" in body["answer"].lower()
    assert body["sources"][0]["metadata"]["category"] == "citizen_complaints"


def test_public_services_participates_in_multi_agent_selection(client_with_orchestrator) -> None:
    registry = create_agent_registry()
    registry.register(_FakeMobilityAgent())
    orchestrator = CityOrchestratorAgent(
        registry,
        _FixedRouter([SpecialistAgentName.MOBILITY, SpecialistAgentName.PUBLIC_SERVICES]),
        synthesizer=DeterministicResultSynthesizer(),
    )
    client, override = client_with_orchestrator
    override(orchestrator)

    response = client.post(
        "/api/v1/chat",
        json={"message": "What's traffic like and where is the nearest hospital in Colombo?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert set(body["metadata"]["successful_agents"]) == {"mobility", "public_services"}
    assert body["metadata"]["execution_status"] == "complete"
    assert "Mobility stub answer." in body["answer"]
    assert any(source["metadata"].get("category") == "hospitals" for source in body["sources"])


def test_registered_public_services_agent_no_longer_returns_agent_not_found(client_with_orchestrator) -> None:
    client, override = client_with_orchestrator
    override(_real_orchestrator_selecting(SpecialistAgentName.PUBLIC_SERVICES))

    response = client.post("/api/v1/chat", json={"message": "I need a hospital in Colombo"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert "agent_not_found" not in response.text


# ---------------------------------------------------------------------------
# Step 7: the real deterministic router (no Gemini) must now route the
# previously-unrecognized citizen-complaint phrasing to public_services.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streetlight_complaint_routes_through_real_deterministic_router() -> None:
    """Reproduces the originally-reported routing gap end-to-end, without Gemini.

    Calls the real orchestrator directly (not through FastAPI) because the
    public ChatResponse schema intentionally allowlists away the agent-level
    `category`/`complaint_type` metadata this test checks (see
    `ChatMetadata.from_agent_metadata`); the registry, agent, NLP, and IR are
    all real and unmodified.
    """
    registry = create_agent_registry()
    orchestrator = CityOrchestratorAgent(registry, DeterministicQueryRouter())
    request = AgentRequest(query="I want to report a broken streetlight")

    response = await orchestrator.execute(request)

    assert response.success is True
    assert response.metadata["routing_method"] == "deterministic_fallback"
    assert response.metadata["selected_agents"] == ["public_services"]
    assert response.metadata["successful_agents"] == ["public_services"]
    assert response.metadata["category"] == "citizen_complaints"
    assert response.metadata["complaint_type"] == "streetlight_malfunction"
    assert any(source.name == "public_services_data.json" for source in response.sources)
    assert any(source.source_type == "synthetic_demo" for source in response.sources)


def test_chat_streetlight_complaint_routes_without_gemini(client_with_orchestrator) -> None:
    """Same gap, verified at the public /chat HTTP boundary."""
    client, override = client_with_orchestrator
    registry = create_agent_registry()
    override(CityOrchestratorAgent(registry, DeterministicQueryRouter()))

    response = client.post("/api/v1/chat", json={"message": "I want to report a broken streetlight"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["metadata"]["routing_method"] == "deterministic_fallback"
    assert body["metadata"]["selected_agents"] == ["public_services"]
    assert body["metadata"]["successful_agents"] == ["public_services"]
    assert body["sources"][0]["name"] == "public_services_data.json"
    assert body["sources"][0]["metadata"]["category"] == "citizen_complaints"
