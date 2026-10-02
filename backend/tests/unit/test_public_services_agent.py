from uuid import uuid4

import pytest

from app.agents.base import BaseAgent
from app.agents.contracts import AgentErrorCode, AgentRequest
from app.agents.public_services.agent import PublicServicesAgent
from app.ir.public_services_ir import (
    PublicServiceCategory,
    PublicServiceRecord,
    SearchMatch,
    SourceInfo,
)
from app.nlp.public_services_nlp import PublicServicesQuery

_SOURCE = SourceInfo(
    source_file="public_services_data.json",
    source_type="synthetic_demo",
    description="Synthetic seed data for development and testing. Not real-world service information.",
)


def _make_match(
    category: PublicServiceCategory,
    record_id: str,
    title: str,
    fields: dict,
    *,
    score: float = 0.9,
    matched_terms: list[str] | None = None,
) -> SearchMatch:
    return SearchMatch(
        record=PublicServiceRecord(category=category, record_id=record_id, title=title, fields=fields),
        score=score,
        matched_terms=matched_terms or [],
        source=_SOURCE,
    )


def _make_query(
    *,
    query: str = "test query",
    category: PublicServiceCategory | None = None,
    location: str | None = None,
    district: str | None = None,
    is_emergency: bool = False,
    complaint_type: str | None = None,
    search_query: str | None = None,
) -> PublicServicesQuery:
    return PublicServicesQuery(
        original_query=query,
        category=category,
        location=location,
        district=district,
        is_emergency=is_emergency,
        complaint_type=complaint_type,
        search_query=search_query if search_query is not None else query,
    )


class _FakeNLP:
    """Stub PublicServicesNLP returning a fixed parsed query and recording calls."""

    def __init__(self, result: PublicServicesQuery) -> None:
        self._result = result
        self.calls: list[str] = []

    def parse(self, query: str) -> PublicServicesQuery:
        self.calls.append(query)
        return self._result


class _FailingNLP:
    def parse(self, query: str) -> PublicServicesQuery:
        raise RuntimeError("nlp internal failure: some private detail")


class _FakeIR:
    """Stub PublicServicesIR returning fixed results and recording calls."""

    def __init__(self, results: list[SearchMatch]) -> None:
        self._results = results
        self.calls: list[tuple[str, PublicServiceCategory | None]] = []

    def search(
        self, query: str, category: PublicServiceCategory | str | None = None, top_k: int = 10
    ) -> list[SearchMatch]:
        self.calls.append((query, category))
        return self._results


class _FailingIR:
    def search(self, query: str, category=None, top_k: int = 10) -> list[SearchMatch]:
        raise RuntimeError("ir internal failure: some private detail")


# ---------------------------------------------------------------------------
# Basic contract (retained from the Step 1 foundation)
# ---------------------------------------------------------------------------


def test_public_services_agent_can_be_instantiated() -> None:
    agent = PublicServicesAgent(nlp=_FakeNLP(_make_query()), ir=_FakeIR([]))

    assert isinstance(agent, BaseAgent)


def test_public_services_agent_name_is_exact() -> None:
    agent = PublicServicesAgent(nlp=_FakeNLP(_make_query()), ir=_FakeIR([]))

    assert agent.name == "public_services"


def test_public_services_agent_exposes_capabilities() -> None:
    agent = PublicServicesAgent(nlp=_FakeNLP(_make_query()), ir=_FakeIR([]))

    assert agent.capabilities == (
        "rule_based_nlp",
        "keyword_search_retrieval",
        "seed_data_grounded_answers",
    )
    assert agent.description


@pytest.mark.asyncio
async def test_whitespace_only_query_returns_invalid_request_without_calling_nlp() -> None:
    fake_nlp = _FakeNLP(_make_query())
    agent = PublicServicesAgent(nlp=fake_nlp, ir=_FakeIR([]))
    request = AgentRequest(query="   ")

    response = await agent.execute(request)

    assert response.request_id == request.request_id
    assert response.success is False
    assert response.error.code == AgentErrorCode.INVALID_REQUEST
    assert fake_nlp.calls == []


# ---------------------------------------------------------------------------
# Category-by-category integration (mocked NLP/IR)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hospital_query_routes_nlp_output_into_ir_and_succeeds() -> None:
    parsed = _make_query(
        query="I need an emergency hospital near Colombo",
        category=PublicServiceCategory.HOSPITALS,
        location="Colombo",
        is_emergency=True,
        search_query="emergency hospital colombo",
    )
    match = _make_match(
        PublicServiceCategory.HOSPITALS,
        "hosp_001",
        "Colombo National General Hospital",
        {
            "id": "hosp_001",
            "type": "teaching",
            "location": "Colombo",
            "services": ["emergency", "surgery"],
            "contact": "+94-000-0001",
            "operating_hours": "24/7",
            "emergency_available": True,
        },
    )
    fake_nlp = _FakeNLP(parsed)
    fake_ir = _FakeIR([match])
    agent = PublicServicesAgent(nlp=fake_nlp, ir=fake_ir)
    request = AgentRequest(query="I need an emergency hospital near Colombo")

    response = await agent.execute(request)

    assert fake_nlp.calls == ["I need an emergency hospital near Colombo"]
    assert fake_ir.calls == [("emergency hospital colombo", PublicServiceCategory.HOSPITALS)]
    assert response.success is True
    assert "Colombo National General Hospital" in response.answer


@pytest.mark.parametrize(
    "category",
    [
        PublicServiceCategory.POLICE_STATIONS,
        PublicServiceCategory.FIRE_STATIONS,
        PublicServiceCategory.GOVERNMENT_SERVICES,
        PublicServiceCategory.EMERGENCY_INFORMATION,
        PublicServiceCategory.CITIZEN_COMPLAINTS,
    ],
)
@pytest.mark.asyncio
async def test_each_category_is_forwarded_from_nlp_to_ir(category: PublicServiceCategory) -> None:
    parsed = _make_query(category=category, search_query="normalized query text")
    match = _make_match(category, "rec_001", "Some Record", {"id": "rec_001"})
    fake_nlp = _FakeNLP(parsed)
    fake_ir = _FakeIR([match])
    agent = PublicServicesAgent(nlp=fake_nlp, ir=fake_ir)

    response = await agent.execute(AgentRequest(query="a relevant query"))

    assert fake_ir.calls == [("normalized query text", category)]
    assert response.success is True
    assert response.metadata["category"] == category.value


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_location_metadata_is_preserved() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS, location="Colombo")
    match = _make_match(PublicServiceCategory.HOSPITALS, "hosp_001", "Hospital", {})
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([match]))

    response = await agent.execute(AgentRequest(query="query"))

    assert response.metadata["location"] == "Colombo"


@pytest.mark.asyncio
async def test_emergency_metadata_is_preserved() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS, is_emergency=True)
    match = _make_match(PublicServiceCategory.HOSPITALS, "hosp_001", "Hospital", {})
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([match]))

    response = await agent.execute(AgentRequest(query="query"))

    assert response.metadata["is_emergency"] is True


@pytest.mark.asyncio
async def test_complaint_type_metadata_is_preserved() -> None:
    parsed = _make_query(
        category=PublicServiceCategory.CITIZEN_COMPLAINTS,
        complaint_type="streetlight_malfunction",
    )
    match = _make_match(PublicServiceCategory.CITIZEN_COMPLAINTS, "complaint_001", "Streetlight", {})
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([match]))

    response = await agent.execute(AgentRequest(query="query"))

    assert response.metadata["complaint_type"] == "streetlight_malfunction"


@pytest.mark.asyncio
async def test_metadata_reflects_no_district_and_default_implementation_status() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS)
    match = _make_match(PublicServiceCategory.HOSPITALS, "hosp_001", "Hospital", {})
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([match]))

    response = await agent.execute(AgentRequest(query="query"))

    assert response.metadata["district"] is None
    assert response.metadata["implementation_status"] == "seed_data"
    assert response.metadata["retrieval_method"] == "keyword_overlap"
    assert response.metadata["data_source"] == "synthetic_demo"


# ---------------------------------------------------------------------------
# AgentSource grounding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_source_entries_are_created_from_search_matches() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS)
    match = _make_match(
        PublicServiceCategory.HOSPITALS,
        "hosp_001",
        "Hospital",
        {},
        score=0.75,
        matched_terms=["hospital", "colombo"],
    )
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([match]))

    response = await agent.execute(AgentRequest(query="query"))

    assert len(response.sources) == 1
    source = response.sources[0]
    assert source.metadata["record_id"] == "hosp_001"
    assert source.metadata["category"] == "hospitals"
    assert source.metadata["relevance_score"] == 0.75
    assert source.metadata["matched_terms"] == ["hospital", "colombo"]


@pytest.mark.asyncio
async def test_agent_source_identifies_synthetic_demo_data_file() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS)
    match = _make_match(PublicServiceCategory.HOSPITALS, "hosp_001", "Hospital", {})
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([match]))

    response = await agent.execute(AgentRequest(query="query"))

    source = response.sources[0]
    assert source.name == "public_services_data.json"
    assert source.source_type == "synthetic_demo"


# ---------------------------------------------------------------------------
# No-result / unsupported handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_result_query_does_not_fabricate_an_answer() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS)
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([]))

    response = await agent.execute(AgentRequest(query="query"))

    assert response.success is False
    assert response.answer == ""
    assert response.sources == []
    assert response.error is not None
    assert response.error.code == AgentErrorCode.UNSUPPORTED_REQUEST
    assert response.metadata["result_count"] == 0


@pytest.mark.asyncio
async def test_unknown_category_returns_structured_response_without_calling_ir() -> None:
    parsed = _make_query(category=None)
    fake_ir = _FakeIR([])
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=fake_ir)

    response = await agent.execute(AgentRequest(query="Tell me a joke"))

    assert response.success is False
    assert response.error.code == AgentErrorCode.UNSUPPORTED_REQUEST
    assert fake_ir.calls == []


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ir_failure_is_handled_without_leaking_internal_details() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS)
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FailingIR())

    response = await agent.execute(AgentRequest(query="query"))

    assert response.success is False
    assert response.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert "private detail" not in response.error.message


@pytest.mark.asyncio
async def test_nlp_failure_is_handled_without_calling_ir() -> None:
    fake_ir = _FakeIR([])
    agent = PublicServicesAgent(nlp=_FailingNLP(), ir=fake_ir)

    response = await agent.execute(AgentRequest(query="query"))

    assert response.success is False
    assert response.error.code == AgentErrorCode.AGENT_EXECUTION_FAILED
    assert "private detail" not in response.error.message
    assert fake_ir.calls == []


# ---------------------------------------------------------------------------
# Grounding and formatting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_is_grounded_only_in_retrieved_record_fields() -> None:
    parsed = _make_query(category=PublicServiceCategory.HOSPITALS)
    match = _make_match(
        PublicServiceCategory.HOSPITALS,
        "hosp_005",
        "Negombo City Hospital",
        {
            "id": "hosp_005",
            "location": "Negombo",
            # contact/operating_hours/emergency_available deliberately absent
        },
    )
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR([match]))

    response = await agent.execute(AgentRequest(query="query"))

    assert "Negombo City Hospital" in response.answer
    assert "Location: Negombo" in response.answer
    assert "Contact" not in response.answer
    assert "Operating hours" not in response.answer
    assert "Emergency available" not in response.answer


@pytest.mark.asyncio
async def test_multiple_results_are_truncated_and_formatted_with_numbering() -> None:
    parsed = _make_query(category=PublicServiceCategory.POLICE_STATIONS)
    matches = [
        _make_match(PublicServiceCategory.POLICE_STATIONS, f"police_00{i}", f"Station {i}", {})
        for i in range(1, 5)
    ]
    agent = PublicServicesAgent(nlp=_FakeNLP(parsed), ir=_FakeIR(matches))

    response = await agent.execute(AgentRequest(query="query"))

    assert response.metadata["result_count"] == 4
    assert len(response.sources) == 3
    assert "1. Station 1" in response.answer
    assert "2. Station 2" in response.answer
    assert "3. Station 3" in response.answer
    assert "Station 4" not in response.answer
    assert "Found 4 matching police stations record(s)." in response.answer
    assert "Showing the top 3." in response.answer


# ---------------------------------------------------------------------------
# Real end-to-end pipeline (no mocks) against the actual seed dataset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_pipeline_answers_a_police_station_query() -> None:
    agent = PublicServicesAgent()
    request = AgentRequest(request_id=uuid4(), query="Where is the nearest police station in Kandy?")

    response = await agent.execute(request)

    assert response.request_id == request.request_id
    assert response.success is True
    assert response.metadata["category"] == "police_stations"
    assert response.metadata["location"] == "Kandy"
    assert "Kandy Central Police Station" in response.answer
    assert response.sources
    assert response.sources[0].metadata["category"] == "police_stations"


@pytest.mark.asyncio
async def test_real_pipeline_returns_unsupported_for_irrelevant_query() -> None:
    agent = PublicServicesAgent()

    response = await agent.execute(AgentRequest(query="Tell me a joke"))

    assert response.success is False
    assert response.error.code == AgentErrorCode.UNSUPPORTED_REQUEST
