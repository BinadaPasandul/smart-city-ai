from datetime import datetime, timezone
from uuid import UUID

from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse, AgentSource


def test_request_uses_generated_uuid_and_safe_context_default() -> None:
    first = AgentRequest(query="How is traffic?")
    second = AgentRequest(query="Where is the library?")

    assert isinstance(first.request_id, UUID)
    assert first.request_id != second.request_id
    assert first.context == {}
    first.context["location"] = "Colombo"
    assert second.context == {}


def test_request_accepts_explicit_id_and_structured_context() -> None:
    request_id = UUID("9b901fa3-95c7-41b8-9124-776392704159")
    request = AgentRequest(request_id=request_id, query="Find a route", context={"origin": "A"})

    assert request.request_id == request_id
    assert request.context == {"origin": "A"}


def test_successful_response_defaults_optional_fields() -> None:
    request = AgentRequest(query="Hello")
    response = AgentResponse(request_id=request.request_id, agent_name="mobility", success=True, answer="Hi")

    assert response.success is True
    assert response.error is None
    assert response.sources == []
    assert response.metadata == {}


def test_response_can_include_sources_and_error() -> None:
    request = AgentRequest(query="Find source")
    source = AgentSource(
        name="City data",
        source_type="api",
        url="https://example.test/data",
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"dataset": "roads"},
    )
    response = AgentResponse(
        request_id=request.request_id,
        agent_name="mobility",
        success=True,
        answer="Found data",
        sources=[source],
    )
    failed = AgentResponse(
        request_id=request.request_id,
        agent_name="mobility",
        success=False,
        error=AgentError(code=AgentErrorCode.TIMEOUT, message="Request timed out"),
    )

    assert response.sources == [source]
    assert failed.error is not None
    assert failed.error.code == AgentErrorCode.TIMEOUT
    assert failed.request_id == request.request_id
    assert failed.agent_name == "mobility"


def test_source_optional_fields_can_be_omitted() -> None:
    source = AgentSource(name="Internal record", source_type="database")

    assert source.url is None
    assert source.retrieved_at is None
    assert source.metadata == {}


def test_response_mutable_defaults_are_isolated() -> None:
    request_id = AgentRequest(query="One").request_id
    first = AgentResponse(request_id=request_id, agent_name="one", success=True)
    second = AgentResponse(request_id=request_id, agent_name="two", success=True)

    first.metadata["key"] = "value"
    first.sources.append(AgentSource(name="source", source_type="document"))

    assert second.metadata == {}
    assert second.sources == []
