import pytest
from uuid import uuid4

from app.agents.contracts import AgentRequest
from app.agents.orchestrator.context_adapter import (
    AmbiguousSpecialistContext,
    SpecialistContextAdapter,
)


def _request(context):
    return AgentRequest(request_id=uuid4(), query="What is the weather?", context=context)


def test_environment_adapter_preserves_request_and_maps_one_trusted_nlp_location():
    request = _request({
        "user_context": {"language": "en", "location": "Untrusted place"},
        "nlp": {"locations": ["Colombo"], "temporal_expressions": ["today"]},
    })

    adapted = SpecialistContextAdapter().adapt("environment", request)

    assert adapted is not request
    assert adapted.request_id == request.request_id
    assert adapted.query == request.query
    assert adapted.context["user_context"] == request.context["user_context"]
    assert adapted.context["nlp"] == request.context["nlp"]
    assert adapted.context["location"] == "Colombo"
    assert "location" not in request.context


def test_valid_coordinates_have_priority_over_nlp_and_user_location():
    request = _request({
        "user_context": {
            "latitude": 6.9271,
            "longitude": 79.8612,
            "location": "Galle",
        },
        "nlp": {"locations": ["Colombo"]},
    })

    adapted = SpecialistContextAdapter().adapt("environment", request)

    assert adapted.context["latitude"] == pytest.approx(6.9271)
    assert adapted.context["longitude"] == pytest.approx(79.8612)
    assert adapted.context["nlp"] == {"locations": ["Colombo"]}


@pytest.mark.parametrize(
    "user_context",
    [
        {"location": "  Colombo  "},
        {"latitude": 6.9, "longitude": 79.8},
    ],
)
def test_validated_user_location_is_used_when_nlp_location_is_absent(user_context):
    adapted = SpecialistContextAdapter().adapt(
        "environment", _request({"user_context": user_context, "nlp": {"locations": []}})
    )

    if "location" in user_context:
        assert adapted.context["location"] == "Colombo"
    else:
        assert adapted.context["latitude"] == 6.9
        assert adapted.context["longitude"] == 79.8


@pytest.mark.parametrize(
    "user_context",
    [
        {"location": ["Colombo"]},
        {"location": "\nColombo"},
        {"latitude": 91, "longitude": 79.8, "location": ""},
        {"latitude": 6.9},
        {"latitude": True, "longitude": 79.8},
    ],
)
def test_invalid_user_location_is_not_promoted(user_context):
    adapter = SpecialistContextAdapter()
    request = _request({"user_context": user_context})

    assert adapter.has_valid_user_location(request.context) is False
    adapted = adapter.adapt("environment", request)
    assert adapted.context["location"] == ""
    assert "latitude" not in adapted.context
    assert "longitude" not in adapted.context


def test_multiple_trusted_environment_locations_are_not_silently_reduced():
    request = _request({"nlp": {"locations": ["Colombo", "Kandy"]}})

    with pytest.raises(AmbiguousSpecialistContext):
        SpecialistContextAdapter().adapt("environment", request)


def test_non_environment_adaptation_returns_independent_copy_without_mapping():
    request = _request({"user_context": {"location": "Colombo"}, "nlp": {"locations": ["Colombo"]}})

    adapted = SpecialistContextAdapter().adapt("mobility", request)
    adapted.context["nlp"]["locations"].append("Kandy")

    assert adapted.query == request.query
    assert adapted.request_id == request.request_id
    assert request.context["nlp"]["locations"] == ["Colombo"]
    assert "location" not in adapted.context
