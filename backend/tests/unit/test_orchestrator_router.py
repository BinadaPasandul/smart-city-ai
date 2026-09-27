import pytest

from app.agents.orchestrator.router import DeterministicQueryRouter


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Is there heavy traffic?", "mobility"),
        ("Where can I park near Colombo Fort?", "mobility"),
        ("How do I use public transport?", "mobility"),
        ("Will the weather be rainy?", "environment"),
        ("Is pollution high today?", "environment"),
        ("How is the air quality today?", "environment"),
        ("Where is the nearest hospital?", "public_services"),
        ("How do I find a police station?", "public_services"),
        ("How can I access government services?", "public_services"),
    ],
)
def test_routes_supported_queries(query: str, expected: str) -> None:
    assert DeterministicQueryRouter().route(query) == expected


def test_routing_is_case_insensitive_and_normalizes_punctuation() -> None:
    assert DeterministicQueryRouter().route("AIR-quality, please!") == "environment"


def test_unsupported_query_has_no_route() -> None:
    assert DeterministicQueryRouter().route("Write me a poem about space.") is None


def test_ambiguous_query_uses_match_count_then_documented_tie_priority() -> None:
    router = DeterministicQueryRouter()

    assert router.route("traffic, parking, hospital") == "mobility"
    # One match each: documented tie order is public_services > environment > mobility.
    assert router.route("traffic and rain and hospital") == "public_services"
