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
@pytest.mark.asyncio
async def test_routes_supported_queries(query: str, expected: str) -> None:
    result = await DeterministicQueryRouter().route(query)
    assert result.decision.agent_name.value == expected
    assert result.routing_method == "deterministic_fallback"


@pytest.mark.asyncio
async def test_routing_is_case_insensitive_and_normalizes_punctuation() -> None:
    result = await DeterministicQueryRouter().route("AIR-quality, please!")
    assert result.decision.agent_name.value == "environment"


@pytest.mark.asyncio
async def test_unsupported_query_has_no_route() -> None:
    result = await DeterministicQueryRouter().route("Write me a poem about space.")
    assert result.decision.agent_name is None
    assert result.decision.needs_clarification is False


@pytest.mark.asyncio
async def test_ambiguous_query_uses_match_count_then_documented_tie_priority() -> None:
    router = DeterministicQueryRouter()

    result = await router.route("traffic, parking, hospital")
    assert result.decision.agent_name.value == "mobility"
    # One match each: documented tie order is public_services > environment > mobility.
    result = await router.route("traffic and rain and hospital")
    assert result.decision.agent_name.value == "public_services"
