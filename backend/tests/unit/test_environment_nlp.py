import pytest

from app.nlp.environment_nlp import EnvironmentIntent, EnvironmentNLP, EnvironmentQuery


@pytest.fixture
def nlp() -> EnvironmentNLP:
    return EnvironmentNLP()


@pytest.mark.parametrize(
    "query",
    ["What's the weather today?", "Will it rain tomorrow?", "Current temperature in Colombo"],
)
def test_weather_only_queries(nlp: EnvironmentNLP, query: str) -> None:
    assert nlp.parse(query).intent == EnvironmentIntent.WEATHER


@pytest.mark.parametrize(
    "query",
    ["Air quality today", "What is the AQI?", "PM2.5 levels", "Is there smog?"],
)
def test_air_quality_only_queries(nlp: EnvironmentNLP, query: str) -> None:
    assert nlp.parse(query).intent == EnvironmentIntent.AIR_QUALITY


@pytest.mark.parametrize(
    "query",
    [
        "Weather and air quality today",
        "Will it rain, and what are the pollution levels?",
        "Temperature and AQI forecast",
    ],
)
def test_queries_requesting_both(nlp: EnvironmentNLP, query: str) -> None:
    assert nlp.parse(query).intent == EnvironmentIntent.BOTH


@pytest.mark.parametrize("query", ["Tell me a joke", "How far is the moon?", "traffic in Colombo"])
def test_unrelated_queries_are_unknown(nlp: EnvironmentNLP, query: str) -> None:
    assert nlp.parse(query).intent == EnvironmentIntent.UNKNOWN


def test_case_whitespace_and_punctuation_are_normalized(nlp: EnvironmentNLP) -> None:
    result = nlp.parse("  AIR\tQUALITY   &   PM2.5?! ")
    assert result.intent == EnvironmentIntent.AIR_QUALITY
    assert result.search_query == "air quality pm2 5"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("weather forecast", EnvironmentIntent.WEATHER),
        ("rainfall outlook", EnvironmentIntent.WEATHER),
        ("air pollution forecast", EnvironmentIntent.AIR_QUALITY),
        ("particulate matter levels", EnvironmentIntent.AIR_QUALITY),
        ("temperature and smog", EnvironmentIntent.BOTH),
    ],
)
def test_common_synonyms_and_phrases(
    nlp: EnvironmentNLP, query: str, expected: EnvironmentIntent
) -> None:
    assert nlp.parse(query).intent == expected


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_input_is_unknown_and_preserves_original(nlp: EnvironmentNLP, query: str) -> None:
    result = nlp.parse(query)
    assert isinstance(result, EnvironmentQuery)
    assert result.original_query == query
    assert result.intent == EnvironmentIntent.UNKNOWN
    assert result.search_query == ""


def test_nonempty_query_preserves_original_and_normalized_search_text(nlp: EnvironmentNLP) -> None:
    query = "  WILL it RAIN in Colombo?! "
    result = nlp.parse(query)
    assert result.original_query == query
    assert result.search_query == "will it rain in colombo"
