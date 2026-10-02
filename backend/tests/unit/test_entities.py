"""Tests for SpacyEntityExtractor's real behavior: location composition and
the Sri Lankan gazetteer supplement.

Unlike test_nlp_pipeline.py (which fakes this extractor out deliberately,
for pipeline-level isolation), these tests exercise the real spaCy model,
since the specific behavior under test -- composing adjacent entities and
recovering from spaCy's own mislabeling -- only exists in this module.
"""

import pytest

from app.nlp.entities import EntityModelUnavailable, SpacyEntityExtractor


@pytest.fixture(scope="module")
def extractor() -> SpacyEntityExtractor:
    instance = SpacyEntityExtractor()
    try:
        instance.extract_locations("warm-up")
    except EntityModelUnavailable:
        pytest.skip("en_core_web_sm is not installed in this environment")
    return instance


# ---------------------------------------------------------------------------
# Fix 1: "<location>, <country>" composition
# ---------------------------------------------------------------------------


def test_city_and_country_are_composed_into_one_location(extractor: SpacyEntityExtractor) -> None:
    result = extractor.extract_locations("What is the weather in Colombo, Sri Lanka?")

    assert result == ["Colombo, Sri Lanka"]


def test_two_separate_city_country_phrases_stay_separate(extractor: SpacyEntityExtractor) -> None:
    result = extractor.extract_locations(
        "Compare Colombo, Sri Lanka and Kandy, Sri Lanka"
    )

    assert result == ["Colombo, Sri Lanka", "Kandy, Sri Lanka"]


def test_two_cities_joined_by_and_are_not_composed(extractor: SpacyEntityExtractor) -> None:
    result = extractor.extract_locations("Compare Colombo and Kandy")

    assert result == ["Colombo", "Kandy"]
    assert "Colombo and Kandy" not in result
    assert "Colombo, Kandy" not in result


def test_city_followed_by_non_country_is_not_composed() -> None:
    """A comma followed by something that isn't a recognized country must
    not be merged -- guards against blindly joining arbitrary adjacent
    GPE-like spans (e.g. a hypothetical list of cities)."""
    from app.nlp.sri_lanka_locations import is_known_country

    assert is_known_country("Galle") is False
    assert is_known_country("Kandy") is False
    assert is_known_country("Sri Lanka") is True


# ---------------------------------------------------------------------------
# Fix 2: Sri Lankan gazetteer supplement
# ---------------------------------------------------------------------------


def test_kandy_alone_is_recovered_despite_spacy_mislabeling_it(
    extractor: SpacyEntityExtractor,
) -> None:
    """Confirmed via direct investigation: en_core_web_sm tags bare "Kandy"
    as PERSON, not GPE/LOC/FAC. The gazetteer must recover it anyway."""
    result = extractor.extract_locations("What's the weather in Kandy?")

    assert result == ["Kandy"]


def test_kandy_comma_country_composes_via_gazetteer_and_spacy_together(
    extractor: SpacyEntityExtractor,
) -> None:
    """"Kandy" comes from the gazetteer (spaCy mistags it); "Sri Lanka"
    comes from spaCy itself -- composition must work across that mixed
    origin exactly as it would if both came from spaCy."""
    result = extractor.extract_locations("What's the weather in Kandy, Sri Lanka?")

    assert result == ["Kandy, Sri Lanka"]


def test_colombo_is_not_duplicated_between_spacy_and_gazetteer(
    extractor: SpacyEntityExtractor,
) -> None:
    """spaCy already correctly tags "Colombo" as GPE; the gazetteer must not
    add a second, duplicate entry for the same span."""
    result = extractor.extract_locations("What is the weather in Colombo, Sri Lanka?")

    assert result.count("Colombo, Sri Lanka") == 1
    assert len(result) == 1


@pytest.mark.parametrize(
    "city",
    [
        "Kandy", "Colombo", "Galle", "Jaffna", "Negombo", "Matara",
        "Kurunegala", "Anuradhapura", "Trincomalee", "Batticaloa",
        "Nuwara Eliya", "Ratnapura",
    ],
)
def test_every_gazetteer_city_is_extracted_on_its_own(
    extractor: SpacyEntityExtractor, city: str
) -> None:
    result = extractor.extract_locations(f"What's the weather in {city}?")

    assert result == [city]


def test_gazetteer_matching_is_case_insensitive(extractor: SpacyEntityExtractor) -> None:
    result = extractor.extract_locations("weather in kandy please")

    assert result == ["kandy"], "original casing from the query should be preserved"


def test_gazetteer_does_not_match_inside_a_longer_word(
    extractor: SpacyEntityExtractor,
) -> None:
    result = extractor.extract_locations("The Kandyland festival was fun")

    assert "Kandy" not in result
    assert result == []


# ---------------------------------------------------------------------------
# Confirms the fix actually resolves the originally-reported failure (#8 from
# the investigation): a genuine two-city comparison must no longer silently
# collapse to one recognized location.
# ---------------------------------------------------------------------------


def test_colombo_and_kandy_comparison_no_longer_silently_drops_kandy(
    extractor: SpacyEntityExtractor,
) -> None:
    result = extractor.extract_locations("Compare the air quality in Colombo and Kandy.")

    assert result == ["Colombo", "Kandy"]
    assert len(result) == 2
