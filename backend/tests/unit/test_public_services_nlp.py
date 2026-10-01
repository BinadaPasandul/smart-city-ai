import pytest

from app.ir.public_services_ir import PublicServiceCategory
from app.nlp.public_services_nlp import PublicServicesNLP, PublicServicesQuery


@pytest.fixture(scope="module")
def nlp() -> PublicServicesNLP:
    return PublicServicesNLP()


def test_hospital_detection(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("Where can I find a hospital?")

    assert result.category == PublicServiceCategory.HOSPITALS


def test_police_station_detection(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("I need to find a police station")

    assert result.category == PublicServiceCategory.POLICE_STATIONS


def test_fire_station_detection(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("Where is the nearest fire station?")

    assert result.category == PublicServiceCategory.FIRE_STATIONS


def test_government_service_detection(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("I need to renew my passport at a government office")

    assert result.category == PublicServiceCategory.GOVERNMENT_SERVICES


def test_emergency_information_detection(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("What is the ambulance emergency number?")

    assert result.category == PublicServiceCategory.EMERGENCY_INFORMATION


def test_citizen_complaint_detection(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("I want to file a complaint about a broken streetlight")

    assert result.category == PublicServiceCategory.CITIZEN_COMPLAINTS


def test_location_extraction(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("I need a hospital in Colombo")

    assert result.location == "Colombo"


def test_district_extraction_when_explicitly_stated(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("Are there any hospitals in Colombo district?")

    assert result.district == "Colombo"


def test_emergency_intent_detection(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("This is an urgent medical emergency")

    assert result.is_emergency is True


def test_no_emergency_intent_for_ordinary_query(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("What are the opening hours of the hospital?")

    assert result.is_emergency is False


def test_complaint_type_extraction(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("I want to report a broken streetlight")

    assert result.complaint_type == "streetlight_malfunction"


@pytest.mark.parametrize(
    ("query", "expected_complaint_type"),
    [
        ("There is a garbage collection problem on my street", "waste_collection_missed"),
        ("I have a water leak outside my house", "water_supply_issue"),
        ("The road near my house has severe road damage", "road_damage"),
    ],
)
def test_additional_complaint_type_mappings(
    nlp: PublicServicesNLP, query: str, expected_complaint_type: str
) -> None:
    result = nlp.parse(query)

    assert result.complaint_type == expected_complaint_type


def test_search_query_generation_strips_filler_words(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("I need an emergency hospital near Colombo")

    assert result.search_query == "emergency hospital colombo"


def test_search_query_generation_for_police_query(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("Where is the nearest police station in Kandy?")

    assert result.search_query == "police station kandy"


def test_mixed_capitalization_is_handled(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("POLICE STATION in KANDY")

    assert result.category == PublicServiceCategory.POLICE_STATIONS
    assert result.location == "Kandy"


def test_punctuation_is_handled(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("Where's the nearest hospital, please??")

    assert result.category == PublicServiceCategory.HOSPITALS


def test_unknown_location_remains_none(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("Is there a hospital in Atlantis?")

    assert result.location is None


def test_unknown_category_remains_none(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("Tell me a joke")

    assert result.category is None


def test_overlapping_category_emergency_hospital_resolves_to_hospitals(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("I need an emergency hospital")

    assert result.category == PublicServiceCategory.HOSPITALS
    assert result.is_emergency is True


def test_ambulance_emergency_number_resolves_to_emergency_information(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("ambulance emergency number")

    assert result.category == PublicServiceCategory.EMERGENCY_INFORMATION


def test_irrelevant_query_does_not_fabricate_a_category(nlp: PublicServicesNLP) -> None:
    result = nlp.parse("What is the capital of France?")

    assert result.category is None
    assert result.location is None
    assert result.complaint_type is None


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_and_whitespace_queries_are_handled_consistently(nlp: PublicServicesNLP, query: str) -> None:
    result = nlp.parse(query)

    assert isinstance(result, PublicServicesQuery)
    assert result.original_query == query
    assert result.category is None
    assert result.location is None
    assert result.district is None
    assert result.is_emergency is False
    assert result.complaint_type is None
    assert result.search_query == ""


def test_original_query_is_preserved_verbatim(nlp: PublicServicesNLP) -> None:
    query = "  I need a Hospital in Colombo!  "
    result = nlp.parse(query)

    assert result.original_query == query
