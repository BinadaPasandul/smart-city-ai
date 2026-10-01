import pytest

from app.ir.public_services_ir import (
    InvalidPublicServiceCategoryError,
    PublicServiceCategory,
    PublicServicesIR,
)


@pytest.fixture(scope="module")
def ir() -> PublicServicesIR:
    return PublicServicesIR()


def test_dataset_loading_succeeds(ir: PublicServicesIR) -> None:
    for category in PublicServiceCategory:
        assert len(ir.records(category)) > 0


def test_all_six_categories_are_recognized(ir: PublicServicesIR) -> None:
    categories = ir.available_categories()

    assert set(categories) == {
        PublicServiceCategory.HOSPITALS,
        PublicServiceCategory.POLICE_STATIONS,
        PublicServiceCategory.FIRE_STATIONS,
        PublicServiceCategory.GOVERNMENT_SERVICES,
        PublicServiceCategory.EMERGENCY_INFORMATION,
        PublicServiceCategory.CITIZEN_COMPLAINTS,
    }


def test_search_hospitals_finds_relevant_records(ir: PublicServicesIR) -> None:
    results = ir.search("emergency hospital Colombo", category=PublicServiceCategory.HOSPITALS)

    assert results
    assert all(r.record.category == PublicServiceCategory.HOSPITALS for r in results)
    assert any(r.record.record_id == "hosp_001" for r in results)


def test_search_police_stations_finds_relevant_records(ir: PublicServicesIR) -> None:
    results = ir.search("police station Kandy", category=PublicServiceCategory.POLICE_STATIONS)

    assert results
    assert all(r.record.category == PublicServiceCategory.POLICE_STATIONS for r in results)
    assert results[0].record.record_id == "police_002"


def test_search_fire_stations_finds_relevant_records(ir: PublicServicesIR) -> None:
    results = ir.search("fire station Galle", category=PublicServiceCategory.FIRE_STATIONS)

    assert results
    assert all(r.record.category == PublicServiceCategory.FIRE_STATIONS for r in results)
    assert results[0].record.record_id == "fire_003"


def test_search_government_services_finds_relevant_records(ir: PublicServicesIR) -> None:
    results = ir.search(
        "vehicle registration driving license",
        category=PublicServiceCategory.GOVERNMENT_SERVICES,
    )

    assert results
    assert results[0].record.record_id == "govserv_001"


def test_search_emergency_information_finds_relevant_records(ir: PublicServicesIR) -> None:
    results = ir.search("ambulance medical emergency", category=PublicServiceCategory.EMERGENCY_INFORMATION)

    assert results
    assert results[0].record.record_id == "emerg_001"


def test_search_citizen_complaints_finds_relevant_records(ir: PublicServicesIR) -> None:
    results = ir.search("report a streetlight problem", category=PublicServiceCategory.CITIZEN_COMPLAINTS)

    assert results
    assert results[0].record.record_id == "complaint_001"


def test_category_filtering_restricts_results(ir: PublicServicesIR) -> None:
    results = ir.search("Colombo", category=PublicServiceCategory.HOSPITALS)

    assert results
    assert all(r.record.category == PublicServiceCategory.HOSPITALS for r in results)


def test_search_without_category_searches_across_categories(ir: PublicServicesIR) -> None:
    results = ir.search("Colombo")

    categories_seen = {r.record.category for r in results}
    assert len(categories_seen) > 1


def test_results_are_ordered_by_descending_relevance_score(ir: PublicServicesIR) -> None:
    results = ir.search("hospital Colombo emergency")

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_blank_query_returns_empty_results(ir: PublicServicesIR) -> None:
    assert ir.search("") == []
    assert ir.search("   ") == []


def test_invalid_category_raises_clear_error(ir: PublicServicesIR) -> None:
    with pytest.raises(InvalidPublicServiceCategoryError):
        ir.search("hospital", category="not_a_real_category")


def test_query_with_no_matching_information_returns_empty_results(ir: PublicServicesIR) -> None:
    results = ir.search("xylophone spaceship unicorn parade")

    assert results == []


def test_matched_terms_and_source_are_populated(ir: PublicServicesIR) -> None:
    results = ir.search("hospital Colombo", category=PublicServiceCategory.HOSPITALS, top_k=1)

    assert results
    match = results[0]
    assert "colombo" in match.matched_terms or "hospital" in match.matched_terms
    assert match.source.source_file == "public_services_data.json"
    assert match.source.source_type == "synthetic_demo"
    assert match.record.fields.get("id") == match.record.record_id


def test_top_k_limits_result_count(ir: PublicServicesIR) -> None:
    results = ir.search("station", category=PublicServiceCategory.POLICE_STATIONS, top_k=2)

    assert len(results) <= 2
