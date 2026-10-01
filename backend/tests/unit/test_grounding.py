import pytest

from app.agents.orchestrator.grounding import validate_grounding


@pytest.mark.parametrize(
    ("evidence", "answer"),
    [
        ("Traffic is moderate.", "Traffic conditions are moderate."),
        ("Air quality index is 74.", "The reported AQI is 74."),
        ("Hospital A is located in Colombo.", "Hospital A is in Colombo."),
        ("The journey takes 20 mins.", "The journey takes 20 minutes."),
        ("Speed is 12 km/h.", "Speed is 12.0 km/h."),
        ("PM2.5 is 12.", "Particulate matter 2.5 is 12."),
        ("Values range from 10 to 20.", "Values ranging from 10 to 20."),
    ],
)
def test_grounding_accepts_narrowly_normalized_paraphrases(evidence, answer):
    result = validate_grounding(answer, [evidence])
    assert result.supported, result
    assert result.reason_codes == []


@pytest.mark.parametrize(
    ("evidence", "answer", "reason"),
    [
        ("Traffic is moderate.", "A major accident is causing heavy delays.", "unsupported_causality"),
        ("AQI is 74.", "AQI is 74 and schools are closed.", "unsupported_closure"),
        ("Construction is nearby.", "Traffic is heavy because of construction.", "unsupported_causality"),
        ("A demonstration is nearby.", "The demonstration caused congestion.", "unsupported_causality"),
        ("Hospital A is in Colombo.", "Hospital A is the largest hospital in Sri Lanka.", "unsupported_named_entity"),
    ],
)
def test_grounding_rejects_high_risk_unsupported_claims(evidence, answer, reason):
    result = validate_grounding(answer, [evidence])
    assert result.supported is False
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    ("evidence", "answer", "supported", "reason"),
    [
        ("74", "74", True, None),
        ("74", "74.0", True, None),
        ("AQI 74", "AQI 74.0", True, None),
        ("AQI 74", "AQI 75", False, "unsupported_numeric_fact"),
        ("AQI 74", "74%", False, "unsupported_numeric_unit"),
        ("12 km/h", "12.0 km/h", True, None),
        ("12 km/h", "12 minutes", False, "unsupported_numeric_unit"),
        ("PM2.5 is 12 μg/m³", "PM2.5 is 12 mg/L", False, "unsupported_numeric_unit"),
        ("20 mins", "20 minutes", True, None),
        ("74%", "74.0%", True, None),
        ("0.74", "74%", False, "unsupported_numeric_fact"),
        ("74%", "0.74", False, "unsupported_numeric_fact"),
    ],
)
def test_numeric_facts_keep_decimal_equivalence_and_unit_context(evidence, answer, supported, reason):
    result = validate_grounding(answer, [evidence])
    assert result.supported is supported, result
    if reason:
        assert reason in result.reason_codes


def test_temporal_claim_needs_evidence_even_when_present_in_user_query():
    # User query is deliberately absent from evidence passed to this validator.
    result = validate_grounding("Traffic is moderate today.", ["Traffic is moderate."])
    assert result.supported is False
    assert "unsupported_temporal_claim" in result.reason_codes


def test_evidenced_iso_date_is_supported_and_not_split_into_numbers():
    result = validate_grounding(
        "On 2026-10-01, AQI is 74.",
        ["Forecast for 2026-10-01: AQI 74."],
    )
    assert result.supported is True, result


def test_safe_provenance_does_not_support_claim_that_demo_data_is_live():
    result = validate_grounding(
        "Hospital A is a live government feed.",
        ["Hospital A is listed.", "synthetic_demo"],
    )
    assert result.supported is False
    assert "unsupported_live_status" in result.reason_codes
    assert "unsupported_source_attribution" in result.reason_codes


def test_closure_claim_does_not_drop_source_uncertainty():
    result = validate_grounding(
        "Schools are closed.", ["Schools might be closed."]
    )
    assert result.supported is False
    assert "unsupported_closure" in result.reason_codes


@pytest.mark.parametrize("word", ["generally", "hourly"])
def test_unreviewed_semantic_or_derivational_variations_are_not_whitelisted(word):
    result = validate_grounding(f"Traffic is {word} moderate.", ["Traffic is moderate."])
    assert result.supported is False
    assert "unsupported_terms" in result.reason_codes


def test_validation_result_reports_safe_categories_and_internal_details():
    result = validate_grounding(
        "Traffic is heavy because of construction.", ["Construction is nearby."]
    )
    assert result.supported is False
    assert result.reason_codes == ["unsupported_causality", "unsupported_terms"]
    assert result.unsupported_tokens
    assert result.high_risk_claims == ["unsupported_causality"]
