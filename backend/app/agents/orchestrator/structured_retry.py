"""Shared one-retry policy for recoverable Gemini structured-output failures."""

from __future__ import annotations

from typing import Any


MAX_STRUCTURED_ATTEMPTS = 2
RECOVERABLE_STRUCTURED_FAILURES = frozenset({
    "invalid_structured_output",
    "empty_structured_output",
    "unsupported_content",
    "ungrounded_location",
    "ungrounded_temporal_expression",
    "required_entity_missing",
})


def is_recoverable_structured_failure(category: str) -> bool:
    return category in RECOVERABLE_STRUCTURED_FAILURES


def correction_payload(category: str, schema: dict[str, Any], *, instruction: str) -> dict[str, Any]:
    """Build server-authored retry context; callers must keep evidence untrusted."""
    return {
        "retry_reason": "structured_output_validation_failed",
        "validation_issue": category,
        "required_schema": schema,
        "instruction": instruction,
    }
