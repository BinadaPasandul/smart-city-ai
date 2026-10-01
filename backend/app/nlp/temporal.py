"""Local, best-effort extraction of textual date/time expressions."""

import re
from typing import Protocol


class TemporalExtractor(Protocol):
    def extract(self, text: str) -> list[str]:
        """Return matched temporal phrases without resolving them to a timezone."""


class DateparserTemporalExtractor:
    """Use dateparser's local search API and preserve matched source phrases."""

    def extract(self, text: str) -> list[str]:
        explicit = list(_explicit_temporal_matches(text))
        try:
            from dateparser.search import search_dates

            matches = search_dates(
                text,
                languages=["en"],
                settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
            )
        except Exception:
            return [value for _, _, value in explicit]

        spans = list(explicit)
        for phrase, _parsed in matches or []:
            value = phrase.strip()
            if not value:
                continue
            start = text.casefold().find(value.casefold())
            if start < 0:
                continue
            end = start + len(value)
            if any(start < old_end and end > old_start for old_start, old_end, _ in spans):
                continue
            spans.append((start, end, value))
        spans.sort(key=lambda item: item[0])
        values: list[str] = []
        seen: set[str] = set()
        for _, _, value in spans:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                values.append(value)
        return values


_EXPLICIT_TEMPORAL = re.compile(
    r"\b(?:"
    r"day after tomorrow|tomorrow\s+(?:morning|afternoon|evening|night)|"
    r"this\s+(?:morning|afternoon|evening)|later\s+today|next\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"tonight|today|tomorrow|later|"
    r"\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)"
    r")\b",
    re.IGNORECASE,
)


def _explicit_temporal_matches(text: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), match.group(0).strip())
        for match in _EXPLICIT_TEMPORAL.finditer(text)
    ]
