"""Conservative, explainable grounding checks for generated synthesis text."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GroundingValidationResult:
    """Internal validation details; do not serialize directly to API responses."""

    supported: bool
    unsupported_tokens: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    unit_mismatches: list[str] = field(default_factory=list)
    high_risk_claims: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    @property
    def retry_reason(self) -> str:
        """Return one safe category suitable for constrained model feedback."""
        return self.reason_codes[0] if self.reason_codes else "unsupported_terms"


_ACRONYM_PHRASES = (
    (re.compile(r"\bair\s+quality\s+index\b", re.I), "aqi"),
    (re.compile(r"\bparticulate\s+matter\s*2\s*\.\s*5\b", re.I), "pm25"),
    (re.compile(r"\bpm\s*2\s*\.\s*5\b", re.I), "pm25"),
    (re.compile(r"\bkm\s*/\s*h\b|\bkm\s+per\s+hour\b", re.I), "kmh"),
    (re.compile(r"\btraffic\s+conditions?\b", re.I), "traffic"),
)

_TOKEN_FORMS = {
    "location": "locate", "locations": "locate", "located": "locate",
    "locating": "locate", "situated": "locate",
    "report": "report", "reports": "report", "reported": "report",
    "reporting": "report",
    "condition": "condition", "conditions": "condition",
    "range": "range", "ranges": "range", "ranged": "range", "ranging": "range",
    "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute",
    "hour": "hour", "hours": "hour",
    "closed": "closure", "closing": "closure", "closure": "closure", "closures": "closure",
    "causes": "cause", "caused": "cause", "causing": "cause", "cause": "cause",
    "delays": "delay", "delay": "delay",
}

_SAFE_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "while", "whereas", "is", "are",
    "was", "were", "be", "been", "being", "of", "to", "for", "in", "on",
    "at", "by", "with", "from", "near", "as", "it", "this", "that", "these",
    "those", "could", "not", "successfully", "report", "reported",
})

_CAUSAL_MARKER = re.compile(
    r"\b(?:because(?:\s+of)?|due\s+to|caused\s+by|caus(?:e|es|ed|ing)|"
    r"result(?:s|ed)?\s+(?:in|from)|lead(?:s|ing)?\s+to)\b",
    re.I,
)
_CAUSAL_CAUSE_FIRST = re.compile(
    r"(?P<cause>[^.!?;]+?)\s+\b(?:causes?|caused|causing|leads?\s+to|"
    r"result(?:s|ed)?\s+in)\s+(?P<effect>[^.!?;]+)", re.I,
)
_CAUSAL_EFFECT_FIRST = re.compile(
    r"(?P<effect>[^.!?;]+?)\s+\b(?:because(?:\s+of)?|due\s+to|caused\s+by|"
    r"result(?:s|ed)?\s+from)\s+(?P<cause>[^.!?;]+)", re.I,
)
_CLOSURE_CLAIM = re.compile(
    r"\b(?P<subject>[^.;:!?\n]{1,48}?)\s+"
    r"(?:(?:is|are|was|were|remain|remains)\s+)?(?:currently\s+)?closed\b", re.I,
)
_ISO_DATE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_CLOCK_TIME = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[ap]m)?\b", re.I)
_TEMPORAL_PATTERNS = (
    re.compile(r"\b(?:today|yesterday|tomorrow|now|currently|current|live|real[- ]time|"
               r"this\s+(?:morning|afternoon|evening|week|month|year)|tonight)\b", re.I),
    re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|"
               r"october|november|december)\b", re.I),
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9])(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9])")
_NAMED_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*|\s+[A-Z]){1,4}\b")
_SINGLE_ENTITY_AFTER_PREPOSITION = re.compile(
    r"\b(?:in|near|at|from|to|around|outside)\s+([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*)\b"
)
_ENTITY_STARTS = frozenset({"The", "A", "An", "This", "That", "Traffic", "Air", "Speed", "Hospital"})
_HIGH_RISK_TERMS = {
    "accident": "unsupported_emergency_or_accident",
    "crash": "unsupported_emergency_or_accident",
    "emergency": "unsupported_emergency_or_accident",
    "closed": "unsupported_closure",
    "closure": "unsupported_closure",
    "school": "unsupported_closure",
    "live": "unsupported_live_status",
    "current": "unsupported_live_status",
    "currently": "unsupported_live_status",
    "official": "unsupported_source_attribution",
    "government": "unsupported_source_attribution",
    "provider": "unsupported_source_attribution",
    "feed": "unsupported_source_attribution",
}


@dataclass(frozen=True, slots=True)
class _NumericFact:
    value: Decimal
    unit: str | None
    display: str


def _replace_known_phrases(text: str) -> str:
    normalized = text
    for pattern, replacement in _ACRONYM_PHRASES:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _lemma(token: str) -> str:
    if token in _TOKEN_FORMS:
        return _TOKEN_FORMS[token]
    # Conservative English inflections only. No stemming of arbitrary words.
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if len(token) > 6 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 6 and token.endswith("ed"):
        stem = token[:-2]
        return stem[:-1] if len(stem) > 3 and stem[-1:] == stem[-2:-1] else stem
    if len(token) > 7 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 3 and stem[-1:] == stem[-2:-1]:
            stem = stem[:-1]
        return stem
    return token


def _tokens(text: str) -> list[str]:
    normalized = _replace_known_phrases(text.casefold())
    return [_lemma(token) for token in re.findall(r"[a-z]+", normalized)]


def _numeric_facts(text: str) -> list[_NumericFact]:
    # PM2.5 is a pollutant label, not the numeric fact 2.5.
    masked = re.sub(
        r"\b(?:pm\s*2\s*\.\s*5|particulate\s+matter\s*2\s*\.\s*5)\b",
        "PM25_LABEL", text, flags=re.I,
    )
    masked = _ISO_DATE.sub(" DATE_LABEL ", masked)
    masked = _CLOCK_TIME.sub(" TIME_LABEL ", masked)
    facts: list[_NumericFact] = []
    for match in _NUMBER.finditer(masked):
        try:
            value = Decimal(match.group("value").replace(",", ""))
        except InvalidOperation:
            continue
        start, end = match.span()
        before = masked[max(0, start - 40):start].casefold()
        after = masked[end:end + 30].casefold()
        suffix_unit = re.match(
            r"\s*(%|percent(?:age)?\b|km\s*/\s*h\b|km\s+per\s+hour\b|"
            r"miles?\s+per\s+hour\b|minutes?\b|mins?\b|hours?\b|"
            r"seconds?\b|secs?\b|kilomet(?:er|re)s?\b|meters?\b|"
            r"(?:µg|μg|ug)\s*/\s*m(?:³|3)|mg\s*/\s*l\b|ppm\b|ppb\b|"
            r"degrees?\s*[cf]\b|°\s*[cf])", after,
        )
        if suffix_unit:
            unit_text = suffix_unit.group(1).replace(" ", "")
            if unit_text in {"%", "percent", "percentage"}:
                unit = "percent"
            elif unit_text in {"km/h", "kmperhour"}:
                unit = "km_per_hour"
            elif unit_text in {"mileperhour", "milesperhour"}:
                unit = "miles_per_hour"
            elif unit_text.startswith(("minute", "min")):
                unit = "minute"
            elif unit_text.startswith("hour"):
                unit = "hour"
            elif unit_text.startswith(("second", "sec")):
                unit = "second"
            elif unit_text.startswith(("kilometer", "kilometre")):
                unit = "kilometer"
            elif unit_text.startswith("meter"):
                unit = "meter"
            elif unit_text in {"µg/m³", "μg/m³", "ug/m3"}:
                unit = "microgram_per_cubic_meter"
            elif unit_text == "mg/l":
                unit = "milligram_per_liter"
            elif unit_text in {"ppm", "ppb"}:
                unit = unit_text
            elif "°" in unit_text or unit_text.startswith("degree"):
                unit = "temperature_" + ("f" if "f" in unit_text else "c")
            else:
                unit = None
        else:
            unit = None
        prefix_unit = re.search(r"\b(?:lkr|usd|eur|gbp)\s*$", before)
        if unit is None and prefix_unit:
            unit = prefix_unit.group(0).strip()
        context = before + " " + after
        if unit is None:
            context_labels = [
                ("aqi_index", re.search(r"\b(?:aqi|air\s+quality\s+index)\b", context)),
                ("pm25", re.search(r"\bpm25_label\b", context)),
            ]
            labels = [(name, match) for name, match in context_labels if match]
            if labels:
                context_number = len(before)
                preceding = [item for item in labels if item[1].end() <= context_number]
                nearest = (
                    max(preceding, key=lambda item: item[1].end())
                    if preceding
                    else min(labels, key=lambda item: item[1].start())
                )
                unit = nearest[0]
        facts.append(_NumericFact(value=value, unit=unit, display=match.group("value")))
    return facts


def _date_facts(text: str) -> set[str]:
    return ({date.casefold() for date in _ISO_DATE.findall(text)}
            | {f"time:{match.group(0).replace(' ', '').casefold()}" for match in _CLOCK_TIME.finditer(text)})


def _temporal_facts(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in _TEMPORAL_PATTERNS:
        found.update(re.sub(r"\s+", " ", match.group(0).casefold()) for match in pattern.finditer(text))
    return found


def _relation_tokens(text: str) -> tuple[str, ...]:
    return tuple(sorted(
        token for token in _tokens(text)
        if token not in _SAFE_FUNCTION_WORDS and token not in {"because", "due", "cause", "result", "lead"}
    ))


def _causal_relations(text: str) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    relations: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for pattern in (_CAUSAL_CAUSE_FIRST, _CAUSAL_EFFECT_FIRST):
        for match in pattern.finditer(text):
            cause = _relation_tokens(match.group("cause"))
            effect = _relation_tokens(match.group("effect"))
            if cause and effect:
                relations.add((cause, effect))
    return relations


def _closure_claims(text: str) -> set[tuple[str, ...]]:
    claims = set()
    for match in _CLOSURE_CLAIM.finditer(text):
        subject = _relation_tokens(match.group("subject"))
        if subject:
            claims.add(subject)
    return claims


def _entity_phrases(text: str) -> set[str]:
    entities: set[str] = set()
    for sentence in re.split(r"[.!?\n]+", text):
        sentence = re.sub(r"^\s*(?:[-*#]+|\d+[.)])\s*", "", sentence)
        for match in _NAMED_PHRASE.finditer(sentence):
            phrase = match.group(0)
            first = phrase.split()[0]
            if first not in _ENTITY_STARTS:
                entities.add(" ".join(_tokens(phrase)))
        for match in _SINGLE_ENTITY_AFTER_PREPOSITION.finditer(sentence):
            entities.add(" ".join(_tokens(match.group(1))))
    return entities


def validate_grounding(answer: str, evidence: Iterable[str]) -> GroundingValidationResult:
    """Check text, temporal/numeric facts, units, entities, and causality."""
    evidence_items = list(evidence)
    evidence_text = "\n".join(evidence_items)
    evidence_tokens = set(_tokens(evidence_text))
    answer_tokens = set(_tokens(answer))
    unsupported_tokens = sorted(answer_tokens - evidence_tokens - _SAFE_FUNCTION_WORDS)

    evidence_counter = Counter((fact.value, fact.unit) for fact in _numeric_facts(evidence_text))
    answer_facts = _numeric_facts(answer)
    unsupported_numbers: list[str] = []
    unit_mismatches: list[str] = []
    for fact in answer_facts:
        key = (fact.value, fact.unit)
        if evidence_counter[key] > 0:
            evidence_counter[key] -= 1
            continue
        alternate_units = sorted({
            unit or "unitless"
            for (value, unit), count in evidence_counter.items()
            if value == fact.value and count > 0
        })
        if alternate_units:
            unit_mismatches.append(
                f"{fact.display}:{'|'.join(alternate_units)}->{fact.unit or 'unitless'}"
            )
        else:
            unsupported_numbers.append(fact.display)

    evidence_dates = _date_facts(evidence_text)
    answer_dates = _date_facts(answer)
    unsupported_dates = sorted(answer_dates - evidence_dates)
    evidence_temporal = _temporal_facts(evidence_text)
    answer_temporal = _temporal_facts(answer)
    unsupported_temporal = sorted(answer_temporal - evidence_temporal)

    evidence_entities = _entity_phrases(evidence_text)
    unsupported_entities = sorted(_entity_phrases(answer) - evidence_entities)

    evidence_relations = _causal_relations(evidence_text)
    answer_has_causal_marker = bool(_CAUSAL_MARKER.search(answer))
    answer_relations = _causal_relations(answer)
    unsupported_causality = answer_has_causal_marker and (
        not answer_relations or not answer_relations.issubset(evidence_relations)
    )
    answer_closures = _closure_claims(answer)
    unsupported_closure_claim = bool(
        answer_closures and not answer_closures.issubset(_closure_claims(evidence_text))
    )

    high_risk_claims: list[str] = []
    evidence_risk_tokens = set(_tokens(evidence_text))
    answer_risk_tokens = set(_tokens(answer))
    risk_codes = {
        code for token, code in _HIGH_RISK_TERMS.items()
        if _lemma(token) in answer_risk_tokens and _lemma(token) not in evidence_risk_tokens
    }
    if unsupported_causality:
        risk_codes.add("unsupported_causality")
    if unsupported_closure_claim:
        risk_codes.add("unsupported_closure")
    if unsupported_dates or unsupported_temporal:
        risk_codes.add("unsupported_temporal_claim")
    if unsupported_entities:
        risk_codes.add("unsupported_named_entity")
    if unit_mismatches:
        risk_codes.add("unsupported_numeric_unit")
    if unsupported_numbers:
        risk_codes.add("unsupported_numeric_fact")
    high_risk_claims = sorted(risk_codes)

    reason_codes: list[str] = []
    for code in (
        "unsupported_causality", "unsupported_temporal_claim", "unsupported_numeric_unit",
        "unsupported_numeric_fact", "unsupported_named_entity", "unsupported_emergency_or_accident",
        "unsupported_closure", "unsupported_live_status", "unsupported_source_attribution",
    ):
        if code in risk_codes:
            reason_codes.append(code)
    if unsupported_tokens:
        reason_codes.append("unsupported_terms")

    return GroundingValidationResult(
        supported=not reason_codes,
        unsupported_tokens=unsupported_tokens,
        unsupported_numbers=unsupported_numbers,
        unit_mismatches=unit_mismatches,
        high_risk_claims=high_risk_claims,
        reason_codes=reason_codes,
    )
