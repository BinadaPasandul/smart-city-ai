"""Local-first request understanding with structured Gemini and keyword fallbacks."""

import asyncio
import logging
import re
from collections.abc import Sequence

from app.agents.orchestrator.gemini_router import GeminiQueryRouter, GeminiRoutingError
from app.agents.orchestrator.router import (
    DeterministicQueryRouter,
    MissingInformation,
    RoutingDecision,
    RoutingResult,
    SpecialistAgentName,
)
from app.core.config import Settings, get_settings
from app.nlp.entities import EntityExtractor, EntityModelUnavailable, SpacyEntityExtractor
from app.nlp.intent import LocalIntentClassifier
from app.nlp.models import NlpAnalysisResult
from app.nlp.normalizer import normalize_text
from app.nlp.temporal import DateparserTemporalExtractor, TemporalExtractor

logger = logging.getLogger(__name__)

_LOCATION_REFERENCE = re.compile(
    r"\b(?:there|here|nearby|that place|that area|this area|in that area)\b", re.I
)
_LOCATION_CUE = re.compile(
    r"\b(?:near|nearby|in|at|around|from|to|there|here|nearest)\b", re.I
)
_REQUEST_CUE = re.compile(
    r"\b(?:where|what|when|how|is|are|will|can|could|find|locate|show|tell|check|"
    r"explain|need|want|report|help|look)\b",
    re.I,
)


class LocalNlpAnalyzer:
    """Combine local entity, temporal, and shared keyword intent signals."""

    def __init__(
        self,
        *,
        entity_extractor: EntityExtractor | None = None,
        temporal_extractor: TemporalExtractor | None = None,
        intent_classifier: LocalIntentClassifier | None = None,
        spacy_model: str | None = None,
    ) -> None:
        self._entities = entity_extractor or SpacyEntityExtractor(
            spacy_model or get_settings().nlp_spacy_model
        )
        self._temporal = temporal_extractor or DateparserTemporalExtractor()
        self._intent = intent_classifier or LocalIntentClassifier()

    def analyze(self, text: str) -> NlpAnalysisResult:
        normalized = normalize_text(text)
        local_available = True
        try:
            locations = self._unique(self._entities.extract_locations(text))
        except EntityModelUnavailable:
            local_available = False
            locations = []
            logger.warning("Local NLP entity model unavailable; semantic fallback may be used")
        except Exception as exc:
            local_available = False
            locations = []
            logger.warning("Local NLP entity extraction failed exception_type=%s", type(exc).__name__)

        try:
            temporal = self._unique(self._temporal.extract(text))
        except Exception as exc:
            temporal = []
            logger.warning("Local NLP temporal extraction failed exception_type=%s", type(exc).__name__)

        intent = self._intent.classify(normalized)
        deictic_location = bool(_LOCATION_REFERENCE.search(text))
        location_cue_without_entity = bool(_LOCATION_CUE.search(text)) and not locations
        missing: list[MissingInformation] = []
        if not intent.candidate_agents:
            missing.append(MissingInformation.INTENT)
        if (deictic_location or location_cue_without_entity) and not locations:
            missing.append(MissingInformation.LOCATION)

        score = self._confidence(
            text,
            candidates=intent.candidate_agents,
            matched_term_count=len(intent.matched_terms),
            locations=locations,
            temporal=temporal,
            deictic_location=deictic_location,
            local_available=local_available,
        )
        return NlpAnalysisResult(
            original_text=text,
            normalized_text=normalized,
            locations=locations,
            temporal_expressions=temporal,
            candidate_agents=intent.candidate_agents,
            confidence=score,
            missing_information=missing,
            local_nlp_available=local_available,
        )

    @staticmethod
    def _unique(values: Sequence[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            value = item.strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                result.append(value)
        return result

    @staticmethod
    def _confidence(
        text: str,
        *,
        candidates: list[SpecialistAgentName],
        matched_term_count: int,
        locations: list[str],
        temporal: list[str],
        deictic_location: bool,
        local_available: bool,
    ) -> float:
        """Heuristic score: intent .80, then extraction/request signals; see README."""
        if not candidates:
            score = 0.30 if deictic_location else 0.0
        else:
            score = 0.80
            score += min(0.08, max(0, matched_term_count - 1) * 0.04)
            if _REQUEST_CUE.search(text) or "?" in text:
                score += 0.06
            if locations:
                score += 0.08
            if temporal:
                score += 0.06
            score += min(0.08, max(0, len(candidates) - 1) * 0.04)
        if deictic_location:
            score -= 0.28
        if not local_available:
            score -= 0.20
        return round(min(1.0, max(0.0, score)), 2)


class RequestUnderstandingPipeline:
    """Prefer local NLP, then structured Gemini, then existing keyword routing."""

    def __init__(
        self,
        *,
        analyzer: LocalNlpAnalyzer | None = None,
        gemini_router: GeminiQueryRouter | None = None,
        deterministic_router: DeterministicQueryRouter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._analyzer = analyzer or LocalNlpAnalyzer(
            spacy_model=self._settings.nlp_spacy_model
        )
        self._gemini = gemini_router or GeminiQueryRouter(
            self._settings.gemini_api_key, model=self._settings.gemini_model
        )
        self._deterministic = deterministic_router or DeterministicQueryRouter()

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        """Return one structured understanding decision while retaining original query data."""
        analysis = (
            await asyncio.to_thread(self._analyzer.analyze, query)
            if self._settings.nlp_enabled
            else None
        )

        if (
            analysis is not None
            and analysis.local_nlp_available
            and analysis.candidate_agents
            and not self._requires_location_clarification(analysis)
            and analysis.confidence >= self._settings.nlp_local_confidence_threshold
        ):
            result = self._from_analysis(analysis, routing_method="local_nlp")
            self._log_result(request_id, result)
            return result

        if self._settings.nlp_gemini_fallback_enabled:
            try:
                decision = await self._gemini.understand(
                    query,
                    local_analysis=(
                        {
                            "locations": analysis.locations,
                            "temporal_expressions": analysis.temporal_expressions,
                            "candidate_agents": [agent.value for agent in analysis.candidate_agents],
                            "missing_information": [item.value for item in analysis.missing_information],
                        }
                        if analysis is not None
                        else {}
                    ),
                    request_id=request_id,
                )
                decision = self._validate_grounding(query, analysis, decision)
                if decision.needs_clarification:
                    result = self._gemini_result(
                        decision, analysis, method="gemini_fallback"
                    )
                    self._log_result(request_id, result, fallback_attempted=True)
                    return result
                if (
                    decision.agent_names
                    and decision.confidence >= self._settings.nlp_local_confidence_threshold
                ):
                    result = self._gemini_result(
                        decision, analysis, method="gemini_fallback"
                    )
                    self._log_result(request_id, result, fallback_attempted=True)
                    return result
                fallback_category = "low_confidence_or_no_intent"
            except GeminiRoutingError as exc:
                fallback_category = exc.category
            except Exception as exc:
                fallback_category = type(exc).__name__
        else:
            fallback_category = "gemini_fallback_disabled"

        deterministic = await self._deterministic.route(query, request_id=request_id)
        if analysis is not None and self._requires_location_clarification(analysis):
            decision = RoutingDecision(
                agent_names=[],
                confidence=analysis.confidence,
                reason="The request refers to a location that was not provided.",
                needs_clarification=True,
                locations=analysis.locations,
                temporal_expressions=analysis.temporal_expressions,
                missing_information=self._unique_missing(
                    [*analysis.missing_information, MissingInformation.LOCATION]
                ),
            )
            result = self._result(
                decision,
                analysis,
                routing_method="deterministic_fallback",
                understanding_method="clarification",
                gemini_used=False,
            )
        elif analysis is not None:
            result = self._from_analysis(
                analysis,
                routing_method="deterministic_fallback",
                decision=deterministic.decision,
                gemini_used=False,
            )
        else:
            result = deterministic.model_copy(
                update={
                    "understanding_method": "deterministic_fallback",
                    "gemini_understanding_fallback_used": False,
                }
            )

        logger.warning(
            "Request understanding fallback request_id=%s reason=%s selected_agents=%s",
            request_id or "unavailable",
            fallback_category,
            [name.value for name in result.decision.agent_names] or "none",
        )
        self._log_result(
            request_id,
            result,
            fallback_attempted=self._settings.nlp_gemini_fallback_enabled,
        )
        return result

    @staticmethod
    def _requires_location_clarification(analysis: NlpAnalysisResult) -> bool:
        return (
            MissingInformation.LOCATION in analysis.missing_information
            and bool(_LOCATION_REFERENCE.search(analysis.original_text))
        )

    @staticmethod
    def _unique_missing(items: list[MissingInformation]) -> list[MissingInformation]:
        return list(dict.fromkeys(items))

    def _from_analysis(
        self,
        analysis: NlpAnalysisResult,
        *,
        routing_method: str,
        decision: RoutingDecision | None = None,
        gemini_used: bool = False,
    ) -> RoutingResult:
        if decision is None:
            decision = RoutingDecision(
                agent_names=analysis.candidate_agents,
                confidence=analysis.confidence,
                reason="Local intent keywords and extracted request details support this routing plan.",
                needs_clarification=False,
                locations=analysis.locations,
                temporal_expressions=analysis.temporal_expressions,
                missing_information=analysis.missing_information,
            )
        else:
            decision = RoutingDecision(
                agent_names=decision.agent_names,
                confidence=decision.confidence,
                reason=decision.reason,
                needs_clarification=decision.needs_clarification,
                locations=self._merge_strings(analysis.locations, decision.locations),
                temporal_expressions=self._merge_strings(
                    analysis.temporal_expressions, decision.temporal_expressions
                ),
                missing_information=self._unique_missing(
                    [*analysis.missing_information, *decision.missing_information]
                ),
            )
        return self._result(
            decision,
            analysis,
            routing_method=routing_method,
            understanding_method=routing_method,
            gemini_used=gemini_used,
        )

    def _gemini_result(self, decision, analysis, *, method: str) -> RoutingResult:
        local = analysis
        merged_locations = self._merge_strings(
            local.locations if local else [], decision.locations
        )
        merged_temporal = self._merge_strings(
            local.temporal_expressions if local else [], decision.temporal_expressions
        )
        merged_missing = decision.missing_information
        if local is not None:
            merged_missing = self._unique_missing(
                [item for item in local.missing_information if item not in (MissingInformation.INTENT,)]
                + list(decision.missing_information)
            )
        normalized_names = self._ordered_agents(decision.agent_names)
        routing_decision = RoutingDecision(
            agent_names=normalized_names,
            confidence=decision.confidence,
            reason=decision.reason,
            needs_clarification=decision.needs_clarification,
            locations=merged_locations,
            temporal_expressions=merged_temporal,
            missing_information=merged_missing,
        )
        return self._result(
            routing_decision,
            analysis,
            routing_method="gemini",
            understanding_method=method,
            gemini_used=True,
        )

    @staticmethod
    def _ordered_agents(names: list[SpecialistAgentName]) -> list[SpecialistAgentName]:
        order = {name: index for index, name in enumerate(SpecialistAgentName)}
        return sorted(names, key=order.__getitem__)

    @staticmethod
    def _merge_strings(first: list[str], second: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in (*first, *second):
            key = item.casefold()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def _result(
        self,
        decision: RoutingDecision,
        analysis: NlpAnalysisResult | None,
        *,
        routing_method: str,
        understanding_method: str,
        gemini_used: bool,
    ) -> RoutingResult:
        return RoutingResult(
            decision=decision,
            routing_method=routing_method,
            understanding_method=understanding_method,
            local_nlp_confidence=analysis.confidence if analysis else None,
            gemini_understanding_fallback_used=gemini_used,
            locations=decision.locations,
            temporal_expressions=decision.temporal_expressions,
            missing_information=decision.missing_information,
            local_nlp_available=analysis.local_nlp_available if analysis else None,
        )

    @staticmethod
    def _validate_grounding(query, analysis, decision):
        """Reject entities Gemini claims that are not grounded in text or local spans."""
        allowed_locations = {normalize_text(value) for value in (analysis.locations if analysis else [])}
        normalized_query = normalize_text(query)
        allowed_times = {
            normalize_text(value) for value in (analysis.temporal_expressions if analysis else [])
        }
        for value in decision.locations:
            normalized = normalize_text(value)
            grounded_in_query = f" {normalized} " in f" {normalized_query} "
            if normalized and not grounded_in_query and normalized not in allowed_locations:
                raise GeminiRoutingError("ungrounded_location")
        for value in decision.temporal_expressions:
            normalized = normalize_text(value)
            grounded_in_query = f" {normalized} " in f" {normalized_query} "
            if normalized and not grounded_in_query and normalized not in allowed_times:
                raise GeminiRoutingError("ungrounded_temporal_expression")
        return decision

    @staticmethod
    def _log_result(
        request_id: str | None,
        result: RoutingResult,
        *,
        fallback_attempted: bool = False,
    ) -> None:
        logger.info(
            "Request understood request_id=%s method=%s local_confidence=%s local_available=%s "
            "gemini_fallback_used=%s gemini_fallback_attempted=%s selected_agents=%s",
            request_id or "unavailable",
            result.understanding_method or result.routing_method,
            result.local_nlp_confidence,
            result.local_nlp_available,
            result.gemini_understanding_fallback_used,
            fallback_attempted,
            [agent.value for agent in result.decision.agent_names],
        )
