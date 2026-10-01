"""Optional live acceptance checks; uses real providers and prints no secrets/data."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

# Allow both ``python scripts/verify_system_acceptance.py`` and module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse, AgentSource
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.gemini_router import GeminiQueryRouter
from app.agents.orchestrator.router import RoutingDecision, RoutingResult
from app.agents.orchestrator.synthesizer import DeterministicResultSynthesizer, GeminiResultSynthesizer
from app.agents.orchestrator.web_search import WebSearchService
from app.agents.registry import AgentRegistry
from app.core.bootstrap import create_agent_registry
from app.core.config import get_settings
from app.ir.web_search import TavilyWebSearchProvider
from app.nlp.pipeline import RequestUnderstandingPipeline


class FixedRouter:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        return RoutingResult(
            decision=RoutingDecision(
                agent_names=list(self.names), confidence=1.0,
                reason="Manual live acceptance route.", needs_clarification=False,
                locations=["Colombo"],
            ),
            routing_method="deterministic_fallback",
            understanding_method="deterministic_fallback",
            local_nlp_confidence=1.0,
            locations=["Colombo"],
        )


class FailingOrSuccessfulAgent(BaseAgent):
    def __init__(self, name: str, *, fail: bool = False) -> None:
        super().__init__(name)
        self.fail = fail

    async def execute(self, request: AgentRequest) -> AgentResponse:
        if self.fail:
            return AgentResponse(
                request_id=request.request_id, agent_name=self.name, success=False,
                error=AgentError(
                    code=AgentErrorCode.AGENT_EXECUTION_FAILED,
                    message="Controlled live fallback verification failure.",
                ),
            )
        return AgentResponse(
            request_id=request.request_id, agent_name=self.name, success=True,
            answer=f"Controlled evidence for {self.name}.",
            sources=[AgentSource(name=f"controlled {self.name}", source_type="test")],
        )


class EvidenceAgent(BaseAgent):
    def __init__(self, name: str, answer: str) -> None:
        super().__init__(name)
        self.answer = answer

    async def execute(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=self.answer,
            sources=[AgentSource(name=f"{self.name} evidence", source_type="test")],
        )


class CountedTavily:
    provider_name = "tavily"

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0

    async def search(self, query: str, max_results: int):
        self.calls += 1
        return await self.delegate.search(query, max_results)

    async def aclose(self) -> None:
        await self.delegate.aclose()


class CapturingGeminiClient:
    """Forward live requests while recording only safe evidence-kind booleans."""

    def __init__(self, api_key: str) -> None:
        from google import genai

        self._delegate = genai.Client(api_key=api_key).aio
        self.synthetic_demo_in_evidence = False
        self.local_static_in_evidence = False
        self.models = self

    async def generate_content(self, **kwargs):
        payload = json.loads(kwargs.get("contents", "{}"))
        evidence = payload.get("untrusted_specialist_evidence", {}).get("successful_results", [])
        kinds = {
            item.get("evidence_attributes", {}).get("data_kind")
            for item in evidence
        }
        self.synthetic_demo_in_evidence |= "synthetic_demo" in kinds
        self.local_static_in_evidence |= "local_static_dataset" in kinds
        return await self._delegate.models.generate_content(**kwargs)

    async def aclose(self) -> None:
        await self._delegate.aclose()


def _report_error(stage: str, exc: Exception) -> None:
    # Provider messages may contain request details; report only exception class.
    print(f"{stage}_ERROR_TYPE={type(exc).__name__}")


async def _live_gemini(settings) -> None:
    if not settings.gemini_api_key:
        print("LIVE_GEMINI_ENTITY_FALLBACK=SKIPPED")
        print("LIVE_GEMINI_CLARIFICATION=SKIPPED")
        return
    query = "What are the traffic conditions on Marine Drive?"
    for run in range(1, 4):
        pipeline = RequestUnderstandingPipeline(settings=settings)
        label = f"RUN_{run}"
        try:
            local = pipeline._analyzer.analyze(query)
            result = await pipeline.route(query)
            passed = (
                not local.locations
                and bool(local.candidate_agents)
                and result.gemini_understanding_fallback_used
                and "Marine Drive" in result.locations
                and [name.value for name in result.decision.agent_names] == ["mobility"]
            )
            print(f"{label}_ENTITY_FALLBACK={'PASS' if passed else 'FAILED'}")
            print(f"{label}_first_attempt={result.gemini_first_attempt_status}")
            print(f"{label}_retry_triggered={str(result.gemini_retry_triggered).lower()}")
            print(f"{label}_final_status={result.gemini_final_status}")
            print(f"{label}_selected_agents=" + ",".join(name.value for name in result.decision.agent_names))
            print(f"{label}_location_present={str('Marine Drive' in result.locations).lower()}")
            print(f"{label}_clarification={str(result.decision.needs_clarification).lower()}")
            print(f"{label}_attempt_count={result.gemini_attempt_count}")
        except Exception as exc:
            print(f"{label}_ENTITY_FALLBACK=FAILED")
            _report_error(label, exc)
        finally:
            router = getattr(pipeline, "_gemini", None)
            if router is not None:
                await router.aclose()

    try:
        clarification = await pipeline.route("How is the traffic there?")
        passed = (
            clarification.gemini_understanding_fallback_used
            and clarification.decision.needs_clarification
            and not clarification.decision.agent_names
            and not clarification.locations
        )
        print(f"LIVE_GEMINI_CLARIFICATION={'PASS' if passed else 'FAILED'}")
    except Exception as exc:
        print("LIVE_GEMINI_CLARIFICATION=FAILED")
        _report_error("LIVE_GEMINI_CLARIFICATION", exc)
    finally:
        router = getattr(pipeline, "_gemini", None)
        if router is not None:
            await router.aclose()


def _make_live_orchestrator(names: tuple[str, ...], settings):
    live_client = None
    if settings.gemini_api_key:
        live_client = CapturingGeminiClient(settings.gemini_api_key)
    primary = (
        GeminiResultSynthesizer(
            settings.gemini_api_key, model=settings.gemini_model, client=live_client
        )
        if settings.gemini_api_key
        else DeterministicResultSynthesizer()
    )
    orchestrator = CityOrchestratorAgent(
        create_agent_registry(),
        FixedRouter(names),
        synthesizer=primary,
        web_search=WebSearchService(None, enabled=False),
        execution_timeout_seconds=settings.agent_execution_timeout_seconds,
    )
    return orchestrator, primary


async def _live_specialists(settings) -> None:
    coordinates = {"user_context": {"latitude": 6.9271, "longitude": 79.8612}}
    cases = [
        ("LIVE_ENVIRONMENT", ("environment",), "What is the weather forecast?", 1),
        (
            "LIVE_TWO_AGENT",
            ("mobility", "environment"),
            "How is traffic near Colombo Fort and what is the air quality forecast?",
            3,
        ),
        (
            "LIVE_THREE_AGENT",
            ("mobility", "environment", "public_services"),
            "How is traffic near Colombo Fort, what is the air quality in Colombo, and where is a hospital in Colombo?",
            3,
        ),
    ]
    for label, names, query, run_count in cases:
        for run in range(1, run_count + 1):
            orchestrator, synthesizer = _make_live_orchestrator(names, settings)
            request = AgentRequest(request_id=uuid4(), query=query, context=coordinates)
            run_label = f"{label}_RUN_{run}" if run_count > 1 else label
            try:
                response = await orchestrator.execute(request)
                summary = response.metadata.get("execution_summary", {})
                expected_sources = {
                    "mobility": any(source.metadata.get("category") == "traffic" for source in response.sources),
                    "environment": any(source.source_type == "api" for source in response.sources),
                    "public_services": any(source.metadata.get("category") == "hospitals" for source in response.sources),
                }
                passed = (
                    response.success
                    and response.request_id == request.request_id
                    and response.metadata.get("selected_agents") == list(names)
                    and response.metadata.get("execution_status") == "complete"
                    and all(expected_sources[name] for name in names)
                    and len(summary.get("successful_agents", [])) == len(names)
                )
                print(f"{run_label}={'PASS' if passed else 'FAILED'}")
                print(f"{run_label}_first_attempt_status={response.metadata.get('gemini_synthesis_first_attempt_status')}")
                print(f"{run_label}_retry_triggered={str(response.metadata.get('gemini_synthesis_retry_triggered', False)).lower()}")
                print(f"{run_label}_retry_reason={response.metadata.get('gemini_synthesis_retry_reason')}")
                print(f"{run_label}_retry_result={response.metadata.get('gemini_synthesis_final_status')}")
                print(f"{run_label}_final_synthesis_method={response.metadata.get('synthesis_method', 'single_specialist_passthrough')}")
                print(f"{run_label}_sources_preserved={str(all(expected_sources[name] for name in names)).lower()}")
                print(f"{run_label}_source_count={len(response.sources)}")
                print(f"{run_label}_execution_status={response.metadata.get('execution_status')}")
                if len(names) == 3:
                    print(f"{run_label}_synthetic_demo_in_evidence={str(synthesizer._client.synthetic_demo_in_evidence).lower()}")
            except Exception as exc:
                print(f"{run_label}=FAILED")
                _report_error(run_label, exc)
            finally:
                primary = getattr(synthesizer, "_primary", synthesizer)
                close = getattr(primary, "aclose", None)
                if close is not None:
                    await close()


async def _live_multi_synthesis(settings) -> None:
    if not settings.gemini_api_key:
        print("LIVE_GEMINI_MULTI_SYNTHESIS=SKIPPED")
        return
    for names in (("mobility", "environment"), ("mobility", "environment", "public_services")):
        registry = AgentRegistry()
        registry.register(EvidenceAgent("mobility", "Traffic is heavy."))
        registry.register(EvidenceAgent("environment", "Air quality is moderate."))
        if "public_services" in names:
            registry.register(EvidenceAgent("public_services", "A hospital record is listed."))
        synthesizer = GeminiResultSynthesizer(settings.gemini_api_key, model=settings.gemini_model)
        orchestrator = CityOrchestratorAgent(
            registry,
            FixedRouter(names),
            synthesizer=synthesizer,
            web_search=WebSearchService(None, enabled=False),
        )
        label = "two" if len(names) == 2 else "three"
        try:
            response = await orchestrator.execute(
                AgentRequest(query="Combine only the supplied specialist evidence.")
            )
            passed = (
                response.metadata.get("synthesis_method") in {"gemini", "gemini_retry"}
                and response.metadata.get("synthesis_used_agents") == list(names)
                and response.metadata.get("execution_status") == "complete"
                and len(response.sources) == len(names)
            )
            print(f"LIVE_GEMINI_{label.upper()}_SYNTHESIS={'PASS' if passed else 'FAILED'}")
            print(f"LIVE_GEMINI_{label.upper()}_SOURCES_PRESERVED={len(response.sources) == len(names)}")
            print(f"LIVE_GEMINI_{label.upper()}_FIRST_ATTEMPT={response.metadata.get('gemini_synthesis_first_attempt_status')}")
            print(f"LIVE_GEMINI_{label.upper()}_RETRY={str(response.metadata.get('gemini_synthesis_retry_triggered', False)).lower()}")
            print(f"LIVE_GEMINI_{label.upper()}_RETRY_RESULT={response.metadata.get('gemini_synthesis_final_status')}")
            print(f"LIVE_GEMINI_{label.upper()}_FINAL_METHOD={response.metadata.get('synthesis_method')}")
            print(f"LIVE_GEMINI_{label.upper()}_REQUEST_COMPLETED={str(response.success).lower()}")
        except Exception as exc:
            print(f"LIVE_GEMINI_{label.upper()}_SYNTHESIS=FAILED")
            _report_error(f"LIVE_GEMINI_{label.upper()}_SYNTHESIS", exc)
        finally:
            await synthesizer.aclose()


async def _live_tavily(settings) -> None:
    if not settings.web_search_enabled or not settings.tavily_api_key:
        print("LIVE_TAVILY_TOTAL_FAILURE=SKIPPED")
        print("LIVE_TAVILY_PARTIAL=SKIPPED")
        return

    async def run_case(label: str, names: tuple[str, ...], fail_names: set[str]):
        from app.agents.registry import AgentRegistry

        registry = AgentRegistry()
        for name in names:
            registry.register(FailingOrSuccessfulAgent(name, fail=name in fail_names))
        counted = CountedTavily(TavilyWebSearchProvider(
            settings.tavily_api_key.get_secret_value(),
            timeout_seconds=settings.web_search_timeout_seconds,
        ))
        service = WebSearchService(
            counted, enabled=True,
            on_partial_failure=settings.web_search_on_partial_failure,
            timeout_seconds=settings.web_search_timeout_seconds,
            max_results=settings.web_search_max_results,
            query_max_length=settings.web_search_query_max_length,
        )
        orchestrator = CityOrchestratorAgent(
            registry, FixedRouter(names),
            synthesizer=DeterministicResultSynthesizer(),
            web_search=service,
        )
        try:
            response = await orchestrator.execute(AgentRequest(
                query="Official air quality and city traffic updates in Colombo."
            ))
            expected_status = "failed" if len(fail_names) == len(names) else "partial_success"
            passed = (
                counted.calls == 1
                and response.metadata.get("execution_status") == expected_status
                and response.metadata.get("web_search_used") is True
                and response.metadata.get("web_search_status") == "success"
                and response.metadata.get("web_search_result_count", 0) > 0
                and any(source.source_type == "web_search" and source.url for source in response.sources)
            )
            print(f"{label}={'PASS' if passed else 'FAILED'}")
            print(f"{label}_CALLS={counted.calls}")
            print(f"{label}_WEB_RESULT_COUNT={response.metadata.get('web_search_result_count', 0)}")
            print(f"{label}_EXECUTION_STATUS={response.metadata.get('execution_status')}")
        except Exception as exc:
            print(f"{label}=FAILED")
            _report_error(label, exc)
        finally:
            await service.aclose()

    await run_case("LIVE_TAVILY_TOTAL_FAILURE", ("mobility",), {"mobility"})
    await run_case(
        "LIVE_TAVILY_PARTIAL", ("mobility", "environment"), {"environment"}
    )


async def main() -> None:
    logging.disable(logging.CRITICAL)
    settings = get_settings()
    print("LIVE_GEMINI_CONFIGURED=" + str(bool(settings.gemini_api_key)))
    print("LIVE_TAVILY_CONFIGURED=" + str(bool(settings.web_search_enabled and settings.tavily_api_key)))
    await _live_gemini(settings)
    await _live_specialists(settings)
    await _live_multi_synthesis(settings)
    await _live_tavily(settings)


if __name__ == "__main__":
    asyncio.run(main())
