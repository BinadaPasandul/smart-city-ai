"""Explicit live Gemini and fake-agent verification; never run as part of pytest."""

import asyncio
import re
from typing import Any

from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.gemini_router import (
    FallbackQueryRouter,
    GeminiQueryRouter,
    GeminiRoutingError,
)
from app.agents.orchestrator.router import DeterministicQueryRouter, RoutingResult
from app.agents.registry import AgentRegistry
from app.core.config import get_settings

SINGLE_QUERY = "Where can I park near Colombo Fort?"
MULTI_QUERY = "Considering both traffic and air quality, is it a good time to cycle in Colombo today?"
EXPECTED_MULTI = {"mobility", "environment"}


class FakeSpecialist(BaseAgent):
    """Small verification-only agent that records request IDs and returns canned text."""

    def __init__(self, name: str, answer: str) -> None:
        super().__init__(name)
        self.answer = answer
        self.request_ids: list[str] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.request_ids.append(str(request.request_id))
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=self.answer,
        )


class FailingRouter:
    """Offline failure source used to verify the deterministic fallback path."""

    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        raise GeminiRoutingError("controlled_offline_failure")


def _safe_error_message(exc: Exception, api_key: str | None) -> str:
    message = str(getattr(exc, "message", None) or "Request failed")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2[REDACTED]",
        message,
    )
    return message[:500]


def _report_live_error(stage: str, exc: Exception, api_key: str | None, model: str) -> None:
    code = getattr(exc, "code", None)
    print(f"{stage}=FAILED")
    print(f"MODEL={model}")
    print(f"ERROR_TYPE={type(exc).__name__}")
    print(f"ERROR_CODE={code if code is not None else 'unavailable'}")
    print(f"ERROR_MESSAGE={_safe_error_message(exc, api_key)}")


async def _basic_live_check(api_key: str, model: str) -> bool:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key).aio
    try:
        response = await client.models.generate_content(
            model=model,
            contents="Reply with exactly: OK",
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=8),
        )
        text = response.text or ""
        if "OK" not in text.upper():
            print(f"LIVE_BASIC_GEMINI=FAILED\nMODEL={model}\nERROR_TYPE=UnexpectedResponse")
            print("ERROR_CODE=unavailable\nERROR_MESSAGE=Response did not contain the expected OK token")
            return False
        print("LIVE_BASIC_GEMINI=PASS")
        print(f"MODEL={model}")
        return True
    except Exception as exc:
        _report_live_error("LIVE_BASIC_GEMINI", exc, api_key, model)
        return False
    finally:
        await client.aclose()


def _make_registry() -> tuple[AgentRegistry, dict[str, FakeSpecialist]]:
    agents = {
        "mobility": FakeSpecialist("mobility", "FAKE_MOBILITY_RESULT"),
        "environment": FakeSpecialist("environment", "FAKE_ENVIRONMENT_RESULT"),
        "public_services": FakeSpecialist("public_services", "FAKE_PUBLIC_SERVICES_RESULT"),
    }
    registry = AgentRegistry()
    for agent in agents.values():
        registry.register(agent)
    return registry, agents


async def _verify_fallback() -> bool:
    registry, agents = _make_registry()
    router = FallbackQueryRouter(FailingRouter(), DeterministicQueryRouter())
    response = await CityOrchestratorAgent(registry, router).execute(
        AgentRequest(query="How are traffic and air quality today?")
    )
    passed = (
        response.metadata.get("routing_method") == "deterministic_fallback"
        and set(response.metadata.get("selected_agents", [])) == EXPECTED_MULTI
        and response.metadata.get("execution_status") == "complete"
        and all(agents[name].request_ids for name in EXPECTED_MULTI)
    )
    print(f"FALLBACK_VERIFICATION={'PASS' if passed else 'FAILED'}")
    return passed


async def _run_live_stages(api_key: str, model: str) -> None:
    if not await _basic_live_check(api_key, model):
        print("LIVE_SINGLE_ROUTE=SKIPPED")
        print("LIVE_MULTI_ROUTE=SKIPPED")
        print("LIVE_GEMINI_ORCHESTRATION=SKIPPED")
        return

    router = GeminiQueryRouter(api_key, model=model)
    try:
        try:
            single = await router.route(SINGLE_QUERY)
            single_ok = (
                single.routing_method == "gemini"
                and [name.value for name in single.decision.agent_names] == ["mobility"]
                and not single.decision.needs_clarification
            )
            print(f"LIVE_SINGLE_ROUTE={'PASS' if single_ok else 'FAILED'}")
            print("SINGLE_ROUTING_METHOD=" + single.routing_method)
        except Exception as exc:
            _report_live_error("LIVE_SINGLE_ROUTE", exc, api_key, model)
            print("LIVE_MULTI_ROUTE=SKIPPED")
            print("LIVE_GEMINI_ORCHESTRATION=SKIPPED")
            return

        if not single_ok:
            print("LIVE_MULTI_ROUTE=SKIPPED")
            print("LIVE_GEMINI_ORCHESTRATION=SKIPPED")
            return

        try:
            multi = await router.route(MULTI_QUERY)
            multi_names = {name.value for name in multi.decision.agent_names}
            multi_ok = (
                multi.routing_method == "gemini"
                and EXPECTED_MULTI <= multi_names
                and "public_services" not in multi_names
                and not multi.decision.needs_clarification
            )
            print(f"LIVE_MULTI_ROUTE={'PASS' if multi_ok else 'FAILED'}")
            print("MULTI_ROUTING_METHOD=" + multi.routing_method)
            print("MULTI_SELECTED_AGENTS=" + ",".join(name.value for name in multi.decision.agent_names))
        except Exception as exc:
            _report_live_error("LIVE_MULTI_ROUTE", exc, api_key, model)
            print("LIVE_GEMINI_ORCHESTRATION=SKIPPED")
            return

        if not multi_ok:
            print("LIVE_GEMINI_ORCHESTRATION=SKIPPED")
            return

        registry, agents = _make_registry()
        orchestrator = CityOrchestratorAgent(
            registry,
            FallbackQueryRouter(router, DeterministicQueryRouter()),
        )
        request = AgentRequest(query=MULTI_QUERY)
        response = await orchestrator.execute(request)
        summary: dict[str, Any] = response.metadata.get("execution_summary", {})
        results = summary.get("results", [])
        answers = {
            item.get("response", {}).get("answer")
            for item in results
            if item.get("success") and item.get("response")
        }
        selected = set(response.metadata.get("selected_agents", []))
        called_with_same_id = all(
            agents[name].request_ids == [str(request.request_id)] for name in EXPECTED_MULTI
        )
        orchestration_ok = (
            response.metadata.get("routing_method") == "gemini"
            and EXPECTED_MULTI <= selected
            and "public_services" not in selected
            and called_with_same_id
            and response.metadata.get("execution_status") == "complete"
            and {"FAKE_MOBILITY_RESULT", "FAKE_ENVIRONMENT_RESULT"} <= answers
        )
        if orchestration_ok:
            print("LIVE_GEMINI_ORCHESTRATION=PASS")
        elif response.metadata.get("routing_method") != "gemini":
            print("LIVE_GEMINI_ORCHESTRATION=FAILED_FALLBACK_USED")
        else:
            print("LIVE_GEMINI_ORCHESTRATION=FAILED")
        print("ORCHESTRATION_ROUTING_METHOD=" + str(response.metadata.get("routing_method")))
        print("ORCHESTRATION_SELECTED_AGENTS=" + ",".join(response.metadata.get("selected_agents", [])))
        print("FAKE_AGENTS_CALLED_WITH_SAME_REQUEST_ID=" + str(called_with_same_id))
        print("EXECUTION_STATUS=" + str(response.metadata.get("execution_status")))
        print("COLLECTED_FAKE_RESULTS=" + ",".join(sorted(answers)))
    finally:
        await router.aclose()


async def main() -> None:
    settings = get_settings()
    model = settings.gemini_model
    if not settings.gemini_api_key:
        print("LIVE_BASIC_GEMINI=SKIPPED")
        print(f"MODEL={model}")
        print("Reason: GEMINI_API_KEY is not configured.")
    else:
        await _run_live_stages(settings.gemini_api_key, model)
    await _verify_fallback()


if __name__ == "__main__":
    asyncio.run(main())
