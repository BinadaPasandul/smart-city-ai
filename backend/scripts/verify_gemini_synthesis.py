"""Optional live Gemini synthesis smoke test using only fake specialist evidence."""

import asyncio
import re
from typing import Any

from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse, AgentSource
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.router import (
    DeterministicQueryRouter,
    RoutingDecision,
    RoutingResult,
)
from app.agents.orchestrator.synthesizer import GeminiResultSynthesizer
from app.agents.registry import AgentRegistry
from app.core.config import get_settings


class FixedMultiRouter:
    async def route(self, query: str, *, request_id: str | None = None) -> RoutingResult:
        return RoutingResult(
            decision=RoutingDecision(
                agent_names=["mobility", "environment"],
                confidence=1,
                reason="Live synthesis verification uses fixed test routing.",
                needs_clarification=False,
            ),
            routing_method="gemini",
        )


class FakeSpecialist(BaseAgent):
    def __init__(self, name: str, answer: str) -> None:
        super().__init__(name)
        self.answer = answer

    async def execute(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=self.answer,
            sources=[AgentSource(name=f"Fake {self.name} evidence", source_type="test")],
        )


class ObservedSynthesizer:
    """Capture a live SDK failure for safe verification diagnostics."""

    def __init__(self, delegate: GeminiResultSynthesizer) -> None:
        self.delegate = delegate
        self.errors: list[Exception] = []

    async def synthesize(self, query: str, summary: Any):
        try:
            return await self.delegate.synthesize(query, summary)
        except Exception as exc:
            self.errors.append(exc)
            raise


async def main() -> None:
    settings = get_settings()
    model = settings.gemini_model
    if not settings.gemini_api_key:
        print("LIVE_GEMINI_SYNTHESIS=SKIPPED")
        print(f"MODEL={model}")
        print("Reason: GEMINI_API_KEY is not configured.")
        return

    registry = AgentRegistry()
    registry.register(FakeSpecialist("mobility", "Traffic is heavy near Colombo Fort."))
    registry.register(FakeSpecialist("environment", "Air quality is moderate."))
    gemini_synthesizer = GeminiResultSynthesizer(settings.gemini_api_key, model=model)
    synthesizer = ObservedSynthesizer(gemini_synthesizer)
    try:
        response = await CityOrchestratorAgent(
            registry,
            FixedMultiRouter(),
            synthesizer=synthesizer,
        ).execute(
            AgentRequest(
                query="Considering traffic and air quality, is it a good time to cycle near Colombo Fort?"
            )
        )
        passed = (
            response.metadata.get("synthesis_method") == "gemini"
            and response.metadata.get("execution_status") == "complete"
            and "heavy" in response.answer.lower()
            and "moderate" in response.answer.lower()
            and not re.search(r"\d", response.answer)
            and len(response.sources) == 2
        )
        print(f"LIVE_GEMINI_SYNTHESIS={'PASS' if passed else 'FAILED'}")
        print(f"MODEL={model}")
        print(f"SYNTHESIS_METHOD={response.metadata.get('synthesis_method')}")
        print(f"EXECUTION_STATUS={response.metadata.get('execution_status')}")
        print(f"SOURCES_PRESERVED={len(response.sources) == 2}")
        grounding_ok = not re.search(r"\d", response.answer)
        print(f"GROUNDING_SANITY={'PASS' if grounding_ok else 'FAILED'}")
        print("ANSWER=" + response.answer)
        if not passed and synthesizer.errors:
            exc = synthesizer.errors[-1]
            message = str(getattr(exc, "message", None) or str(exc) or "Synthesis request failed")
            message = message.replace(settings.gemini_api_key, "[REDACTED]")
            message = re.sub(r"(?i)(api[_ -]?key|authorization|bearer)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", message)
            print(f"ERROR_TYPE={type(exc).__name__}")
            print(f"ERROR_CODE={getattr(exc, 'code', 'unavailable')}")
            print("ERROR_MESSAGE=" + message[:500])
    except Exception as exc:
        message = str(getattr(exc, "message", None) or "Synthesis request failed")
        message = message.replace(settings.gemini_api_key, "[REDACTED]")
        message = re.sub(r"(?i)(api[_ -]?key|authorization|bearer)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", message)
        print("LIVE_GEMINI_SYNTHESIS=FAILED")
        print(f"MODEL={model}")
        print(f"ERROR_TYPE={type(exc).__name__}")
        print(f"ERROR_CODE={getattr(exc, 'code', 'unavailable')}")
        print("ERROR_MESSAGE=" + message[:500])
    finally:
        await gemini_synthesizer.aclose()


if __name__ == "__main__":
    asyncio.run(main())
