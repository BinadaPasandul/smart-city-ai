"""Optional live HTTP + Gemini verification using verification-only fake agents."""

import re
import asyncio
from typing import Any

from httpx import ASGITransport, AsyncClient

from app.agents.base import BaseAgent
from app.agents.contracts import AgentRequest, AgentResponse, AgentSource
from app.agents.orchestrator.agent import CityOrchestratorAgent
from app.agents.orchestrator.gemini_router import FallbackQueryRouter, GeminiQueryRouter
from app.agents.orchestrator.router import DeterministicQueryRouter
from app.agents.orchestrator.synthesizer import GeminiResultSynthesizer
from app.agents.registry import AgentRegistry
from app.api.dependencies import get_orchestrator
from app.core.config import get_settings
from app.main import app

QUERY = "Considering traffic and air quality, is it a good time to cycle in Colombo today?"


class FakeSpecialist(BaseAgent):
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
            sources=[AgentSource(name=f"Fake {self.name} evidence", source_type="test")],
        )


def _safe_message(exc: Exception, api_key: str) -> str:
    message = str(getattr(exc, "message", None) or str(exc) or "Request failed")
    message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"(?i)(api[_ -]?key|authorization|bearer)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", message)
    return message[:500]


async def main() -> None:
    settings = get_settings()
    model = settings.gemini_model
    if not settings.gemini_api_key:
        print("LIVE_CHAT_API=SKIPPED")
        print(f"MODEL={model}")
        print("Reason: GEMINI_API_KEY is not configured.")
        return

    mobility = FakeSpecialist("mobility", "Traffic is heavy near Colombo Fort.")
    environment = FakeSpecialist("environment", "Air quality is moderate.")
    registry = AgentRegistry()
    registry.register(mobility)
    registry.register(environment)
    gemini_router = GeminiQueryRouter(settings.gemini_api_key, model=model)
    gemini_synthesizer = GeminiResultSynthesizer(settings.gemini_api_key, model=model)
    orchestrator = CityOrchestratorAgent(
        registry,
        FallbackQueryRouter(gemini_router, DeterministicQueryRouter()),
        synthesizer=gemini_synthesizer,
    )
    prior_override = app.dependency_overrides.get(get_orchestrator)
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post("/api/v1/chat", json={"message": QUERY})
        body: dict[str, Any] = response.json()
        agents = set(body.get("metadata", {}).get("selected_agents", []))
        route_method = body.get("metadata", {}).get("routing_method")
        synthesis_method = body.get("metadata", {}).get("synthesis_method")
        request_id = body.get("request_id")
        same_id = (
            request_id is not None
            and mobility.request_ids == [request_id]
            and environment.request_ids == [request_id]
        )
        answer = body.get("answer", "")
        grounding_ok = not re.search(r"\d", answer)
        passed = (
            response.status_code == 200
            and route_method == "gemini"
            and synthesis_method == "gemini"
            and {"mobility", "environment"} <= agents
            and body.get("metadata", {}).get("execution_status") == "complete"
            and same_id
            and len(body.get("sources", [])) == 2
            and "heavy" in answer.lower()
            and "moderate" in answer.lower()
            and grounding_ok
        )
        if passed:
            print("LIVE_CHAT_API=PASS")
        elif route_method != "gemini" or synthesis_method != "gemini":
            print("LIVE_CHAT_API=FAILED_FALLBACK_USED")
        else:
            print("LIVE_CHAT_API=FAILED")
        print(f"MODEL={model}")
        print(f"HTTP_STATUS={response.status_code}")
        print(f"ROUTING_METHOD={route_method}")
        print(f"SYNTHESIS_METHOD={synthesis_method}")
        print(f"SELECTED_AGENTS={','.join(sorted(agents))}")
        print(f"EXECUTION_STATUS={body.get('metadata', {}).get('execution_status')}")
        print(f"REQUEST_ID_PRESERVED={same_id}")
        print(f"SOURCES_PRESERVED={len(body.get('sources', [])) == 2}")
        print(f"GROUNDING_SANITY={'PASS' if grounding_ok else 'FAILED'}")
        print("ANSWER=" + answer)
    except Exception as exc:
        print("LIVE_CHAT_API=FAILED")
        print(f"MODEL={model}")
        print(f"ERROR_TYPE={type(exc).__name__}")
        print(f"ERROR_CODE={getattr(exc, 'code', 'unavailable')}")
        print("ERROR_MESSAGE=" + _safe_message(exc, settings.gemini_api_key))
    finally:
        if prior_override is None:
            app.dependency_overrides.pop(get_orchestrator, None)
        else:
            app.dependency_overrides[get_orchestrator] = prior_override
        await gemini_router.aclose()
        await gemini_synthesizer.aclose()


if __name__ == "__main__":
    asyncio.run(main())
