"""City Orchestrator with bounded concurrent in-process agent execution."""

import asyncio
import logging
from urllib.parse import urlsplit, urlunsplit

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse, AgentSource
from app.agents.orchestrator.execution import (
    ExecutionStatus,
    OrchestrationExecutionSummary,
    SpecialistExecutionResult,
)
from app.agents.orchestrator.context_adapter import (
    AmbiguousSpecialistContext,
    SpecialistContextAdapter,
)
from app.agents.orchestrator.router import (
    QueryRouter,
    RoutingResult,
    SpecialistAgentName,
)
from app.agents.orchestrator.synthesizer import (
    FallbackResultSynthesizer,
    GeminiResultSynthesizer,
    ResultSynthesizer,
)
from app.agents.orchestrator.web_search import WebSearchService, WebSearchStatus
from app.agents.registry import AgentNotFoundError, AgentRegistry
from app.core.config import settings

logger = logging.getLogger(__name__)


class CityOrchestratorAgent(BaseAgent):
    """Plan a request, execute selected specialists concurrently, and collect results."""

    def __init__(
        self,
        registry: AgentRegistry,
        router: QueryRouter | None = None,
        *,
        execution_timeout_seconds: float | None = None,
        synthesizer: ResultSynthesizer | None = None,
        web_search: WebSearchService | None = None,
    ) -> None:
        super().__init__(
            "orchestrator",
            description="Plans citizen queries and coordinates registered specialist agents.",
            capabilities=("gemini_routing", "deterministic_fallback", "parallel_execution"),
        )
        self._registry = registry
        self._context_adapter = SpecialistContextAdapter()
        self._web_search = web_search or WebSearchService(None, enabled=False)
        self._synthesizer = FallbackResultSynthesizer(
            synthesizer or GeminiResultSynthesizer(
                settings.gemini_api_key, model=settings.gemini_model
            )
        )
        if router is None:
            from app.nlp.pipeline import RequestUnderstandingPipeline

            self._router = RequestUnderstandingPipeline(settings=settings)
        else:
            self._router = router
        self._execution_timeout_seconds = (
            execution_timeout_seconds
            if execution_timeout_seconds is not None
            else settings.agent_execution_timeout_seconds
        )
        if self._execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be greater than zero")

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Route a request to up to three specialists and aggregate their outcomes."""
        if not request.query.strip():
            return self._failure(request, AgentErrorCode.INVALID_REQUEST, "The request query must not be empty.")

        route_request = getattr(self._router, "route_request", None)
        if callable(route_request):
            routing = await route_request(request)
        else:
            routing = await self._router.route(
                request.query, request_id=str(request.request_id)
            )
        decision = routing.decision
        understanding_method = routing.understanding_method or routing.routing_method
        routing_metadata = {
            "routing_method": routing.routing_method,
            "understanding_method": understanding_method,
            "nlp_confidence": routing.local_nlp_confidence,
            "gemini_understanding_fallback_used": routing.gemini_understanding_fallback_used,
            "gemini_attempt_count": routing.gemini_attempt_count,
            "gemini_first_attempt_status": routing.gemini_first_attempt_status,
            "gemini_retry_triggered": routing.gemini_retry_triggered,
            "gemini_retry_reason": routing.gemini_retry_reason,
            "gemini_final_status": routing.gemini_final_status,
        }

        if decision.needs_clarification:
            return self._failure(
                request,
                AgentErrorCode.NEEDS_CLARIFICATION,
                "Could you provide a little more detail so I can direct your request?",
                metadata=routing_metadata,
            )
        if not decision.agent_names:
            return self._failure(
                request,
                AgentErrorCode.UNSUPPORTED_REQUEST,
                "The request does not match a supported city service category.",
                metadata=routing_metadata,
            )

        selected = decision.agent_names
        logger.info(
            "Starting orchestration request_id=%s routing_method=%s selected_agents=%s",
            request.request_id,
            routing.routing_method,
            [agent.value for agent in selected],
        )
        specialist_request = self._enrich_request(request, routing)
        try:
            adapted_requests = {
                name: self._context_adapter.adapt(name.value, specialist_request)
                for name in selected
            }
        except AmbiguousSpecialistContext:
            return self._failure(
                request,
                AgentErrorCode.NEEDS_CLARIFICATION,
                "Please provide one location for the environmental request.",
                metadata=routing_metadata | {
                    "selected_agents": [agent.value for agent in selected],
                    "execution_status": ExecutionStatus.FAILED.value,
                },
            )
        results = await asyncio.gather(
            *(self._execute_specialist(name, adapted_requests[name]) for name in selected)
        )
        summary = self._summarize(selected, results)
        search_outcome = await self._web_search.run(
            request.query, summary, request_id=str(request.request_id)
        )
        web_evidence = search_outcome.evidence
        metadata = {
            **routing_metadata,
            "selected_agents": [agent.value for agent in selected],
            "execution_status": summary.status.value,
            "successful_agents": [agent.value for agent in summary.successful_agents],
            "failed_agents": [agent.value for agent in summary.failed_agents],
            "execution_summary": summary.model_dump(mode="json"),
            "web_search_used": search_outcome.status is WebSearchStatus.SUCCESS,
            "web_search_status": search_outcome.status.value,
            "web_search_provider": search_outcome.provider,
            "web_search_result_count": len(web_evidence),
            "answer_basis": (
                "specialist_plus_web" if web_evidence and summary.successful_agents
                else "web_fallback" if web_evidence
                else "specialist" if summary.successful_agents
                else None
            ),
        }
        logger.info(
            "Finished orchestration request_id=%s status=%s succeeded=%d failed=%d",
            request.request_id,
            summary.status.value,
            len(summary.successful_agents),
            len(summary.failed_agents),
        )

        # Preserve the single-specialist response without an extra synthesis request.
        if len(results) == 1 and results[0].success and results[0].response is not None:
            response = results[0].response
            return response.model_copy(
                update={
                    "metadata": {
                        **response.metadata,
                        **metadata,
                        "synthesis_method": "single_specialist_passthrough",
                    }
                }
            )

        errors = [result.error for result in results if result.error is not None]
        overall_error = None
        if summary.status is ExecutionStatus.FAILED:
            if len(results) == 1 and results[0].error is not None:
                # Exactly one specialist was selected and it failed: its own
                # error (e.g. "the city name is ambiguous") is specific and
                # useful, unlike the generic aggregation message below, which
                # exists for combining >=2 failures and isn't meaningful for
                # exactly one. Only used when web search didn't rescue the
                # request (see `error=None if web_evidence else overall_error`
                # below) -- multi-agent aggregation is unchanged.
                overall_error = results[0].error
            else:
                error_codes = {error.code for error in errors}
                if error_codes == {AgentErrorCode.AGENT_NOT_FOUND}:
                    error_code = AgentErrorCode.AGENT_NOT_FOUND
                elif error_codes == {AgentErrorCode.TIMEOUT}:
                    error_code = AgentErrorCode.TIMEOUT
                else:
                    error_code = AgentErrorCode.AGENT_EXECUTION_FAILED
                overall_error = AgentError(
                    code=error_code,
                    message="No selected specialist completed the request.",
                )

        specialist_sources = [
            source
            for result in results
            if result.success and result.response is not None
            for source in result.response.sources
        ]
        merged_sources = self._merge_sources(
            specialist_sources, [item.to_agent_source() for item in web_evidence]
        )
        answer = "No specialist agent completed the request."
        if summary.status is not ExecutionStatus.FAILED or web_evidence:
            if web_evidence:
                synthesis_result, synthesis_method = await self._synthesizer.synthesize(
                    request.query, summary, web_evidence
                )
            else:
                synthesis_result, synthesis_method = await self._synthesizer.synthesize(
                    request.query, summary
                )
            answer = synthesis_result.answer
            if summary.status is ExecutionStatus.PARTIAL_SUCCESS:
                unavailable = [name.value.replace("_", " ") for name in summary.failed_agents]
                answer = (
                    f"{answer.rstrip()}\nUnavailable information: "
                    f"{', '.join(unavailable)} could not be retrieved."
                )
            metadata["synthesis_method"] = synthesis_method
            metadata["synthesis_used_agents"] = [name.value for name in synthesis_result.used_agents]
            metadata["synthesis_limitations"] = synthesis_result.limitations
            metadata["gemini_synthesis_attempt_count"] = synthesis_result.gemini_attempt_count
            metadata["gemini_synthesis_first_attempt_status"] = synthesis_result.gemini_first_attempt_status
            metadata["gemini_synthesis_retry_triggered"] = synthesis_result.gemini_retry_triggered
            metadata["gemini_synthesis_retry_reason"] = synthesis_result.gemini_retry_reason
            metadata["gemini_synthesis_final_status"] = synthesis_result.gemini_final_status
            logger.info(
                "Synthesis finished request_id=%s method=%s successful_results=%d",
                request.request_id,
                synthesis_method,
                len(summary.successful_agents),
            )

        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=summary.status is not ExecutionStatus.FAILED or bool(web_evidence),
            answer=answer,
            sources=merged_sources,
            metadata=metadata,
            error=None if web_evidence else overall_error,
        )

    async def _execute_specialist(
        self, agent_name: SpecialistAgentName, request: AgentRequest
    ) -> SpecialistExecutionResult:
        """Resolve and execute one specialist with an independent timeout."""
        try:
            agent = self._registry.get(agent_name.value)
        except AgentNotFoundError:
            return SpecialistExecutionResult(
                agent_name=agent_name,
                success=False,
                error=AgentError(
                    code=AgentErrorCode.AGENT_NOT_FOUND,
                    message=f"The selected agent '{agent_name.value}' is not registered.",
                ),
            )

        try:
            async with asyncio.timeout(self._execution_timeout_seconds):
                response = await agent.execute(request)
        except TimeoutError:
            logger.warning(
                "Specialist timed out request_id=%s agent=%s timeout_seconds=%s",
                request.request_id,
                agent_name.value,
                self._execution_timeout_seconds,
            )
            return SpecialistExecutionResult(
                agent_name=agent_name,
                success=False,
                error=AgentError(
                    code=AgentErrorCode.TIMEOUT,
                    message="The specialist exceeded its execution time limit.",
                ),
                timed_out=True,
            )
        except Exception as exc:
            logger.error(
                "Specialist failed request_id=%s agent=%s exception_type=%s",
                request.request_id,
                agent_name.value,
                type(exc).__name__,
            )
            return SpecialistExecutionResult(
                agent_name=agent_name,
                success=False,
                error=AgentError(
                    code=AgentErrorCode.AGENT_EXECUTION_FAILED,
                    message="The specialist could not complete the request.",
                ),
            )

        if not response.success or not response.answer.strip():
            return SpecialistExecutionResult(
                agent_name=agent_name,
                success=False,
                response=response,
                error=response.error
                or AgentError(
                    code=AgentErrorCode.AGENT_EXECUTION_FAILED,
                    message="The specialist reported that it could not complete the request.",
                ),
            )
        return SpecialistExecutionResult(agent_name=agent_name, success=True, response=response)

    @staticmethod
    def _merge_sources(
        specialist_sources: list[AgentSource], web_sources: list[AgentSource]
    ) -> list[AgentSource]:
        """Keep every specialist source and omit duplicate supplemental URLs."""
        merged = list(specialist_sources)
        seen: set[str] = set()

        def url_key(source: AgentSource) -> str | None:
            if not source.url:
                return None
            try:
                parsed = urlsplit(source.url)
                return urlunsplit((
                    parsed.scheme.lower(), parsed.netloc.lower(),
                    parsed.path or "/", parsed.query, "",
                ))
            except ValueError:
                return source.url

        seen.update(key for source in specialist_sources if (key := url_key(source)))
        for source in web_sources:
            key = url_key(source)
            if key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(source)
        return merged

    @staticmethod
    def _summarize(
        selected: list[SpecialistAgentName],
        results: list[SpecialistExecutionResult],
    ) -> OrchestrationExecutionSummary:
        successful = [result.agent_name for result in results if result.success]
        failed = [result.agent_name for result in results if not result.success]
        if not failed:
            status = ExecutionStatus.COMPLETE
        elif successful:
            status = ExecutionStatus.PARTIAL_SUCCESS
        else:
            status = ExecutionStatus.FAILED
        return OrchestrationExecutionSummary(
            requested_agents=selected,
            successful_agents=successful,
            failed_agents=failed,
            status=status,
            results=results,
        )

    def _failure(
        self,
        request: AgentRequest,
        code: AgentErrorCode,
        message: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=False,
            metadata=metadata or {},
            error=AgentError(code=code, message=message),
        )

    @staticmethod
    def _enrich_request(request: AgentRequest, routing: RoutingResult) -> AgentRequest:
        """Copy the request and attach server-derived NLP facts before fan-out."""
        context = {
            key: value for key, value in request.context.items()
            if key != "nlp"
        }
        has_understanding = (
            routing.understanding_method is not None
            or routing.local_nlp_confidence is not None
            or bool(routing.locations)
            or bool(routing.temporal_expressions)
        )
        if has_understanding:
            context["nlp"] = {
                "locations": list(routing.locations),
                "temporal_expressions": list(routing.temporal_expressions),
                "missing_information": [item.value for item in routing.missing_information],
                "understanding_method": routing.understanding_method or routing.routing_method,
                "confidence": routing.local_nlp_confidence,
            }
        return request.model_copy(update={"context": context}, deep=True)
