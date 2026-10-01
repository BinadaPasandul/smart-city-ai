"""One-shot, bounded web fallback after eligible specialist failures."""

import asyncio
from enum import Enum
import logging

from pydantic import BaseModel, Field

from app.agents.contracts import AgentErrorCode
from app.agents.orchestrator.execution import ExecutionStatus, OrchestrationExecutionSummary
from app.ir.web_search import WebEvidence, WebSearchProvider, WebSearchProviderError, WebSearchResult

logger = logging.getLogger(__name__)


class WebSearchStatus(str, Enum):
    NOT_NEEDED = "not_needed"
    DISABLED = "disabled"
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    FAILED = "failed"
    TIMEOUT = "timeout"


class WebSearchOutcome(BaseModel):
    """Safe search state for orchestration and API metadata."""

    status: WebSearchStatus
    provider: str | None = None
    results: list[WebSearchResult] = Field(default_factory=list)
    failure_category: str | None = None

    @property
    def evidence(self) -> list[WebEvidence]:
        return [WebEvidence.from_result(result) for result in self.results]


class WebSearchFallbackPolicy:
    """Search only when routed specialists fail with operational errors."""

    ELIGIBLE_ERRORS = {
        AgentErrorCode.AGENT_NOT_FOUND,
        AgentErrorCode.AGENT_EXECUTION_FAILED,
        AgentErrorCode.TIMEOUT,
    }

    def __init__(self, *, on_partial_failure: bool = True) -> None:
        self.on_partial_failure = on_partial_failure

    def should_search(self, summary: OrchestrationExecutionSummary) -> bool:
        if summary.status is ExecutionStatus.COMPLETE:
            return False
        if summary.status is ExecutionStatus.PARTIAL_SUCCESS and not self.on_partial_failure:
            return False
        failures = [result for result in summary.results if not result.success]
        return bool(failures) and all(
            result.error is not None and result.error.code in self.ELIGIBLE_ERRORS
            for result in failures
        )


class WebSearchService:
    """Apply policy, query limits and timeout around one provider call."""

    def __init__(
        self,
        provider: WebSearchProvider | None,
        *,
        enabled: bool,
        on_partial_failure: bool = True,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        query_max_length: int = 500,
    ) -> None:
        if timeout_seconds <= 0 or not 1 <= max_results <= 10 or query_max_length <= 0:
            raise ValueError("invalid web search limits")
        if enabled and provider is None:
            raise ValueError("enabled web search needs a provider")
        self._provider = provider
        self._enabled = enabled
        self._policy = WebSearchFallbackPolicy(on_partial_failure=on_partial_failure)
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._query_max_length = query_max_length

    async def run(
        self, query: str, summary: OrchestrationExecutionSummary, *, request_id: str
    ) -> WebSearchOutcome:
        if not self._policy.should_search(summary):
            return WebSearchOutcome(status=WebSearchStatus.NOT_NEEDED)
        if not self._enabled or self._provider is None:
            return WebSearchOutcome(status=WebSearchStatus.DISABLED)
        provider_name = self._provider.provider_name
        try:
            async with asyncio.timeout(self._timeout_seconds):
                results = await self._provider.search(query.strip()[: self._query_max_length], self._max_results)
        except TimeoutError:
            outcome = WebSearchOutcome(status=WebSearchStatus.TIMEOUT, provider=provider_name)
        except WebSearchProviderError as exc:
            status = WebSearchStatus.TIMEOUT if exc.category == "timeout" else WebSearchStatus.FAILED
            outcome = WebSearchOutcome(
                status=status, provider=provider_name, failure_category=exc.category
            )
        except Exception as exc:
            logger.warning(
                "Web search failed request_id=%s provider=%s exception_type=%s",
                request_id, provider_name, type(exc).__name__,
            )
            outcome = WebSearchOutcome(
                status=WebSearchStatus.FAILED, provider=provider_name, failure_category="unexpected"
            )
        else:
            valid: list[WebSearchResult] = []
            seen_urls: set[str] = set()
            if not isinstance(results, list):
                outcome = WebSearchOutcome(
                    status=WebSearchStatus.FAILED, provider=provider_name,
                    failure_category="invalid_schema",
                )
                logger.info(
                    "Web search request_id=%s provider=%s status=%s result_count=0",
                    request_id, provider_name, outcome.status.value,
                )
                return outcome
            for item in results:
                if len(valid) >= self._max_results:
                    break
                try:
                    result = WebSearchResult.model_validate(
                        item.model_dump() if isinstance(item, WebSearchResult) else item
                    )
                except (ValueError, TypeError):
                    continue
                if result.provider != provider_name:
                    continue
                if result.url not in seen_urls:
                    valid.append(result)
                    seen_urls.add(result.url)
            outcome = WebSearchOutcome(
                status=WebSearchStatus.SUCCESS if valid else WebSearchStatus.NO_RESULTS,
                provider=provider_name,
                results=valid,
            )
        logger.info(
            "Web search request_id=%s provider=%s status=%s result_count=%d",
            request_id, provider_name, outcome.status.value, len(outcome.results),
        )
        return outcome

    async def aclose(self) -> None:
        if self._provider is not None:
            close = getattr(self._provider, "aclose", None)
            if close is not None:
                await close()
