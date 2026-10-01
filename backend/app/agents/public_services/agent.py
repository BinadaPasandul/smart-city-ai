"""Public Services specialist agent: NLP query understanding + IR retrieval.

Pipeline:

    AgentRequest
        -> validate query
        -> PublicServicesNLP.parse()      (category/location/district/... detection)
        -> PublicServicesIR.search()      (keyword-overlap retrieval over seed data)
        -> grounded AgentResponse          (answer built only from retrieved fields)

No LLM, external API, database, or security logic is used here. Every fact
in the answer is copied from `SearchMatch.record.fields`; nothing is
inferred or invented. See `PublicServicesNLP`/`PublicServicesIR` for the
component-level contracts this agent composes.
"""

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.contracts import AgentError, AgentErrorCode, AgentRequest, AgentResponse, AgentSource
from app.ir.public_services_ir import PublicServiceCategory, PublicServiceRecord, PublicServicesIR, SearchMatch
from app.nlp.public_services_nlp import PublicServicesNLP, PublicServicesQuery

logger = logging.getLogger(__name__)

# Which raw seed-record fields are surfaced in the generated answer, per
# category, in display order. Fields absent from a given record (or
# blank/empty) are silently skipped -- nothing is filled in or guessed.
_DISPLAY_FIELDS: dict[PublicServiceCategory, tuple[str, ...]] = {
    PublicServiceCategory.HOSPITALS: (
        "type", "location", "district", "address", "services",
        "emergency_available", "contact", "operating_hours",
    ),
    PublicServiceCategory.POLICE_STATIONS: (
        "location", "district", "address", "services", "contact", "operating_hours",
    ),
    PublicServiceCategory.FIRE_STATIONS: (
        "location", "district", "address", "emergency_services", "contact", "operating_hours",
    ),
    PublicServiceCategory.GOVERNMENT_SERVICES: (
        "department", "location", "district", "services", "contact", "operating_hours",
    ),
    PublicServiceCategory.EMERGENCY_INFORMATION: (
        "emergency_type", "description", "contact",
    ),
    PublicServiceCategory.CITIZEN_COMPLAINTS: (
        "responsible_department", "description", "submission_method",
    ),
}


class PublicServicesAgent(BaseAgent):
    """Handle citizen queries about public services using seed-data retrieval.

    Covers hospitals, police stations, fire stations, government services,
    emergency information, and citizen complaints. Answers are grounded
    strictly in `backend/data/seed/public_services_data.json` (synthetic
    demo data) via `PublicServicesNLP` + `PublicServicesIR`; there is no
    LLM, live API, or database behind this agent yet.
    """

    _MAX_DISPLAYED_RESULTS = 3

    def __init__(
        self,
        nlp: PublicServicesNLP | None = None,
        ir: PublicServicesIR | None = None,
    ) -> None:
        super().__init__(
            "public_services",
            description=(
                "Handles citizen queries about public services, including hospitals, "
                "police stations, fire stations, government services, emergency "
                "information, and citizen complaints, grounded in synthetic seed data."
            ),
            capabilities=(
                "rule_based_nlp",
                "keyword_search_retrieval",
                "seed_data_grounded_answers",
            ),
        )
        self._nlp = nlp or PublicServicesNLP()
        self._ir = ir or PublicServicesIR()

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Parse the query, retrieve matching records, and build a grounded answer."""
        if not request.query.strip():
            return self._failure(request, AgentErrorCode.INVALID_REQUEST, "The request query must not be empty.")

        try:
            parsed = self._nlp.parse(request.query)
        except Exception:
            logger.exception("Public services NLP parsing failed request_id=%s", request.request_id)
            return self._failure(
                request,
                AgentErrorCode.AGENT_EXECUTION_FAILED,
                "The request could not be interpreted.",
            )

        if parsed.category is None:
            logger.info(
                "Public services agent could not determine a category request_id=%s", request.request_id
            )
            return self._failure(
                request,
                AgentErrorCode.UNSUPPORTED_REQUEST,
                "Could not confidently determine which public service category this "
                "request relates to.",
                metadata=self._build_metadata(parsed, result_count=0),
            )

        try:
            results = self._ir.search(parsed.search_query, category=parsed.category)
        except Exception:
            logger.exception(
                "Public services IR search failed request_id=%s category=%s",
                request.request_id,
                parsed.category.value,
            )
            return self._failure(
                request,
                AgentErrorCode.AGENT_EXECUTION_FAILED,
                "The public services search could not be completed.",
                metadata=self._build_metadata(parsed, result_count=0),
            )

        if not results:
            logger.info(
                "Public services agent found no matching records request_id=%s category=%s",
                request.request_id,
                parsed.category.value,
            )
            category_label = parsed.category.value.replace("_", " ")
            return self._failure(
                request,
                AgentErrorCode.UNSUPPORTED_REQUEST,
                f"No {category_label} information matched this request.",
                metadata=self._build_metadata(parsed, result_count=0),
            )

        top_results = results[: self._MAX_DISPLAYED_RESULTS]
        logger.info(
            "Public services agent request_id=%s category=%s result_count=%d displayed=%d",
            request.request_id,
            parsed.category.value,
            len(results),
            len(top_results),
        )
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=True,
            answer=self._build_answer(parsed, top_results, total_count=len(results)),
            sources=self._build_sources(top_results),
            metadata=self._build_metadata(parsed, result_count=len(results)),
        )

    def _build_answer(
        self, parsed: PublicServicesQuery, results: list[SearchMatch], *, total_count: int
    ) -> str:
        """Compose a concise, numbered answer using only retrieved record fields."""
        category_label = parsed.category.value.replace("_", " ") if parsed.category else "public services"
        header = f"Found {total_count} matching {category_label} record(s)."
        if total_count > len(results):
            header += f" Showing the top {len(results)}."
        entries = [
            f"{index}. {self._format_record(match.record)}"
            for index, match in enumerate(results, start=1)
        ]
        return "\n".join([header, *entries])

    @staticmethod
    def _format_record(record: PublicServiceRecord) -> str:
        """Render one record as its title plus only the fields it actually has."""
        display_fields = _DISPLAY_FIELDS.get(record.category, ())
        detail_lines = []
        for field_name in display_fields:
            formatted = PublicServicesAgent._format_field_value(record.fields.get(field_name))
            if formatted is None:
                continue
            label = field_name.replace("_", " ").capitalize()
            detail_lines.append(f"   - {label}: {formatted}")
        title = record.title or f"Record {record.record_id}"
        return "\n".join([title, *detail_lines])

    @staticmethod
    def _format_field_value(value: Any) -> str | None:
        """Format one raw seed-data field for display, or None to omit it."""
        if value is None:
            return None
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, list):
            items = [str(item) for item in value if str(item).strip()]
            return ", ".join(items) if items else None
        if isinstance(value, str):
            return value if value.strip() else None
        return str(value)

    @staticmethod
    def _build_sources(results: list[SearchMatch]) -> list[AgentSource]:
        """Build grounded AgentSource entries from the displayed search matches."""
        return [
            AgentSource(
                name=match.source.source_file,
                source_type=match.source.source_type,
                metadata={
                    "category": match.record.category.value,
                    "record_id": match.record.record_id,
                    "relevance_score": match.score,
                    "matched_terms": match.matched_terms,
                    "description": match.source.description,
                },
            )
            for match in results
        ]

    @staticmethod
    def _build_metadata(parsed: PublicServicesQuery, *, result_count: int) -> dict[str, Any]:
        """Build response metadata reflecting the actual parsed query and result count."""
        return {
            "implementation_status": "seed_data",
            "category": parsed.category.value if parsed.category else None,
            "location": parsed.location,
            "district": parsed.district,
            "is_emergency": parsed.is_emergency,
            "complaint_type": parsed.complaint_type,
            "result_count": result_count,
            "retrieval_method": "keyword_overlap",
            "data_source": "synthetic_demo",
        }

    def _failure(
        self,
        request: AgentRequest,
        code: AgentErrorCode,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            success=False,
            metadata=metadata or {},
            error=AgentError(code=code, message=message),
        )
