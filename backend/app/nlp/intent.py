"""Local keyword intent classification sharing deterministic router rules."""

from pydantic import BaseModel, ConfigDict, Field

from app.agents.orchestrator.router import DeterministicQueryRouter, SpecialistAgentName


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_agents: list[SpecialistAgentName] = Field(default_factory=list, max_length=3)
    matched_terms: list[str] = Field(default_factory=list)


class LocalIntentClassifier:
    """Expose the deterministic router's common keyword matches as NLP signals."""

    def classify(self, normalized_text: str) -> IntentClassification:
        padded = f" {normalized_text} "
        matched_by_category: dict[str, list[str]] = {
            category: [
                term for term in terms
                if f" {DeterministicQueryRouter._normalize(term)} " in padded
            ]
            for category, terms in DeterministicQueryRouter.KEYWORDS.items()
        }
        agents = [
            SpecialistAgentName(category)
            for category, terms in matched_by_category.items()
            if terms
        ]
        terms = list(dict.fromkeys(term for matches in matched_by_category.values() for term in matches))
        return IntentClassification(candidate_agents=agents, matched_terms=terms)
