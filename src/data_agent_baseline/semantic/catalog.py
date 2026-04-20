from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EntitySpec:
    name: str
    aliases: list[str]
    description: str
    sources: list[str]
    primary_keys: list[str]
    candidate_keys: list[str]
    confidence: str
    provenance: str


@dataclass(frozen=True, slots=True)
class RelationKeyPair:
    left_field: str
    right_field: str


@dataclass(frozen=True, slots=True)
class RelationSpec:
    left_entity: str
    right_entity: str
    join_keys: list[RelationKeyPair]
    cardinality: str
    description: str
    confidence: str
    provenance: str


@dataclass(frozen=True, slots=True)
class DimensionSpec:
    name: str
    entity: str
    field_ref: str
    data_type: str
    semantic_type: str
    time_grain: str | None
    aliases: list[str]
    confidence: str
    provenance: str
    sample_values: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MeasureSpec:
    name: str
    entity: str
    field_ref: str
    default_agg: str
    unit: str | None
    value_type: str
    constraints: list[str]
    confidence: str
    provenance: str


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    description: str
    formula: str
    base_measures: list[str]
    required_dimensions: list[str]
    filters: dict[str, str]
    grain: str | None
    confidence: str
    provenance: str
    evidence_refs: list[str]


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    id: str
    claim: str
    source_type: str
    source_file: str
    location_hint: str
    snippet: str
    confidence: str
    provenance: str


@dataclass(frozen=True, slots=True)
class KnowledgeContract:
    sections: dict[str, str] = field(default_factory=dict)
    entity_field_rules: list[dict[str, Any]] = field(default_factory=list)
    metric_rules: list[dict[str, Any]] = field(default_factory=list)
    constraint_rules: list[dict[str, Any]] = field(default_factory=list)
    example_rules: list[dict[str, Any]] = field(default_factory=list)
    ambiguity_rules: list[dict[str, Any]] = field(default_factory=list)
    output_constraints: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PlannerQuestionSlot:
    slot_type: str
    phrase: str


@dataclass(frozen=True, slots=True)
class PlannerBindingCandidate:
    field_ref: str
    entity: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class PlannerChosenBinding:
    slot_type: str
    phrase: str
    field_ref: str
    entity: str
    resolved_value: str | int | float | None = None


@dataclass(frozen=True, slots=True)
class PlannerJoinCandidate:
    left_entity: str
    right_entity: str
    left_field: str
    right_field: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class PlannerConflict:
    phrase: str
    candidates: list[str]
    resolution: str


@dataclass(frozen=True, slots=True)
class SemanticCatalog:
    entities: list[EntitySpec] = field(default_factory=list)
    relations: list[RelationSpec] = field(default_factory=list)
    dimensions: list[DimensionSpec] = field(default_factory=list)
    measures: list[MeasureSpec] = field(default_factory=list)
    metrics: list[MetricSpec] = field(default_factory=list)
    evidence: list[EvidenceSpec] = field(default_factory=list)
    knowledge_contract: KnowledgeContract = field(default_factory=KnowledgeContract)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(
        self, *, max_items_per_section: int = 10, include_evidence: bool = False
    ) -> dict[str, Any]:
        if max_items_per_section < 0:
            raise ValueError("max_items_per_section must be >= 0")

        n = max_items_per_section

        # Avoid serializing the full catalog (especially evidence) when only a summary is requested.
        summary: dict[str, Any] = {
            "entities": [asdict(spec) for spec in self.entities[:n]],
            "relations": [asdict(spec) for spec in self.relations[:n]],
            "dimensions": [asdict(spec) for spec in self.dimensions[:n]],
            "measures": [asdict(spec) for spec in self.measures[:n]],
            "metrics": [asdict(spec) for spec in self.metrics[:n]],
            "knowledge_contract": asdict(self.knowledge_contract),
            "warnings": [],
        }
        if include_evidence:
            summary["evidence"] = [asdict(spec) for spec in self.evidence[:n]]
        return summary
