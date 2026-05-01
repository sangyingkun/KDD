from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GraphNodeType(str, Enum):
    SOURCE = "source"
    FIELD = "field"
    JOIN = "join"
    METRIC = "metric"
    CONSTRAINT = "constraint"
    VALUE_CONCEPT = "value_concept"
    AMBIGUITY = "ambiguity"
    USE_CASE = "use_case"


class GraphEdgeType(str, Enum):
    HAS_FIELD = "has_field"
    JOIN_ON = "join_on"
    ALIAS_OF = "alias_of"
    MAPS_VALUE = "maps_value"
    CONSTRAINS = "constrains"
    USES_FIELD = "uses_field"
    USES_SOURCE = "uses_source"
    DISAMBIGUATES = "disambiguates"
    EVIDENCED_BY = "evidenced_by"
    CO_OCCURS_WITH = "co_occurs_with"


@dataclass(frozen=True, slots=True)
class GraphNodeHit:
    node_id: str
    node_type: str
    label: str
    canonical_text: str
    score: float
    lexical_score: float
    dense_score: float
    hop_distance: int
    source_id: str | None = None
    field_name: str | None = None
    resolved_value: str | None = None
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class GraphEdgeHit:
    edge_id: str
    edge_type: str
    source_node_id: str
    target_node_id: str
    score: float
    hop_distance: int
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class GraphRetrievalResult:
    seed_node_ids: tuple[str, ...]
    expanded_node_ids: tuple[str, ...]
    node_hits: tuple[GraphNodeHit, ...]
    edge_hits: tuple[GraphEdgeHit, ...]
    notes: tuple[str, ...] = ()
    context_sections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphCandidate:
    node_id: str
    node_type: str
    label: str
    canonical_text: str
    score: float
    metadata: dict[str, Any]
    source_id: str | None = None
    field_name: str | None = None
    resolved_value: str | None = None
    rationale: str = ""
    evidence_node_ids: tuple[str, ...] = ()
    evidence_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphCandidateSet:
    source_candidates: tuple[GraphCandidate, ...]
    field_candidates: tuple[GraphCandidate, ...]
    join_candidates: tuple[GraphCandidate, ...]
    value_candidates: tuple[GraphCandidate, ...]
    metric_candidates: tuple[GraphCandidate, ...]
    constraint_candidates: tuple[GraphCandidate, ...]
    ambiguity_candidates: tuple[GraphCandidate, ...]
    use_case_candidates: tuple[GraphCandidate, ...]
    evidence_node_ids: tuple[str, ...]
    evidence_edge_ids: tuple[str, ...]

    @property
    def source_node_ids(self) -> tuple[str, ...]:
        return tuple(candidate.node_id for candidate in self.source_candidates)

    @property
    def field_node_ids(self) -> tuple[str, ...]:
        return tuple(candidate.node_id for candidate in self.field_candidates)

    @property
    def join_node_ids(self) -> tuple[str, ...]:
        return tuple(candidate.node_id for candidate in self.join_candidates)

    @property
    def value_node_ids(self) -> tuple[str, ...]:
        return tuple(candidate.node_id for candidate in self.value_candidates)
