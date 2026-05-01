from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class AssetKind(str, Enum):
    DB = "db"
    CSV = "csv"
    JSON = "json"
    DOC = "doc"
    KNOWLEDGE = "knowledge"
    OTHER = "other"


class QuestionType(str, Enum):
    LOOKUP = "lookup"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    RANKING = "ranking"
    TEMPORAL = "temporal"
    EXTRACTION = "extraction"
    EXPLANATORY = "explanatory"
    HYBRID = "hybrid"


class SelectionRole(str, Enum):
    DIMENSION = "dimension"
    MEASURE = "measure"
    FILTER = "filter"
    JOIN_KEY = "join_key"
    ORDER_BY = "order_by"
    ANSWER = "answer"


class ExecutionHint(str, Enum):
    SQL_ONLY = "sql_only"
    HYBRID = "hybrid"
    PYTHON_ONLY = "python_only"


class RoutingAnswerKind(str, Enum):
    IDENTIFIER = "identifier"
    ATTRIBUTE = "attribute"
    MEASURE = "measure"
    GROUP_BY = "group_by"


@dataclass(frozen=True, slots=True)
class ContextAsset:
    relative_path: str
    absolute_path: Path
    kind: AssetKind
    size_bytes: int


@dataclass(frozen=True, slots=True)
class TaskBundle:
    task_id: str
    difficulty: str
    question: str
    task_dir: Path
    context_dir: Path
    assets: tuple[ContextAsset, ...]

    def assets_by_kind(self, kind: AssetKind) -> tuple[ContextAsset, ...]:
        return tuple(asset for asset in self.assets if asset.kind == kind)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_dir"] = str(self.task_dir)
        payload["context_dir"] = str(self.context_dir)
        for asset in payload["assets"]:
            asset["absolute_path"] = str(asset["absolute_path"])
        return payload


@dataclass(frozen=True, slots=True)
class GroundedTerm:
    term: str
    normalized_term: str
    grounding_type: str
    resolved_value: str | None = None
    source_id: str | None = None
    field_name: str | None = None
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    graph_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class GroundingResult:
    question_type: QuestionType
    grounded_terms: tuple[GroundedTerm, ...]
    unresolved_terms: tuple[str, ...] = ()
    supporting_facts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceScope:
    source_id: str
    source_kind: AssetKind
    asset_path: str
    rationale: str
    confidence: float = 0.0
    priority: int = 0
    required_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class JoinEdge:
    left_source_id: str
    left_field: str
    right_source_id: str
    right_field: str
    join_type: str = "inner"
    confidence: float = 0.0
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class RoutingFieldRef:
    source_id: str
    field_name: str
    answer_kind: RoutingAnswerKind
    confidence: float = 0.0
    rationale: str = ""
    graph_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoutingFilter:
    source_id: str
    field_name: str
    operator: str
    value: str
    confidence: float = 0.0
    rationale: str = ""
    graph_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoutingMetric:
    node_id: str
    label: str
    formula: str | None = None
    source_fields: tuple[str, ...] = ()
    requires_time_grain: str | None = None
    confidence: float = 0.0
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class PostSQLEnrichment:
    source_id: str
    asset_path: str
    source_kind: AssetKind
    match_field: str | None = None
    purpose: str = ""
    confidence: float = 0.0
    rationale: str = ""
    graph_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class AmbiguityWarning:
    node_id: str
    label: str
    message: str
    preferred_source_id: str | None = None
    preferred_field_name: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class SemanticRoutingSpec:
    question_type: QuestionType
    target_sources: tuple[SourceScope, ...]
    join_path: tuple[JoinEdge, ...]
    answer_slots: tuple[RoutingFieldRef, ...]
    metrics: tuple[RoutingMetric, ...] = ()
    filters: tuple[RoutingFilter, ...] = ()
    time_constraints: tuple[RoutingFilter, ...] = ()
    post_sql_enrichments: tuple[PostSQLEnrichment, ...] = ()
    ambiguity_warnings: tuple[AmbiguityWarning, ...] = ()
    supporting_node_ids: tuple[str, ...] = ()
    supporting_edge_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class TargetGrain:
    entity: str
    grain_fields: tuple[str, ...]
    time_grain: str | None = None
    measure_scope: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticSelection:
    source_id: str
    field_name: str
    role: SelectionRole
    alias: str | None = None
    expression: str | None = None
    confidence: float = 0.0
    rationale: str = ""
    graph_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class LogicalFilter:
    source_id: str
    field_name: str
    operator: str
    value: str
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class LogicalAggregation:
    source_id: str
    field_name: str
    function: str
    alias: str | None = None
    distinct: bool = False


@dataclass(frozen=True, slots=True)
class LogicalOrdering:
    source_id: str
    field_name: str
    direction: str = "asc"


@dataclass(frozen=True, slots=True)
class LogicalPlan:
    task_id: str
    question_type: QuestionType
    target_grain: TargetGrain
    sources: tuple[SourceScope, ...]
    joins: tuple[JoinEdge, ...]
    selections: tuple[SemanticSelection, ...]
    filters: tuple[LogicalFilter, ...] = ()
    aggregations: tuple[LogicalAggregation, ...] = ()
    orderings: tuple[LogicalOrdering, ...] = ()
    answer_columns: tuple[str, ...] = ()
    execution_hint: ExecutionHint = ExecutionHint.HYBRID
    limit: int | None = None
    notes: tuple[str, ...] = ()
    verification_focus: tuple[str, ...] = ()
    answer_aliases: dict[str, str] | None = None
    post_sql_enrichments: tuple[PostSQLEnrichment, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class ArtifactLoadResult:
    artifact: Any
    mode: str
    artifact_path: str | None = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
