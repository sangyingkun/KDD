from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class PhysicalStageKind(str, Enum):
    LOAD_SOURCE = "load_source"
    JOIN = "join"
    PROJECT = "project"
    ENRICH = "enrich"


@dataclass(frozen=True, slots=True)
class PhysicalStage:
    stage_id: str
    kind: PhysicalStageKind
    source_ids: tuple[str, ...]
    operation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True, slots=True)
class ColumnBinding:
    source_id: str
    field_name: str
    physical_name: str
    semantic_dtype: str | None = None
    nullable: bool = True

    @property
    def qualified_name(self) -> str:
        return f"{self.source_id}.{self.field_name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "field_name": self.field_name,
            "qualified_name": self.qualified_name,
            "physical_name": self.physical_name,
            "semantic_dtype": self.semantic_dtype,
            "nullable": self.nullable,
        }


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    task_id: str
    stages: tuple[PhysicalStage, ...]
    answer_columns: tuple[str, ...]
    answer_bindings: tuple[ColumnBinding, ...]
    post_sql_enrichments: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stages": [stage.to_dict() for stage in self.stages],
            "answer_columns": list(self.answer_columns),
            "answer_bindings": [binding.to_dict() for binding in self.answer_bindings],
            "post_sql_enrichments": list(self.post_sql_enrichments),
        }


@dataclass(frozen=True, slots=True)
class BoundRowset:
    stage_id: str
    source_ids: tuple[str, ...]
    frame: Any
    bindings: tuple[ColumnBinding, ...]

    def qualified_binding_map(self) -> dict[str, ColumnBinding]:
        return {binding.qualified_name: binding for binding in self.bindings}


@dataclass(frozen=True, slots=True)
class StageResult:
    stage_id: str
    row_count: int
    columns: tuple[str, ...]
    bindings: tuple[ColumnBinding, ...] = ()
    dtypes: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "bindings": [binding.to_dict() for binding in self.bindings],
            "dtypes": dict(self.dtypes or {}),
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    succeeded: bool
    answer_rows: tuple[tuple[Any, ...], ...]
    answer_columns: tuple[str, ...]
    stage_results: tuple[StageResult, ...]
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "answer_columns": list(self.answer_columns),
            "answer_rows": [list(row) for row in self.answer_rows],
            "stage_results": [result.to_dict() for result in self.stage_results],
            "failure_reason": self.failure_reason,
        }
