from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.execution.types import ExecutionResult
from data_agent_competition.runtime.result_writer import AnswerTable
from data_agent_competition.semantic.types import LogicalPlan, SemanticRoutingSpec, TaskBundle


class VerificationStage(str, Enum):
    LOGICAL_PLAN = "logical_plan"
    GRAPH = "graph"
    ROUTING = "routing"
    EXECUTION = "execution"
    ANSWER = "answer"


class VerificationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: str
    message: str
    stage: VerificationStage = VerificationStage.LOGICAL_PLAN
    severity: VerificationSeverity = VerificationSeverity.ERROR
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage.value,
            "severity": self.severity.value,
            "metadata": _json_ready(self.metadata) if self.metadata else None,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    ok: bool
    issues: tuple[VerificationIssue, ...]
    summary: str = ""
    schema_version: str = "verification.v1"

    @property
    def primary_issue(self) -> VerificationIssue | None:
        return self.issues[0] if self.issues else None

    @property
    def signature(self) -> str | None:
        issue = self.primary_issue
        return None if issue is None else issue.code

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "signature": self.signature,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def verify_logical_plan(
    *,
    task: TaskBundle,
    plan: LogicalPlan,
    artifact: SemanticArtifact,
    routing_spec: SemanticRoutingSpec | None = None,
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    if not plan.sources:
        issues.append(_issue("missing_sources", "Logical plan has no scoped sources."))
    if not plan.answer_columns:
        issues.append(_issue("missing_answer_columns", "No answer columns were selected."))
    if plan.target_grain is None or not plan.target_grain.entity:
        issues.append(_issue("missing_target_grain", "Target grain is not resolved."))
    if len(plan.sources) > 1 and not plan.joins:
        issues.append(_issue("missing_joins", "Multiple sources were scoped without joins."))
    issues.extend(_answer_column_consistency_issues(task, plan, artifact))
    issues.extend(_graph_consistency_issues(plan, artifact))
    if routing_spec is not None:
        issues.extend(_routing_spec_consistency_issues(plan, routing_spec))
    return _result(issues, summary="logical plan verification")


def verify_execution_outcome(
    *,
    logical_plan: LogicalPlan | None,
    execution_result: ExecutionResult | None,
    answer_table: AnswerTable | None,
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    if logical_plan is None:
        issues.append(
            _issue(
                "missing_logical_plan",
                "Logical plan missing during execution verification.",
                stage=VerificationStage.EXECUTION,
            )
        )
        return _result(issues, summary="execution verification")
    if execution_result is None:
        issues.append(
            _issue(
                "missing_execution_result",
                "Execution result missing.",
                stage=VerificationStage.EXECUTION,
            )
        )
        return _result(issues, summary="execution verification")
    if not execution_result.succeeded:
        issues.append(
            _issue(
                "execution_failed",
                execution_result.failure_reason or "Execution did not succeed.",
                stage=VerificationStage.EXECUTION,
            )
        )
        return _result(issues, summary="execution verification")
    if answer_table is None:
        issues.append(
            _issue(
                "missing_answer_table",
                "Execution succeeded without answer materialization.",
                stage=VerificationStage.EXECUTION,
            )
        )
    else:
        expected_columns = _expected_output_columns(logical_plan)
        if expected_columns and answer_table.columns != expected_columns:
            issues.append(
                _issue(
                    "answer_shape_mismatch",
                    f"Expected answer columns {list(expected_columns)}, got {list(answer_table.columns)}.",
                    stage=VerificationStage.ANSWER,
                    expected_columns=list(expected_columns),
                    actual_columns=list(answer_table.columns),
                )
            )
        issues.extend(_answer_table_quality_issues(logical_plan, answer_table))
    if answer_table is not None and not answer_table.rows and not logical_plan.filters:
        issues.append(_issue("empty_answer", "Answer table is empty.", stage=VerificationStage.ANSWER))
    return _result(issues, summary="execution verification")


def _expected_output_columns(logical_plan: LogicalPlan) -> tuple[str, ...]:
    if logical_plan.answer_aliases:
        return tuple(logical_plan.answer_aliases.get(column, column.split(".")[-1]) for column in logical_plan.answer_columns)
    return tuple(column.split(".")[-1] for column in logical_plan.answer_columns)


def _answer_column_consistency_issues(
    task: TaskBundle,
    plan: LogicalPlan,
    artifact: SemanticArtifact,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    source_fields = {
        source.source_id: {field.field_name for field in source.fields}
        for source in artifact.sources
    }
    for column in plan.answer_columns:
        if "." not in column:
            continue
        source_id, field_name = column.rsplit(".", maxsplit=1)
        if field_name not in source_fields.get(source_id, set()):
            issues.append(
                _issue(
                    "invalid_answer_column",
                    f"Answer column {column} is not present in the scoped semantic artifact.",
                    source_id=source_id,
                    field_name=field_name,
                )
            )
    grain_fields = {field.lower() for field in plan.target_grain.grain_fields}
    if grain_fields and not any(column.split(".")[-1].lower() in grain_fields for column in plan.answer_columns):
        issues.append(
            _issue(
                "target_grain_not_projected",
                f"Question '{task.question}' resolved grain {sorted(grain_fields)} but answer columns do not project it.",
                grain_fields=sorted(grain_fields),
            )
        )
    return issues


def _answer_table_quality_issues(
    logical_plan: LogicalPlan,
    answer_table: AnswerTable,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    if not answer_table.rows:
        return issues
    column_index = {column: index for index, column in enumerate(answer_table.columns)}
    if "ID" in column_index:
        id_values = [row[column_index["ID"]] for row in answer_table.rows]
        if any(value in (None, "") for value in id_values):
            issues.append(
                _issue("null_identifier", "Answer rows contain null identifier values.", stage=VerificationStage.ANSWER)
            )
    for alias in ("Disease", "Diagnosis"):
        if alias not in column_index:
            continue
        values = [row[column_index[alias]] for row in answer_table.rows]
        null_ratio = sum(1 for value in values if value in (None, "")) / max(len(values), 1)
        if null_ratio >= 0.34:
            issues.append(
                _issue(
                    "sparse_answer_column",
                    f"Answer column {alias} is sparse in the materialized answer.",
                    stage=VerificationStage.ANSWER,
                    answer_column=alias,
                    null_ratio=round(null_ratio, 4),
                )
            )
        break
    if logical_plan.target_grain.grain_fields and "ID" in column_index:
        ids = [row[column_index["ID"]] for row in answer_table.rows]
        normalized_ids = [str(value) for value in ids if value not in (None, "")]
        if len(normalized_ids) != len(set(normalized_ids)):
            issues.append(
                _issue(
                    "duplicate_target_grain",
                    "Materialized answer contains duplicate target grain rows.",
                    stage=VerificationStage.ANSWER,
                    distinct_count=len(set(normalized_ids)),
                    row_count=len(normalized_ids),
                )
            )
        if any(identifier.endswith(".0") for identifier in normalized_ids):
            issues.append(
                _issue(
                    "identifier_coercion",
                    "Identifier values were coerced to float-like strings during execution.",
                    stage=VerificationStage.ANSWER,
                )
            )
    return issues


def _graph_consistency_issues(
    plan: LogicalPlan,
    artifact: SemanticArtifact,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    graph_node_ids = {node.node_id for node in artifact.graph_nodes}
    join_signatures = {
        (candidate.left_source_id, candidate.left_field, candidate.right_source_id, candidate.right_field)
        for candidate in artifact.join_candidates
    }
    source_fields = {
        (source.source_id, field.field_name)
        for source in artifact.sources
        for field in source.fields
    }
    for selection in plan.selections:
        if selection.graph_node_id and selection.graph_node_id not in graph_node_ids:
            issues.append(
                _issue(
                    "selection_graph_node_missing",
                    f"Selection {selection.source_id}.{selection.field_name} references a missing graph node.",
                    stage=VerificationStage.GRAPH,
                    graph_node_id=selection.graph_node_id,
                )
            )
    for logical_filter in plan.filters:
        if (logical_filter.source_id, logical_filter.field_name) not in source_fields:
            issues.append(
                _issue(
                    "filter_field_missing",
                    f"Filter {logical_filter.source_id}.{logical_filter.field_name} is not present in artifact sources.",
                    stage=VerificationStage.GRAPH,
                    source_id=logical_filter.source_id,
                    field_name=logical_filter.field_name,
                )
            )
    for join in plan.joins:
        direct = (join.left_source_id, join.left_field, join.right_source_id, join.right_field)
        reverse = (join.right_source_id, join.right_field, join.left_source_id, join.left_field)
        if direct not in join_signatures and reverse not in join_signatures:
            issues.append(
                _issue(
                    "join_not_in_graph",
                    (
                        f"Join {join.left_source_id}.{join.left_field} = "
                        f"{join.right_source_id}.{join.right_field} is not supported by the task semantic graph."
                    ),
                    stage=VerificationStage.GRAPH,
                    left_source_id=join.left_source_id,
                    left_field=join.left_field,
                    right_source_id=join.right_source_id,
                    right_field=join.right_field,
                )
            )
    return issues


def _routing_spec_consistency_issues(
    plan: LogicalPlan,
    routing_spec: SemanticRoutingSpec,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    allowed_answer_columns = {
        f"{slot.source_id}.{slot.field_name}"
        for slot in routing_spec.answer_slots
    }
    allowed_filter_fields = {
        (item.source_id, item.field_name, item.operator, item.value)
        for item in (*routing_spec.filters, *routing_spec.time_constraints)
    }
    allowed_joins = {
        (join.left_source_id, join.left_field, join.right_source_id, join.right_field)
        for join in routing_spec.join_path
    }
    for column in plan.answer_columns:
        if allowed_answer_columns and column not in allowed_answer_columns and not any(
            column == f"{aggregation.source_id}.{aggregation.field_name}" for aggregation in plan.aggregations
        ):
            issues.append(
                _issue(
                    "answer_column_not_in_routing_spec",
                    f"Answer column {column} is not supported by the semantic routing spec.",
                    stage=VerificationStage.ROUTING,
                    answer_column=column,
                )
            )
    for item in plan.filters:
        if allowed_filter_fields and (
            item.source_id,
            item.field_name,
            item.operator,
            item.value,
        ) not in allowed_filter_fields:
            issues.append(
                _issue(
                    "filter_not_in_routing_spec",
                    (
                        f"Filter {item.source_id}.{item.field_name} {item.operator} {item.value} "
                        "is not supported by the semantic routing spec."
                    ),
                    stage=VerificationStage.ROUTING,
                    source_id=item.source_id,
                    field_name=item.field_name,
                    operator=item.operator,
                    value=item.value,
                )
            )
    for join in plan.joins:
        direct = (join.left_source_id, join.left_field, join.right_source_id, join.right_field)
        reverse = (join.right_source_id, join.right_field, join.left_source_id, join.left_field)
        if allowed_joins and direct not in allowed_joins and reverse not in allowed_joins:
            issues.append(
                _issue(
                    "join_not_in_routing_spec",
                    (
                        f"Join {join.left_source_id}.{join.left_field} = "
                        f"{join.right_source_id}.{join.right_field} is not supported by the semantic routing spec."
                    ),
                    stage=VerificationStage.ROUTING,
                    left_source_id=join.left_source_id,
                    left_field=join.left_field,
                    right_source_id=join.right_source_id,
                    right_field=join.right_field,
                )
            )
    if routing_spec.ambiguity_warnings and not plan.notes:
        issues.append(
            _issue(
                "ambiguity_not_reflected",
                "Semantic routing spec contains ambiguity warnings but logical plan notes are empty.",
                stage=VerificationStage.ROUTING,
                ambiguity_warning_count=len(routing_spec.ambiguity_warnings),
            )
        )
    return issues


def _issue(
    code: str,
    message: str,
    *,
    stage: VerificationStage = VerificationStage.LOGICAL_PLAN,
    severity: VerificationSeverity = VerificationSeverity.ERROR,
    **metadata: Any,
) -> VerificationIssue:
    normalized_metadata = {key: _json_ready(value) for key, value in metadata.items()} if metadata else None
    return VerificationIssue(
        code=code,
        message=message,
        stage=stage,
        severity=severity,
        metadata=normalized_metadata,
    )


def _result(issues: list[VerificationIssue], *, summary: str) -> VerificationResult:
    ok = not any(issue.severity == VerificationSeverity.ERROR for issue in issues)
    return VerificationResult(ok=ok, issues=tuple(issues), summary=summary)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value
