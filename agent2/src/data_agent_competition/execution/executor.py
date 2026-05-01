from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.execution.python_fusion_backend import apply_filters, join_rowsets, load_file_source
from data_agent_competition.execution.result_materializer import materialize_answer
from data_agent_competition.execution.sql_backend import load_sql_source
from data_agent_competition.execution.types import BoundRowset, ExecutionResult, PhysicalPlan, PhysicalStageKind, StageResult
from data_agent_competition.runtime.result_writer import AnswerTable
from data_agent_competition.semantic.types import LogicalPlan, TaskBundle


def execute_physical_plan(
    task: TaskBundle,
    artifact: SemanticArtifact,
    logical_plan: LogicalPlan,
    physical_plan: PhysicalPlan,
) -> tuple[ExecutionResult, AnswerTable | None]:
    source_index = {source.source_id: source for source in artifact.sources}
    rowsets: dict[str, BoundRowset] = {}
    stage_results: list[StageResult] = []
    answer_table: AnswerTable | None = None

    try:
        working_rowset: BoundRowset | None = None
        for stage in physical_plan.stages:
            if stage.kind == PhysicalStageKind.LOAD_SOURCE:
                source_id = str(stage.operation["source_id"])
                source_descriptor = source_index[source_id]
                source_filters = tuple(
                    filter_item for filter_item in logical_plan.filters if filter_item.source_id == source_id
                )
                rowset = _load_source_rowset(
                    task.context_dir,
                    source_id,
                    source_descriptor.asset_path,
                    source_descriptor.object_name,
                    source_descriptor.source_kind,
                    source_filters,
                )
                rowset = apply_filters(
                    rowset,
                    tuple(
                        filter_item
                        for filter_item in source_filters
                        if filter_item.operator != "=" or source_descriptor.source_kind != "db"
                    ),
                )
                rowsets[source_id] = rowset
                stage_results.append(_stage_result(rowset))
                if working_rowset is None:
                    working_rowset = rowset
                continue

            if stage.kind == PhysicalStageKind.JOIN:
                left_source_id = str(stage.operation["left_source_id"])
                right_source_id = str(stage.operation["right_source_id"])
                if working_rowset is not None and left_source_id in working_rowset.source_ids:
                    left_rowset = working_rowset
                    right_rowset = rowsets[right_source_id]
                elif working_rowset is not None and right_source_id in working_rowset.source_ids:
                    left_rowset = working_rowset
                    right_rowset = rowsets[left_source_id]
                    left_source_id, right_source_id = right_source_id, left_source_id
                    stage_left_field = str(stage.operation["right_field"])
                    stage_right_field = str(stage.operation["left_field"])
                else:
                    left_rowset = rowsets[left_source_id]
                    right_rowset = rowsets[right_source_id]
                    stage_left_field = str(stage.operation["left_field"])
                    stage_right_field = str(stage.operation["right_field"])
                if "stage_left_field" not in locals():
                    stage_left_field = str(stage.operation["left_field"])
                    stage_right_field = str(stage.operation["right_field"])
                working_rowset = join_rowsets(
                    left_rowset,
                    right_rowset,
                    left_source_id=left_source_id,
                    left_field=stage_left_field,
                    right_source_id=right_source_id,
                    right_field=stage_right_field,
                    join_type=str(stage.operation["join_type"]),
                    stage_id=stage.stage_id,
                )
                del stage_left_field
                del stage_right_field
                stage_results.append(_stage_result(working_rowset))
                continue

            if stage.kind == PhysicalStageKind.PROJECT:
                if working_rowset is None:
                    raise ValueError("Project stage encountered before any working rowset was produced.")
                working_rowset = _apply_aggregations(working_rowset, logical_plan)
                answer_table = materialize_answer(working_rowset, logical_plan, physical_plan)
                stage_results.append(
                    StageResult(
                        stage_id=stage.stage_id,
                        row_count=len(answer_table.rows),
                        columns=answer_table.columns,
                        bindings=physical_plan.answer_bindings,
                    )
                )
                continue

            if stage.kind == PhysicalStageKind.ENRICH:
                if answer_table is None:
                    raise ValueError("Enrich stage encountered before answer materialization.")
                answer_table = _apply_post_sql_enrichment(
                    answer_table=answer_table,
                    task=task,
                    source_index=source_index,
                    operation=stage.operation,
                )
                stage_results.append(
                    StageResult(
                        stage_id=stage.stage_id,
                        row_count=len(answer_table.rows),
                        columns=answer_table.columns,
                    )
                )
                continue

        if answer_table is None:
            raise ValueError("Physical plan completed without a project stage.")
        return (
            ExecutionResult(
                succeeded=True,
                answer_rows=answer_table.rows,
                answer_columns=answer_table.columns,
                stage_results=tuple(stage_results),
            ),
            answer_table,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            ExecutionResult(
                succeeded=False,
                answer_rows=(),
                answer_columns=(),
                stage_results=tuple(stage_results),
                failure_reason=str(exc),
            ),
            None,
        )


def _load_source_rowset(
    context_dir: Path,
    source_id: str,
    asset_path: str,
    object_name: str,
    source_kind: str,
    source_filters,
) -> BoundRowset:
    absolute_path = context_dir / asset_path
    if source_kind == "db":
        return load_sql_source(absolute_path, object_name, source_id, source_filters)
    return load_file_source(path=absolute_path, source_id=source_id)


def _apply_aggregations(rowset: BoundRowset, logical_plan: LogicalPlan) -> BoundRowset:
    if not logical_plan.aggregations:
        return rowset

    frame = rowset.frame
    binding_map = rowset.qualified_binding_map()
    aggregation_spec: dict[str, str] = {}
    alias_by_field: dict[str, str] = {}
    numeric_fields: list[str] = []
    for aggregation in logical_plan.aggregations:
        binding = binding_map.get(f"{aggregation.source_id}.{aggregation.field_name}")
        if binding is None or binding.physical_name not in frame.columns:
            continue
        aggregation_spec[binding.physical_name] = aggregation.function
        alias_by_field[binding.physical_name] = aggregation.alias or binding.physical_name
        if aggregation.function in {"sum", "avg", "mean", "min", "max"}:
            numeric_fields.append(binding.physical_name)
    if not aggregation_spec:
        return rowset
    for field_name in numeric_fields:
        frame[field_name] = pd.to_numeric(frame[field_name], errors="coerce")

    group_columns = [
        binding_map[column].physical_name
        for column in logical_plan.answer_columns
        if column in binding_map and binding_map[column].physical_name not in aggregation_spec
    ]
    if group_columns:
        aggregated = frame.groupby(group_columns, dropna=False).agg(aggregation_spec).reset_index()
    else:
        aggregated = frame.agg(aggregation_spec).to_frame().T
    aggregated = aggregated.rename(columns=alias_by_field).reset_index(drop=True)
    aggregation_bindings = list(rowset.bindings)
    return BoundRowset(
        stage_id=rowset.stage_id,
        source_ids=rowset.source_ids,
        frame=aggregated,
        bindings=tuple(aggregation_bindings),
    )


def _stage_result(rowset: BoundRowset) -> StageResult:
    return StageResult(
        stage_id=rowset.stage_id,
        row_count=len(rowset.frame),
        columns=tuple(str(column) for column in rowset.frame.columns),
        bindings=rowset.bindings,
        dtypes={column: str(dtype) for column, dtype in rowset.frame.dtypes.items()},
    )


def _apply_post_sql_enrichment(
    *,
    answer_table: AnswerTable,
    task: TaskBundle,
    source_index: dict[str, object],
    operation: dict[str, object],
) -> AnswerTable:
    source_id = str(operation["source_id"])
    source_descriptor = source_index.get(source_id)
    if source_descriptor is None:
        return answer_table
    match_field = str(operation.get("match_field") or "").strip()
    if not match_field:
        return answer_table
    source_rowset = _load_source_rowset(
        task.context_dir,
        source_id,
        source_descriptor.asset_path,
        source_descriptor.object_name,
        source_descriptor.source_kind,
        (),
    )
    if not answer_table.rows:
        return answer_table

    answer_frame = pd.DataFrame(answer_table.rows, columns=answer_table.columns)
    answer_match_column = _best_answer_match_column(answer_table.columns, match_field)
    if answer_match_column is None:
        return answer_table
    source_match_physical = f"{source_id}::{match_field}"
    if source_match_physical not in source_rowset.frame.columns:
        return answer_table

    enrichment_frame = source_rowset.frame.copy()
    enrichment_frame = enrichment_frame.rename(
        columns={
            column: column.split("::", maxsplit=1)[-1]
            for column in enrichment_frame.columns
        }
    )
    enrichment_columns = [
        column
        for column in enrichment_frame.columns
        if column != match_field and column not in answer_frame.columns
    ]
    if not enrichment_columns:
        return answer_table
    merged = answer_frame.merge(
        enrichment_frame[[match_field, *enrichment_columns]],
        how="left",
        left_on=answer_match_column,
        right_on=match_field,
    )
    if match_field != answer_match_column and match_field in merged.columns:
        merged = merged.drop(columns=[match_field])
    rows = tuple(tuple(_to_output_scalar(value) for value in row) for row in merged.itertuples(index=False, name=None))
    return AnswerTable(columns=tuple(str(column) for column in merged.columns), rows=rows)


def _best_answer_match_column(columns: tuple[str, ...], match_field: str) -> str | None:
    lowered_match = match_field.lower()
    for column in columns:
        if column.lower() == lowered_match:
            return column
    for column in columns:
        if lowered_match in column.lower() or column.lower() in lowered_match:
            return column
    return None


def _to_output_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    return value
