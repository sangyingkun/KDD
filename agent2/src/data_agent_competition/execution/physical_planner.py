from __future__ import annotations

from data_agent_competition.execution.types import ColumnBinding, PhysicalPlan, PhysicalStage, PhysicalStageKind
from data_agent_competition.semantic.types import LogicalPlan


def build_physical_plan(logical_plan: LogicalPlan) -> PhysicalPlan:
    selection_index = {
        f"{selection.source_id}.{selection.field_name}": selection
        for selection in logical_plan.selections
    }
    stages: list[PhysicalStage] = []
    for source in logical_plan.sources:
        stages.append(
            PhysicalStage(
                stage_id=f"load::{source.source_id}",
                kind=PhysicalStageKind.LOAD_SOURCE,
                source_ids=(source.source_id,),
                operation={
                    "asset_path": source.asset_path,
                    "source_kind": source.source_kind.value,
                    "source_id": source.source_id,
                },
            )
        )
    for index, join in enumerate(logical_plan.joins, start=1):
        stages.append(
            PhysicalStage(
                stage_id=f"join::{index}",
                kind=PhysicalStageKind.JOIN,
                source_ids=(join.left_source_id, join.right_source_id),
                operation={
                    "left_source_id": join.left_source_id,
                    "left_field": join.left_field,
                    "right_source_id": join.right_source_id,
                    "right_field": join.right_field,
                    "join_type": join.join_type,
                },
            )
        )
    stages.append(
        PhysicalStage(
            stage_id="project::final",
            kind=PhysicalStageKind.PROJECT,
            source_ids=tuple(source.source_id for source in logical_plan.sources),
            operation={
                "answer_columns": list(logical_plan.answer_columns),
                "answer_bindings": [
                    _answer_binding(
                        qualified_name=qualified_name,
                        logical_plan=logical_plan,
                        selection_index=selection_index,
                    ).to_dict()
                    for qualified_name in logical_plan.answer_columns
                ],
            },
        )
    )
    for index, enrichment in enumerate(logical_plan.post_sql_enrichments, start=1):
        stages.append(
            PhysicalStage(
                stage_id=f"enrich::{index}",
                kind=PhysicalStageKind.ENRICH,
                source_ids=(enrichment.source_id,),
                operation={
                    "source_id": enrichment.source_id,
                    "asset_path": enrichment.asset_path,
                    "source_kind": enrichment.source_kind.value,
                    "match_field": enrichment.match_field,
                    "purpose": enrichment.purpose,
                    "confidence": enrichment.confidence,
                    "rationale": enrichment.rationale,
                },
            )
        )
    return PhysicalPlan(
        task_id=logical_plan.task_id,
        stages=tuple(stages),
        answer_columns=logical_plan.answer_columns,
        answer_bindings=tuple(
            _answer_binding(
                qualified_name=qualified_name,
                logical_plan=logical_plan,
                selection_index=selection_index,
            )
            for qualified_name in logical_plan.answer_columns
        ),
        post_sql_enrichments=tuple(
            {
                "source_id": item.source_id,
                "asset_path": item.asset_path,
                "source_kind": item.source_kind.value,
                "match_field": item.match_field,
                "purpose": item.purpose,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in logical_plan.post_sql_enrichments
        ),
    )


def _answer_binding(
    *,
    qualified_name: str,
    logical_plan: LogicalPlan,
    selection_index: dict[str, object],
) -> ColumnBinding:
    if "." in qualified_name:
        source_id, field_name = qualified_name.rsplit(".", maxsplit=1)
    else:
        source_id = next((source.source_id for source in logical_plan.sources), "")
        field_name = qualified_name
    physical_name = f"{source_id}::{field_name}"
    selection = selection_index.get(f"{source_id}.{field_name}")
    return ColumnBinding(
        source_id=source_id,
        field_name=field_name,
        physical_name=physical_name,
        semantic_dtype=None,
        nullable=True if selection is None else True,
    )
