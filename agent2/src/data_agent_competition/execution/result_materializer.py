from __future__ import annotations

from typing import Any

import pandas as pd

from data_agent_competition.execution.types import BoundRowset, ColumnBinding, PhysicalPlan
from data_agent_competition.runtime.result_writer import AnswerTable
from data_agent_competition.semantic.types import LogicalPlan


def materialize_answer(
    rowset: BoundRowset,
    logical_plan: LogicalPlan,
    physical_plan: PhysicalPlan,
) -> AnswerTable:
    projection_map = _resolve_projection_map(rowset, physical_plan)
    selected_columns = [binding.physical_name for binding in projection_map]
    alias_map = logical_plan.answer_aliases or {}
    renamed_columns = {
        binding.physical_name: alias_map.get(binding.qualified_name, binding.field_name)
        for binding in projection_map
    }
    projected = rowset.frame[selected_columns].drop_duplicates().rename(columns=renamed_columns)
    rows = tuple(tuple(_to_scalar(value) for value in row) for row in projected.itertuples(index=False, name=None))
    return AnswerTable(columns=tuple(projected.columns), rows=rows)


def _resolve_projection_map(
    rowset: BoundRowset,
    physical_plan: PhysicalPlan,
) -> tuple[ColumnBinding, ...]:
    binding_map = rowset.qualified_binding_map()
    resolved: list[ColumnBinding] = []
    for answer_binding in physical_plan.answer_bindings:
        binding = binding_map.get(answer_binding.qualified_name)
        if binding is None:
            raise ValueError(f"Missing bound answer column: {answer_binding.qualified_name}")
        resolved.append(binding)
    return tuple(resolved)


def _to_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value
