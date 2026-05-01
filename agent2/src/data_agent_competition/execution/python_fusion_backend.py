from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_agent_competition.execution.types import BoundRowset, ColumnBinding
from data_agent_competition.semantic.types import LogicalFilter


def load_file_source(*, path: Path, source_id: str) -> BoundRowset:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, dtype=object, keep_default_na=False)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            frame = pd.json_normalize(payload)
        elif isinstance(payload, dict):
            list_values = [value for value in payload.values() if isinstance(value, list)]
            if list_values and all(isinstance(item, dict) for item in list_values[0]):
                frame = pd.json_normalize(list_values[0])
            else:
                frame = pd.json_normalize(payload)
        else:
            raise ValueError(f"Unsupported JSON root for {path}")
        frame = frame.fillna("")
    else:
        raise ValueError(f"Unsupported file source: {path}")
    return _bind_source_frame(frame, source_id=source_id, stage_id=f"load::{source_id}")


def apply_filters(rowset: BoundRowset, filters: tuple[LogicalFilter, ...]) -> BoundRowset:
    result = rowset.frame.copy()
    binding_map = rowset.qualified_binding_map()
    for filter_item in filters:
        binding = binding_map.get(f"{filter_item.source_id}.{filter_item.field_name}")
        if binding is None or binding.physical_name not in result.columns:
            continue
        column_name = binding.physical_name
        if filter_item.operator == "=":
            result = result[result[column_name].map(_normalize_scalar) == _normalize_scalar(filter_item.value)]
        elif filter_item.operator == "month_year_equals":
            series = pd.to_datetime(result[column_name], errors="coerce")
            result = result[series.dt.strftime("%Y-%m") == filter_item.value]
        elif filter_item.operator == "year_equals":
            series = pd.to_datetime(result[column_name], errors="coerce")
            result = result[series.dt.strftime("%Y") == filter_item.value]
        elif filter_item.operator == "month_equals":
            series = pd.to_datetime(result[column_name], errors="coerce")
            result = result[series.dt.strftime("%m") == filter_item.value]
    return BoundRowset(
        stage_id=rowset.stage_id,
        source_ids=rowset.source_ids,
        frame=result.reset_index(drop=True),
        bindings=rowset.bindings,
    )


def join_rowsets(
    left_rowset: BoundRowset,
    right_rowset: BoundRowset,
    *,
    left_source_id: str,
    left_field: str,
    right_source_id: str,
    right_field: str,
    join_type: str,
    stage_id: str,
) -> BoundRowset:
    left_binding = left_rowset.qualified_binding_map().get(f"{left_source_id}.{left_field}")
    right_binding = right_rowset.qualified_binding_map().get(f"{right_source_id}.{right_field}")
    if left_binding is None or right_binding is None:
        raise ValueError(
            f"Join binding missing for {left_source_id}.{left_field} -> {right_source_id}.{right_field}"
        )
    left_frame = left_rowset.frame.copy()
    right_frame = right_rowset.frame.copy()
    left_frame[left_binding.physical_name] = left_frame[left_binding.physical_name].map(_normalize_join_key)
    right_frame[right_binding.physical_name] = right_frame[right_binding.physical_name].map(_normalize_join_key)
    merged = left_frame.merge(
        right_frame,
        how=join_type,
        left_on=left_binding.physical_name,
        right_on=right_binding.physical_name,
        suffixes=("", "__dup"),
    )
    duplicate_columns = [column for column in merged.columns if column.endswith("__dup")]
    if duplicate_columns:
        raise ValueError(f"Unexpected duplicate physical columns after join: {duplicate_columns}")
    bindings = tuple(_deduplicate_bindings(left_rowset.bindings + right_rowset.bindings))
    return BoundRowset(
        stage_id=stage_id,
        source_ids=tuple(dict.fromkeys(left_rowset.source_ids + right_rowset.source_ids)),
        frame=merged.reset_index(drop=True),
        bindings=bindings,
    )


def _bind_source_frame(frame: pd.DataFrame, *, source_id: str, stage_id: str) -> BoundRowset:
    renamed_columns = {column: _physical_name(source_id, str(column)) for column in frame.columns}
    bound_frame = frame.rename(columns=renamed_columns).reset_index(drop=True)
    for column in bound_frame.columns:
        bound_frame[column] = bound_frame[column].map(_preserve_scalar)
    bindings = tuple(
        ColumnBinding(
            source_id=source_id,
            field_name=str(original_name),
            physical_name=physical_name,
            semantic_dtype=None,
            nullable=True,
        )
        for original_name, physical_name in renamed_columns.items()
    )
    return BoundRowset(
        stage_id=stage_id,
        source_ids=(source_id,),
        frame=bound_frame,
        bindings=bindings,
    )


def _deduplicate_bindings(bindings: tuple[ColumnBinding, ...] | list[ColumnBinding]) -> list[ColumnBinding]:
    deduped: dict[str, ColumnBinding] = {}
    for binding in bindings:
        deduped[binding.qualified_name] = binding
    return list(deduped.values())


def _physical_name(source_id: str, field_name: str) -> str:
    return f"{source_id}::{field_name}"


def _preserve_scalar(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _normalize_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalize_join_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    rendered = str(value).strip()
    if rendered.endswith(".0"):
        return rendered[:-2]
    return rendered
