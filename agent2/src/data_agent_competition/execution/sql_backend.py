from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from data_agent_competition.execution.types import BoundRowset, ColumnBinding
from data_agent_competition.execution.sql_subplanner import build_select_sql
from data_agent_competition.semantic.types import LogicalFilter


def load_sql_source(
    db_path: Path,
    table_name: str,
    source_id: str,
    filters: tuple[LogicalFilter, ...],
) -> BoundRowset:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        pushdown_filters = tuple(filter_item for filter_item in filters if filter_item.operator == "=")
        query, parameters = build_select_sql(table_name, pushdown_filters)
        frame = pd.read_sql_query(query, connection, params=parameters, dtype=object).fillna("")
    finally:
        connection.close()
    renamed_columns = {column: f"{source_id}::{column}" for column in frame.columns}
    bound_frame = frame.rename(columns=renamed_columns).reset_index(drop=True)
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
        stage_id=f"load::{source_id}",
        source_ids=(source_id,),
        frame=bound_frame,
        bindings=bindings,
    )
