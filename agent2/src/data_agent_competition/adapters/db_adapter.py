from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatabaseColumn:
    name: str
    declared_type: str
    nullable: bool
    is_primary_key: bool


@dataclass(frozen=True, slots=True)
class DatabaseTable:
    name: str
    columns: tuple[DatabaseColumn, ...]
    sample_rows: tuple[dict[str, Any], ...]


def inspect_database(db_path: Path, *, sample_limit: int = 3) -> tuple[DatabaseTable, ...]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table_names = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        tables: list[DatabaseTable] = []
        for table_name in table_names:
            columns = _inspect_table_columns(connection, table_name)
            sample_rows = _sample_table_rows(connection, table_name, limit=sample_limit)
            tables.append(DatabaseTable(name=table_name, columns=columns, sample_rows=sample_rows))
        return tuple(tables)
    finally:
        connection.close()


def _inspect_table_columns(connection: sqlite3.Connection, table_name: str) -> tuple[DatabaseColumn, ...]:
    pragma_sql = f'PRAGMA table_info("{table_name}")'
    rows = connection.execute(pragma_sql).fetchall()
    return tuple(
        DatabaseColumn(
            name=str(row["name"]),
            declared_type=str(row["type"] or ""),
            nullable=not bool(row["notnull"]),
            is_primary_key=bool(row["pk"]),
        )
        for row in rows
    )


def _sample_table_rows(
    connection: sqlite3.Connection,
    table_name: str,
    *,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    query = f'SELECT * FROM "{table_name}" LIMIT {int(limit)}'
    rows = connection.execute(query).fetchall()
    return tuple({key: row[key] for key in row.keys()} for row in rows)
