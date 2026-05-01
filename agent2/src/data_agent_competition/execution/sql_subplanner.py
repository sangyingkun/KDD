from __future__ import annotations

from data_agent_competition.semantic.types import LogicalFilter


def build_select_sql(table_name: str, filters: tuple[LogicalFilter, ...]) -> tuple[str, list[str]]:
    query = f'SELECT * FROM "{table_name}"'
    predicates: list[str] = []
    parameters: list[str] = []
    for filter_item in filters:
        if filter_item.operator != "=":
            continue
        predicates.append(f'"{filter_item.field_name}" = ?')
        parameters.append(filter_item.value)
    if predicates:
        query += " WHERE " + " AND ".join(predicates)
    return query, parameters
