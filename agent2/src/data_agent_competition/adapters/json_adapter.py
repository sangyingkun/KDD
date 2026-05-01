from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class JsonPreview:
    root_type: str
    field_names: tuple[str, ...]
    sample_records: tuple[dict[str, Any], ...]


def inspect_json_file(path: Path, *, sample_limit: int = 5) -> JsonPreview:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
        field_names = _union_fields(records)
        return JsonPreview(
            root_type="list",
            field_names=field_names,
            sample_records=tuple(records[:sample_limit]),
        )
    if isinstance(payload, dict):
        records = _records_from_mapping(payload)
        return JsonPreview(
            root_type="dict",
            field_names=_union_fields(records),
            sample_records=tuple(records[:sample_limit]),
        )
    return JsonPreview(root_type=type(payload).__name__, field_names=(), sample_records=())


def _records_from_mapping(payload: dict[str, Any]) -> list[dict[str, Any]]:
    list_values = [value for value in payload.values() if isinstance(value, list)]
    for value in list_values:
        if value and all(isinstance(item, dict) for item in value):
            return [dict(item) for item in value]
    return [payload]


def _union_fields(records: list[dict[str, Any]]) -> tuple[str, ...]:
    field_names: set[str] = set()
    for record in records:
        field_names.update(str(key) for key in record.keys())
    return tuple(sorted(field_names))
