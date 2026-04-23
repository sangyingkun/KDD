from __future__ import annotations

import csv
from datetime import datetime
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.semantic.catalog import (
    CrossSourceAnchorSpec,
    DimensionSpec,
    EntitySpec,
    EvidenceSpec,
    KnowledgeContract,
    MeasureSpec,
    RelationKeyPair,
    RelationSpec,
    RoutingRuleSpec,
    SemanticCatalog,
    SourceItemSpec,
)


_SQLITE_SUFFIXES = {".sqlite", ".db"}
_MEASURE_KEYWORDS = ("amount", "price", "cost", "revenue", "sales", "total")
_TIME_KEYWORDS = ("date", "time", "at")
_GEO_NAMES = {"region", "region_name", "country", "state"}
_STATUS_NAMES = {"status"}
_CATEGORY_NAMES = {"sex", "gender", "diagnosis", "disease"}
_KNOWLEDGE_SECTION_NAMES = {
    "1. introduction": "introduction",
    "2. core entities & fields": "core_entities_fields",
    "3. metric definitions": "metric_definitions",
    "4. constraints & conventions": "constraints_conventions",
    "5. exemplar use cases": "exemplar_use_cases",
    "6. ambiguity resolution": "ambiguity_resolution",
}


def _camel_to_snake(value: str) -> str:
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)


def _normalize_identifier(value: str) -> str:
    normalized = _camel_to_snake(value).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _singularize(name: str) -> str:
    normalized = re.sub(r"\.[^.]+$", "", name.lower())
    normalized = _normalize_identifier(normalized)
    if normalized.endswith("ies"):
        return normalized[:-3] + "y"
    # Avoid turning words like "business" -> "busines" or "analysis" -> "analysi".
    if normalized.endswith(("ss", "is")):
        return normalized
    if normalized.endswith("s") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def _read_csv_header(path: Path) -> list[str]:
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            return list(header) if header else []
    except (OSError, UnicodeDecodeError):
        return []


def _read_csv_sample_rows(path: Path, max_rows: int = 20) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append({str(key): str(value) for key, value in row.items() if key is not None})
                if len(rows) >= max_rows:
                    break
            return rows
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _collect_sample_values(values: list[Any], max_values: int = 5) -> list[str]:
    samples: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or len(text) > 80:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        samples.append(text)
        if len(samples) >= max_values:
            break
    return samples


def _infer_text_data_type(values: list[Any]) -> str:
    normalized_values = [str(value).strip() for value in values if str(value).strip()]
    if not normalized_values:
        return "string"

    bool_tokens = {"true", "false", "yes", "no", "y", "n", "0", "1"}
    if all(value.lower() in bool_tokens for value in normalized_values):
        return "bool"
    if all(re.fullmatch(r"[-+]?\d+", value) for value in normalized_values):
        return "int"
    if all(re.fullmatch(r"[-+]?(?:\d+\.\d+|\d+)", value) for value in normalized_values):
        return "float"

    datetime_formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m",
        "%Y%m",
    )
    for fmt in datetime_formats:
        try:
            for value in normalized_values:
                datetime.strptime(value, fmt)
            return "datetime" if "H" in fmt else "date"
        except ValueError:
            continue
    return "string"


def _infer_value_range(values: list[Any], data_type: str) -> str | None:
    normalized_values = [str(value).strip() for value in values if str(value).strip()]
    if not normalized_values:
        return None

    if data_type == "int":
        parsed = [int(value) for value in normalized_values if re.fullmatch(r"[-+]?\d+", value)]
        if parsed:
            return f"{min(parsed)}..{max(parsed)}"
    if data_type == "float":
        parsed = [float(value) for value in normalized_values if re.fullmatch(r"[-+]?(?:\d+\.\d+|\d+)", value)]
        if parsed:
            return f"{min(parsed):g}..{max(parsed):g}"
    if data_type in {"date", "datetime"}:
        return f"{min(normalized_values)}..{max(normalized_values)}"
    return None


def _infer_format_pattern(sample_values: list[str], data_type: str) -> str | None:
    if not sample_values:
        return None
    if data_type == "bool":
        lowered = sorted({value.lower() for value in sample_values})
        return f"boolean-like: {', '.join(lowered)}"
    if data_type == "int":
        if all(re.fullmatch(r"\d{6}", value) for value in sample_values):
            return "fixed-width 6-digit integer"
        return "integer numbers"
    if data_type == "float":
        return "decimal numbers"
    if data_type == "date":
        return "date-like values"
    if data_type == "datetime":
        return "datetime-like values"
    if all(re.fullmatch(r"[A-Z]{2,5}", value) for value in sample_values):
        return "uppercase code tokens"
    if all(re.fullmatch(r"[A-Za-z0-9_-]{4,}", value) for value in sample_values):
        return "identifier-like alphanumeric tokens"
    if all("@" in value for value in sample_values):
        return "email-like strings"
    if all(re.fullmatch(r"[\d:+-]+", value) for value in sample_values):
        return "numeric-symbol text tokens"
    return "free-form text"


def _build_field_description(
    *,
    entity_name: str,
    field_name: str,
    semantic_type: str,
    source_ref: str,
) -> str:
    return (
        f"{field_name} on {entity_name} from {source_ref}; "
        f"auto-profiled as {semantic_type}."
    )


def _json_path(path_parts: list[str]) -> str:
    escaped_parts = [part.replace('"', '\\"') for part in path_parts if part]
    if not escaped_parts:
        return "$"
    return "$." + ".".join(escaped_parts)


def _source_item_retrieval_text(
    *,
    item_type: str,
    entity_name: str,
    display_name: str,
    field_ref: str,
    source_file: str,
    source_type: str,
    data_type: str,
    semantic_role: str,
    description: str,
    sample_values: list[str],
    value_range: str | None,
    format_pattern: str | None,
    aliases: list[str],
    anchor_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> str:
    metadata = metadata or {}
    return "\n".join(
        part
        for part in [
            f"item_type: {item_type}",
            f"entity: {entity_name}",
            f"display_name: {display_name}",
            f"field_ref: {field_ref}",
            f"source_file: {source_file}",
            f"source_type: {source_type}",
            f"data_type: {data_type}",
            f"semantic_role: {semantic_role}",
            f"description: {description}",
            f"sample_values: {', '.join(sample_values) if sample_values else 'none'}",
            f"value_range: {value_range or 'none'}",
            f"format_pattern: {format_pattern or 'none'}",
            f"aliases: {', '.join(aliases) if aliases else 'none'}",
            f"anchor_names: {', '.join(anchor_names) if anchor_names else 'none'}",
            f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata else 'none'}",
        ]
        if part
    )


def _build_source_item(
    *,
    item_id: str,
    item_type: str,
    entity_name: str,
    source_type: str,
    source_file: str,
    source_path: str,
    field_ref: str,
    display_name: str,
    normalized_name: str,
    data_type: str,
    semantic_role: str,
    description: str,
    sample_values: list[str] | None = None,
    value_range: str | None = None,
    format_pattern: str | None = None,
    aliases: list[str] | None = None,
    anchor_names: list[str] | None = None,
    confidence: str = "medium",
    provenance: str = "auto_schema",
    metadata: dict[str, Any] | None = None,
) -> SourceItemSpec:
    normalized_anchor_names = [
        _normalize_identifier(item)
        for item in (anchor_names or [])
        if _normalize_identifier(item)
    ]
    unique_anchor_names: list[str] = []
    seen_anchors: set[str] = set()
    for item in normalized_anchor_names:
        if item in seen_anchors:
            continue
        seen_anchors.add(item)
        unique_anchor_names.append(item)

    normalized_aliases = [
        str(item).strip()
        for item in (aliases or [])
        if str(item).strip()
    ]
    return SourceItemSpec(
        item_id=item_id,
        item_type=item_type,
        entity=entity_name,
        source_type=source_type,
        source_file=source_file,
        source_path=source_path,
        field_ref=field_ref,
        display_name=display_name,
        normalized_name=normalized_name,
        data_type=data_type,
        semantic_role=semantic_role,
        description=description,
        retrieval_text=_source_item_retrieval_text(
            item_type=item_type,
            entity_name=entity_name,
            display_name=display_name,
            field_ref=field_ref,
            source_file=source_file,
            source_type=source_type,
            data_type=data_type,
            semantic_role=semantic_role,
            description=description,
            sample_values=list(sample_values or []),
            value_range=value_range,
            format_pattern=format_pattern,
            aliases=normalized_aliases,
            anchor_names=unique_anchor_names,
            metadata=metadata,
        ),
        sample_values=list(sample_values or []),
        value_range=value_range,
        format_pattern=format_pattern,
        aliases=normalized_aliases,
        anchor_names=unique_anchor_names,
        confidence=confidence,
        provenance=provenance,
        metadata=dict(metadata or {}),
    )


def _merge_entity_key_maps(
    *,
    entity_name: str,
    primary_key_map: dict[str, str],
    candidate_key_map: dict[str, str],
    entity_primary_keys: dict[str, dict[str, str]],
    entity_candidate_keys: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    merged_primary = entity_primary_keys.setdefault(entity_name, {})
    merged_candidate = entity_candidate_keys.setdefault(entity_name, {})
    for key, value in primary_key_map.items():
        merged_primary.setdefault(key, value)
        merged_candidate.setdefault(key, value)
    for key, value in candidate_key_map.items():
        merged_candidate.setdefault(key, value)
    return list(merged_primary.values()), list(merged_candidate.values())


def _infer_dimension_spec(
    *,
    entity_name: str,
    field_ref: str,
    field_name: str,
    provenance: str,
    data_type: str = "string",
    sample_values: list[str] | None = None,
    description: str | None = None,
    value_range: str | None = None,
    format_pattern: str | None = None,
) -> DimensionSpec | None:
    normalized = _normalize_identifier(field_name)
    if normalized in _GEO_NAMES:
        return DimensionSpec(
            name=field_name,
            entity=entity_name,
            field_ref=field_ref,
            data_type=data_type,
            semantic_type="geo",
            time_grain=None,
            aliases=[],
            confidence="medium",
            provenance=provenance,
            description=description or _build_field_description(
                entity_name=entity_name,
                field_name=field_name,
                semantic_type="geo",
                source_ref=field_ref.split("::", 1)[0],
            ),
            sample_values=list(sample_values or []),
            value_range=value_range,
            format_pattern=format_pattern,
        )
    if normalized in _STATUS_NAMES:
        return DimensionSpec(
            name=field_name,
            entity=entity_name,
            field_ref=field_ref,
            data_type=data_type,
            semantic_type="status",
            time_grain=None,
            aliases=[],
            confidence="medium",
            provenance=provenance,
            description=description or _build_field_description(
                entity_name=entity_name,
                field_name=field_name,
                semantic_type="status",
                source_ref=field_ref.split("::", 1)[0],
            ),
            sample_values=list(sample_values or []),
            value_range=value_range,
            format_pattern=format_pattern,
        )
    if normalized.endswith("_id"):
        return DimensionSpec(
            name=field_name,
            entity=entity_name,
            field_ref=field_ref,
            data_type=data_type,
            semantic_type="identifier",
            time_grain=None,
            aliases=[],
            confidence="medium",
            provenance=provenance,
            description=description or _build_field_description(
                entity_name=entity_name,
                field_name=field_name,
                semantic_type="identifier",
                source_ref=field_ref.split("::", 1)[0],
            ),
            sample_values=list(sample_values or []),
            value_range=value_range,
            format_pattern=format_pattern,
        )
    if normalized.endswith(_TIME_KEYWORDS) or any(
        keyword in normalized for keyword in ("created", "updated", "signup")
    ):
        return DimensionSpec(
            name=field_name,
            entity=entity_name,
            field_ref=field_ref,
            data_type=data_type,
            semantic_type="time",
            time_grain="day",
            aliases=[],
            confidence="medium",
            provenance=provenance,
            description=description or _build_field_description(
                entity_name=entity_name,
                field_name=field_name,
                semantic_type="time",
                source_ref=field_ref.split("::", 1)[0],
            ),
            sample_values=list(sample_values or []),
            value_range=value_range,
            format_pattern=format_pattern,
        )
    if normalized in _CATEGORY_NAMES:
        return DimensionSpec(
            name=field_name,
            entity=entity_name,
            field_ref=field_ref,
            data_type=data_type,
            semantic_type="category",
            time_grain=None,
            aliases=[],
            confidence="medium",
            provenance=provenance,
            description=description or _build_field_description(
                entity_name=entity_name,
                field_name=field_name,
                semantic_type="category",
                source_ref=field_ref.split("::", 1)[0],
            ),
            sample_values=list(sample_values or []),
            value_range=value_range,
            format_pattern=format_pattern,
        )
    if data_type in {"str", "string"} and sample_values:
        return DimensionSpec(
            name=field_name,
            entity=entity_name,
            field_ref=field_ref,
            data_type=data_type,
            semantic_type="text",
            time_grain=None,
            aliases=[],
            confidence="low",
            provenance=provenance,
            description=description or _build_field_description(
                entity_name=entity_name,
                field_name=field_name,
                semantic_type="text",
                source_ref=field_ref.split("::", 1)[0],
            ),
            sample_values=list(sample_values or []),
            value_range=value_range,
            format_pattern=format_pattern,
        )
    return None


def _infer_measure_spec(
    *,
    entity_name: str,
    field_ref: str,
    field_name: str,
    provenance: str,
    data_type: str = "numeric",
    sample_values: list[str] | None = None,
    description: str | None = None,
    value_range: str | None = None,
    format_pattern: str | None = None,
) -> MeasureSpec | None:
    normalized = _normalize_identifier(field_name)
    if not any(keyword in normalized for keyword in _MEASURE_KEYWORDS):
        return None
    return MeasureSpec(
        name=field_name,
        entity=entity_name,
        field_ref=field_ref,
        default_agg="sum",
        unit="currency",
        value_type=data_type,
        constraints=[],
        confidence="medium",
        provenance=provenance,
        aliases=[],
        description=description or _build_field_description(
            entity_name=entity_name,
            field_name=field_name,
            semantic_type="measure",
            source_ref=field_ref.split("::", 1)[0],
        ),
        sample_values=list(sample_values or []),
        value_range=value_range,
        format_pattern=format_pattern,
    )


def _walk_json_paths(
    payload: Any,
    *,
    prefix: list[str],
    max_depth: int,
    leaf_paths: list[tuple[list[str], str, Any]],
    key_names: dict[str, str],
) -> None:
    if max_depth < 0:
        return

    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key)
            normalized_key = _normalize_identifier(key)
            if normalized_key.endswith("_id"):
                key_names.setdefault(normalized_key, key)
            next_prefix = [*prefix, key]
            if isinstance(value, (dict, list)):
                _walk_json_paths(
                    value,
                    prefix=next_prefix,
                    max_depth=max_depth - 1,
                    leaf_paths=leaf_paths,
                    key_names=key_names,
                )
            else:
                leaf_paths.append((next_prefix, type(value).__name__, value))
    elif isinstance(payload, list):
        scanned = 0
        for item in payload:
            if scanned >= 20:
                break
            scanned += 1
            _walk_json_paths(
                item,
                prefix=prefix,
                max_depth=max_depth,
                leaf_paths=leaf_paths,
                key_names=key_names,
            )


def _sqlite_column_profile(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    declared_type: str,
) -> tuple[list[str], str | None, str | None]:
    quoted_table = table_name.replace('"', '""')
    quoted_column = column_name.replace('"', '""')
    samples: list[str] = []
    value_range: str | None = None
    format_pattern: str | None = None
    try:
        sample_rows = conn.execute(
            f'''
            SELECT "{quoted_column}"
            FROM "{quoted_table}"
            WHERE "{quoted_column}" IS NOT NULL
            LIMIT 20
            '''
        ).fetchall()
        raw_values = [row[0] for row in sample_rows]
        samples = _collect_sample_values(raw_values)
        inferred_type = _infer_text_data_type(raw_values)
        format_pattern = _infer_format_pattern(samples, inferred_type)

        declared = str(declared_type or "").lower()
        numeric_like = any(token in declared for token in ("int", "real", "num", "float", "double", "decimal"))
        if numeric_like or inferred_type in {"int", "float"}:
            range_row = conn.execute(
                f'''
                SELECT MIN("{quoted_column}"), MAX("{quoted_column}")
                FROM "{quoted_table}"
                WHERE "{quoted_column}" IS NOT NULL
                '''
            ).fetchone()
            if range_row is not None and range_row[0] is not None and range_row[1] is not None:
                value_range = f"{range_row[0]}..{range_row[1]}"
        elif inferred_type in {"date", "datetime"} and samples:
            value_range = _infer_value_range(samples, inferred_type)
    except sqlite3.Error:
        return samples, value_range, format_pattern
    return samples, value_range, format_pattern


def _add_sqlite_catalog_entries(
    path: Path,
    *,
    rel_path: str,
    entities: list[EntitySpec],
    dimensions: list[DimensionSpec],
    measures: list[MeasureSpec],
    source_items: list[SourceItemSpec],
    entity_primary_keys: dict[str, dict[str, str]],
    entity_candidate_keys: dict[str, dict[str, str]],
) -> None:
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error:
        return

    try:
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for (table_name,) in tables:
            entity_name = _singularize(str(table_name))
            primary_key_map: dict[str, str] = {}
            candidate_key_map: dict[str, str] = {}
            try:
                columns = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            except sqlite3.Error:
                continue

            for _, column_name, declared_type, _, _, pk in columns:
                original_name = str(column_name)
                normalized_name = _normalize_identifier(original_name)
                sample_values, value_range, format_pattern = _sqlite_column_profile(
                    conn,
                    table_name=str(table_name),
                    column_name=original_name,
                    declared_type=str(declared_type or "unknown"),
                )
                if pk:
                    primary_key_map.setdefault(normalized_name, original_name)
                if normalized_name.endswith("_id"):
                    candidate_key_map.setdefault(normalized_name, original_name)

                field_ref = f"{rel_path}::{table_name}::{original_name}"
                dimension = _infer_dimension_spec(
                    entity_name=entity_name,
                    field_ref=field_ref,
                    field_name=original_name,
                    provenance="auto_schema",
                    data_type=str(declared_type or "unknown"),
                    sample_values=sample_values,
                    description=f"{original_name} column from table {table_name} in {rel_path}",
                    value_range=value_range,
                    format_pattern=format_pattern,
                )
                if dimension is not None:
                    dimensions.append(dimension)
                    source_items.append(
                        _build_source_item(
                            item_id=f"sqlite::{rel_path}::{table_name}::{original_name}::dimension",
                            item_type="sqlite_field",
                            entity_name=entity_name,
                            source_type="sqlite",
                            source_file=rel_path,
                            source_path=f"{table_name}.{original_name}",
                            field_ref=field_ref,
                            display_name=original_name,
                            normalized_name=normalized_name,
                            data_type=str(declared_type or "unknown"),
                            semantic_role=dimension.semantic_type,
                            description=dimension.description,
                            sample_values=dimension.sample_values,
                            value_range=dimension.value_range,
                            format_pattern=dimension.format_pattern,
                            aliases=dimension.aliases,
                            anchor_names=[normalized_name] if normalized_name.endswith("_id") else [],
                            confidence=dimension.confidence,
                            provenance=dimension.provenance,
                            metadata={"table_name": str(table_name), "field_kind": "dimension"},
                        )
                    )

                measure = _infer_measure_spec(
                    entity_name=entity_name,
                    field_ref=field_ref,
                    field_name=original_name,
                    provenance="auto_schema",
                    data_type=str(declared_type or "numeric"),
                    sample_values=sample_values,
                    description=f"{original_name} numeric column from table {table_name} in {rel_path}",
                    value_range=value_range,
                    format_pattern=format_pattern,
                )
                if measure is not None:
                    measures.append(measure)
                    source_items.append(
                        _build_source_item(
                            item_id=f"sqlite::{rel_path}::{table_name}::{original_name}::measure",
                            item_type="sqlite_field",
                            entity_name=entity_name,
                            source_type="sqlite",
                            source_file=rel_path,
                            source_path=f"{table_name}.{original_name}",
                            field_ref=field_ref,
                            display_name=original_name,
                            normalized_name=normalized_name,
                            data_type=str(declared_type or "numeric"),
                            semantic_role="measure",
                            description=measure.description,
                            sample_values=measure.sample_values,
                            value_range=measure.value_range,
                            format_pattern=measure.format_pattern,
                            aliases=measure.aliases,
                            anchor_names=[normalized_name] if normalized_name.endswith("_id") else [],
                            confidence=measure.confidence,
                            provenance=measure.provenance,
                            metadata={"table_name": str(table_name), "field_kind": "measure", "default_agg": measure.default_agg},
                        )
                    )

            primary_keys, candidate_keys = _merge_entity_key_maps(
                entity_name=entity_name,
                primary_key_map=primary_key_map,
                candidate_key_map=candidate_key_map,
                entity_primary_keys=entity_primary_keys,
                entity_candidate_keys=entity_candidate_keys,
            )
            entities.append(
                EntitySpec(
                    name=entity_name,
                    aliases=[],
                    description=f"Auto-extracted from {rel_path}::{table_name}",
                    sources=[rel_path],
                    primary_keys=primary_keys,
                    candidate_keys=candidate_keys,
                    confidence="medium",
                    provenance="auto_schema",
                )
            )
    finally:
        conn.close()


def _extract_document_evidence(rel_path: str, text: str) -> list[EvidenceSpec]:
    evidences: list[EvidenceSpec] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        line_lc = line.lower()
        is_metric_like = "count" in line_lc or "rate" in line_lc
        is_field_definition = line.startswith("- **") and ": " in line
        is_usage_constraint = "always use both fields" in line_lc or "always use" in line_lc
        if not (is_metric_like or is_field_definition or is_usage_constraint):
            continue
        evidences.append(
            EvidenceSpec(
                id=f"ev_{len(evidences) + 1}",
                claim=line,
                source_type="document",
                source_file=rel_path,
                location_hint="document line",
                snippet=line[:240],
                confidence="medium",
                provenance="auto_doc",
            )
        )
    if evidences:
        return evidences

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text)
        if item.strip()
    ]
    if not sentences and text.strip():
        sentences = [text.strip()]

    for sentence in sentences:
        sentence_lc = sentence.lower()
        if "count" in sentence_lc or "rate" in sentence_lc:
            evidences.append(
                EvidenceSpec(
                    id=f"ev_{len(evidences) + 1}",
                    claim=sentence,
                    source_type="document",
                    source_file=rel_path,
                    location_hint="document sentence",
                    snippet=sentence[:240],
                    confidence="medium",
                    provenance="auto_doc",
                )
            )
    return evidences


def _parse_knowledge_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {value: [] for value in _KNOWLEDGE_SECTION_NAMES.values()}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading_match = re.match(r"^##\s+(.*)$", line.strip())
        if heading_match is not None:
            normalized_heading = heading_match.group(1).strip().lower()
            current_section = _KNOWLEDGE_SECTION_NAMES.get(normalized_heading)
            continue
        if current_section is not None:
            sections[current_section].append(line)
    return {
        key: "\n".join(line for line in values if line.strip()).strip()
        for key, values in sections.items()
        if any(line.strip() for line in values)
    }


def _extract_knowledge_contract(text: str) -> KnowledgeContract:
    sections = _parse_knowledge_sections(text)
    entity_field_rules: list[dict[str, Any]] = []
    metric_rules: list[dict[str, Any]] = []
    constraint_rules: list[dict[str, Any]] = []
    example_rules: list[dict[str, Any]] = []
    ambiguity_rules: list[dict[str, Any]] = []
    output_constraints: list[dict[str, Any]] = []

    for line in sections.get("core_entities_fields", "").splitlines():
        field_match = re.match(r"^- \*\*([^*]+)\*\*:\s*(.+)$", line.strip())
        if field_match is None:
            continue
        field_name = field_match.group(1).strip()
        description = field_match.group(2).strip()
        entity_field_rules.append({"field": field_name, "description": description})
        if "full name" in description.lower():
            fields = [
                _normalize_identifier(part)
                for part in field_name.split(",")
                if _normalize_identifier(part)
            ]
            if len(fields) >= 2:
                output_constraints.append({"concept": "full name", "fields": fields})

    for line in sections.get("metric_definitions", "").splitlines():
        metric_match = re.match(r"^- \*\*([^*]+)\*\*:\s*(.+)$", line.strip())
        if metric_match is None:
            continue
        metric_rules.append(
            {
                "metric_name": metric_match.group(1).strip(),
                "definition": metric_match.group(2).strip(),
            }
        )

    for line in sections.get("constraints_conventions", "").splitlines():
        constraint_match = re.match(r"^- \*\*([^*]+)\*\*:\s*(.+)$", line.strip())
        if constraint_match is None:
            continue
        field_name = constraint_match.group(1).strip()
        raw_text = constraint_match.group(2).strip()
        allowed_values = re.findall(r"'([^']+)'", raw_text)
        normalized_field = field_name
        if "Admission" in raw_text and "Admission" not in field_name:
            normalized_field = "Admission"
        elif field_name.lower().startswith("admission"):
            normalized_field = "Admission"
        constraint_rules.append(
            {
                "field": normalized_field,
                "allowed_values": allowed_values,
                "raw_text": raw_text,
            }
        )

    for line in sections.get("exemplar_use_cases", "").splitlines():
        example_match = re.match(r"^- \*\*([^*]+)\*\*:\s*(.+)$", line.strip())
        if example_match is None:
            continue
        example_rules.append(
            {
                "label": example_match.group(1).strip(),
                "content": example_match.group(2).strip(),
            }
        )

    for line in sections.get("ambiguity_resolution", "").splitlines():
        ambiguity_match = re.match(r"^- \*\*([^*]+)\*\*:\s*(.+)$", line.strip())
        if ambiguity_match is None:
            continue
        ambiguity_rules.append(
            {
                "term": ambiguity_match.group(1).strip(),
                "rule": ambiguity_match.group(2).strip(),
            }
        )

    return KnowledgeContract(
        sections=sections,
        entity_field_rules=entity_field_rules,
        metric_rules=metric_rules,
        constraint_rules=constraint_rules,
        example_rules=example_rules,
        ambiguity_rules=ambiguity_rules,
        output_constraints=output_constraints,
    )


def _build_knowledge_source_items(
    *,
    rel_path: str,
    knowledge_contract: KnowledgeContract,
) -> tuple[list[SourceItemSpec], list[RoutingRuleSpec]]:
    source_items: list[SourceItemSpec] = []
    routing_rules: list[RoutingRuleSpec] = []
    item_index = 0
    rule_index = 0

    for section_name, text in knowledge_contract.sections.items():
        if not text.strip():
            continue
        item_index += 1
        normalized_section = _normalize_identifier(section_name) or "knowledge_rule"
        source_items.append(
            _build_source_item(
                item_id=f"knowledge::{item_index}",
                item_type="knowledge_rule",
                entity_name="knowledge",
                source_type="document",
                source_file=rel_path,
                source_path=section_name,
                field_ref=f"{rel_path}::{section_name}",
                display_name=section_name,
                normalized_name=normalized_section,
                data_type="text",
                semantic_role="rule",
                description=text.strip(),
                sample_values=[],
                aliases=[section_name],
                anchor_names=[],
                confidence="medium",
                provenance="auto_doc",
                metadata={"section": section_name},
            )
        )

        lowered_text = text.lower()
        source_mentions = sorted({
            source_name
            for source_name in ("json", "db", "sqlite", "csv", "document")
            if source_name in lowered_text
        })
        anchor_names = sorted({
            _normalize_identifier(match)
            for match in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*_id\b", text)
            if _normalize_identifier(match)
        })
        if source_mentions:
            rule_index += 1
            routing_rules.append(
                RoutingRuleSpec(
                    rule_id=f"routing::{rule_index}",
                    rule_type="source_preference",
                    description=text.strip(),
                    source_file=rel_path,
                    condition=f"knowledge section {section_name}",
                    target_sources=source_mentions,
                    anchor_names=anchor_names,
                    confidence="medium",
                    provenance="auto_doc",
                    metadata={"section": section_name},
                )
            )

    return source_items, routing_rules


def _build_cross_source_anchors(source_items: list[SourceItemSpec]) -> list[CrossSourceAnchorSpec]:
    members_by_anchor: dict[str, list[SourceItemSpec]] = defaultdict(list)
    for item in source_items:
        for anchor_name in item.anchor_names:
            members_by_anchor[anchor_name].append(item)

    anchors: list[CrossSourceAnchorSpec] = []
    for anchor_name, members in sorted(members_by_anchor.items()):
        source_files = sorted({member.source_file for member in members if member.source_file})
        if len(source_files) < 2:
            continue
        anchors.append(
            CrossSourceAnchorSpec(
                anchor_name=anchor_name,
                members=[member.item_id for member in members],
                source_files=source_files,
                description=f"Cross-source anchor derived from shared key {anchor_name}",
                confidence="medium",
                provenance="auto_schema",
                metadata={
                    "member_refs": [member.field_ref for member in members],
                    "source_types": sorted({member.source_type for member in members}),
                },
            )
        )
    return anchors


def build_base_semantic_catalog(task: PublicTask) -> SemanticCatalog:
    entities: list[EntitySpec] = []
    relations: list[RelationSpec] = []
    dimensions: list[DimensionSpec] = []
    measures: list[MeasureSpec] = []
    evidence: list[EvidenceSpec] = []
    source_items: list[SourceItemSpec] = []
    routing_rules: list[RoutingRuleSpec] = []
    knowledge_contract = KnowledgeContract()

    # Keep small, local structures for deterministic relation inference without relying on path ordering.
    # Store both original column names and a lowercased lookup for case-insensitive key detection.
    entity_primary_keys: dict[str, dict[str, str]] = {}
    entity_candidate_keys: dict[str, dict[str, str]] = {}

    try:
        context_entries = sorted(
            (path for path in task.context_dir.rglob("*") if path.is_file()),
            key=lambda item: item.as_posix(),
        )
    except OSError:
        return SemanticCatalog()

    for path in context_entries:
        rel_path = path.relative_to(task.context_dir).as_posix()
        entity_name = _singularize(path.name)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            header = _read_csv_header(path)
            sample_rows = _read_csv_sample_rows(path)
            sample_values_by_column: dict[str, list[str]] = {}
            raw_values_by_column: dict[str, list[str]] = {}
            for column in header:
                raw_values = [row.get(column, "") for row in sample_rows]
                raw_values_by_column[column] = raw_values
                sample_values_by_column[column] = _collect_sample_values(raw_values)

            # Heuristic: `{entity}_id` and similar `{entity}_*_id` columns are treated as primary keys.
            primary_key_map: dict[str, str] = {}
            candidate_key_map: dict[str, str] = {}
            entity_prefix = f"{entity_name}_"
            entity_pk = f"{entity_name}_id"
            for column in header:
                normalized_column = _normalize_identifier(column)
                if not normalized_column.endswith("_id"):
                    continue
                candidate_key_map.setdefault(normalized_column, column)
                if normalized_column == entity_pk or normalized_column.startswith(entity_prefix):
                    primary_key_map.setdefault(normalized_column, column)

            primary_keys, candidate_keys = _merge_entity_key_maps(
                entity_name=entity_name,
                primary_key_map=primary_key_map,
                candidate_key_map=candidate_key_map,
                entity_primary_keys=entity_primary_keys,
                entity_candidate_keys=entity_candidate_keys,
            )

            entities.append(
                EntitySpec(
                    name=entity_name,
                    aliases=[],
                    description=f"Auto-extracted from {rel_path}",
                    sources=[rel_path],
                    primary_keys=primary_keys,
                    candidate_keys=candidate_keys,
                    confidence="medium",
                    provenance="auto_schema",
                )
            )

            for column in header:
                field_ref = f"{rel_path}::{column}"
                raw_values = raw_values_by_column.get(column, [])
                inferred_data_type = _infer_text_data_type(raw_values)
                value_range = _infer_value_range(raw_values, inferred_data_type)
                format_pattern = _infer_format_pattern(
                    sample_values_by_column.get(column, []),
                    inferred_data_type,
                )
                dimension = _infer_dimension_spec(
                    entity_name=entity_name,
                    field_ref=field_ref,
                    field_name=column,
                    provenance="auto_schema",
                    data_type=inferred_data_type,
                    sample_values=sample_values_by_column.get(column, []),
                    description=f"{column} column from {rel_path}",
                    value_range=value_range,
                    format_pattern=format_pattern,
                )
                if dimension is not None:
                    dimensions.append(dimension)
                    source_items.append(
                        _build_source_item(
                            item_id=f"csv::{rel_path}::{column}::dimension",
                            item_type="csv_field",
                            entity_name=entity_name,
                            source_type="csv",
                            source_file=rel_path,
                            source_path=column,
                            field_ref=field_ref,
                            display_name=column,
                            normalized_name=_normalize_identifier(column),
                            data_type=inferred_data_type,
                            semantic_role=dimension.semantic_type,
                            description=dimension.description,
                            sample_values=dimension.sample_values,
                            value_range=dimension.value_range,
                            format_pattern=dimension.format_pattern,
                            aliases=dimension.aliases,
                            anchor_names=[_normalize_identifier(column)] if _normalize_identifier(column).endswith("_id") else [],
                            confidence=dimension.confidence,
                            provenance=dimension.provenance,
                            metadata={"field_kind": "dimension"},
                        )
                    )
                measure = _infer_measure_spec(
                    entity_name=entity_name,
                    field_ref=field_ref,
                    field_name=column,
                    provenance="auto_schema",
                    data_type=inferred_data_type,
                    sample_values=sample_values_by_column.get(column, []),
                    description=f"{column} measure candidate from {rel_path}",
                    value_range=value_range,
                    format_pattern=format_pattern,
                )
                if measure is not None:
                    measures.append(measure)
                    source_items.append(
                        _build_source_item(
                            item_id=f"csv::{rel_path}::{column}::measure",
                            item_type="csv_field",
                            entity_name=entity_name,
                            source_type="csv",
                            source_file=rel_path,
                            source_path=column,
                            field_ref=field_ref,
                            display_name=column,
                            normalized_name=_normalize_identifier(column),
                            data_type=inferred_data_type,
                            semantic_role="measure",
                            description=measure.description,
                            sample_values=measure.sample_values,
                            value_range=measure.value_range,
                            format_pattern=measure.format_pattern,
                            aliases=measure.aliases,
                            anchor_names=[_normalize_identifier(column)] if _normalize_identifier(column).endswith("_id") else [],
                            confidence=measure.confidence,
                            provenance=measure.provenance,
                            metadata={"field_kind": "measure", "default_agg": measure.default_agg},
                        )
                    )

        elif suffix in {".md", ".txt"}:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if rel_path == "knowledge.md":
                knowledge_contract = _extract_knowledge_contract(text)
                knowledge_source_items, extracted_routing_rules = _build_knowledge_source_items(
                    rel_path=rel_path,
                    knowledge_contract=knowledge_contract,
                )
                source_items.extend(knowledge_source_items)
                routing_rules.extend(extracted_routing_rules)
            for item in _extract_document_evidence(rel_path, text):
                evidence.append(
                    EvidenceSpec(
                        id=f"ev_{len(evidence) + 1}",
                        claim=item.claim,
                        source_type=item.source_type,
                        source_file=item.source_file,
                        location_hint=item.location_hint,
                        snippet=item.snippet,
                        confidence=item.confidence,
                        provenance=item.provenance,
                    )
                )

        elif suffix == ".json":
            try:
                payload = json.loads(path.read_text(errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue

            candidate_key_map: dict[str, str] = {}
            leaf_paths: list[tuple[list[str], str, Any]] = []
            _walk_json_paths(
                payload,
                prefix=[],
                max_depth=3,
                leaf_paths=leaf_paths,
                key_names=candidate_key_map,
            )
            sample_values_by_path: dict[tuple[str, ...], list[Any]] = defaultdict(list)
            for path_parts, _, sample_value in leaf_paths:
                sample_values_by_path[tuple(path_parts)].append(sample_value)

            if candidate_key_map or isinstance(payload, (dict, list)):
                _, candidate_keys = _merge_entity_key_maps(
                    entity_name=entity_name,
                    primary_key_map={},
                    candidate_key_map=candidate_key_map,
                    entity_primary_keys=entity_primary_keys,
                    entity_candidate_keys=entity_candidate_keys,
                )
                entities.append(
                    EntitySpec(
                        name=entity_name,
                        aliases=[],
                        description=f"JSON object extracted from {rel_path}",
                        sources=[rel_path],
                        primary_keys=[],
                        candidate_keys=candidate_keys,
                        confidence="low",
                        provenance="auto_values",
                    )
                )
                for path_parts, value_type, _ in leaf_paths:
                    if not path_parts:
                        continue
                    field_name = path_parts[-1]
                    normalized_path = _json_path(path_parts)
                    field_ref = f"{rel_path}::{normalized_path}"
                    raw_values = sample_values_by_path.get(tuple(path_parts), [])
                    sample_values = _collect_sample_values(raw_values)
                    inferred_data_type = (
                        value_type
                        if value_type not in {"str", "string"}
                        else _infer_text_data_type(raw_values)
                    )
                    value_range = _infer_value_range(raw_values, inferred_data_type)
                    format_pattern = _infer_format_pattern(sample_values, inferred_data_type)
                    dimension = _infer_dimension_spec(
                        entity_name=entity_name,
                        field_ref=field_ref,
                        field_name=field_name,
                        provenance="auto_values",
                        data_type=inferred_data_type,
                        sample_values=sample_values,
                        description=f"{field_name} JSON field from {rel_path}",
                        value_range=value_range,
                        format_pattern=format_pattern,
                    )
                    if dimension is not None:
                        dimensions.append(dimension)
                        source_items.append(
                            _build_source_item(
                                item_id=f"json::{rel_path}::{normalized_path}::dimension",
                                item_type="json_path_field",
                                entity_name=entity_name,
                                source_type="json",
                                source_file=rel_path,
                                source_path=normalized_path,
                                field_ref=field_ref,
                                display_name=field_name,
                                normalized_name=_normalize_identifier(field_name),
                                data_type=inferred_data_type,
                                semantic_role=dimension.semantic_type,
                                description=dimension.description,
                                sample_values=dimension.sample_values,
                                value_range=dimension.value_range,
                                format_pattern=dimension.format_pattern,
                                aliases=[normalized_path],
                                anchor_names=[_normalize_identifier(field_name)] if _normalize_identifier(field_name).endswith("_id") else [],
                                confidence=dimension.confidence,
                                provenance=dimension.provenance,
                                metadata={"json_path": normalized_path, "field_kind": "dimension"},
                            )
                        )
                    measure = _infer_measure_spec(
                        entity_name=entity_name,
                        field_ref=field_ref,
                        field_name=field_name,
                        provenance="auto_values",
                        data_type=inferred_data_type,
                        sample_values=sample_values,
                        description=f"{field_name} JSON measure candidate from {rel_path}",
                        value_range=value_range,
                        format_pattern=format_pattern,
                    )
                    if measure is not None:
                        measures.append(measure)
                        source_items.append(
                            _build_source_item(
                                item_id=f"json::{rel_path}::{normalized_path}::measure",
                                item_type="json_path_field",
                                entity_name=entity_name,
                                source_type="json",
                                source_file=rel_path,
                                source_path=normalized_path,
                                field_ref=field_ref,
                                display_name=field_name,
                                normalized_name=_normalize_identifier(field_name),
                                data_type=inferred_data_type,
                                semantic_role="measure",
                                description=measure.description,
                                sample_values=measure.sample_values,
                                value_range=measure.value_range,
                                format_pattern=measure.format_pattern,
                                aliases=[normalized_path],
                                anchor_names=[_normalize_identifier(field_name)] if _normalize_identifier(field_name).endswith("_id") else [],
                                confidence=measure.confidence,
                                provenance=measure.provenance,
                                metadata={"json_path": normalized_path, "field_kind": "measure", "default_agg": measure.default_agg},
                            )
                        )

        elif suffix in _SQLITE_SUFFIXES:
            _add_sqlite_catalog_entries(
                path,
                rel_path=rel_path,
                entities=entities,
                dimensions=dimensions,
                measures=measures,
                source_items=source_items,
                entity_primary_keys=entity_primary_keys,
                entity_candidate_keys=entity_candidate_keys,
            )

    # Infer relations from shared foreign-key naming conventions: e.g. orders.customer_id -> customers.customer_id.
    entity_names = {spec.name for spec in entities}
    seen_relations: set[tuple[str, str, str]] = set()
    for left_entity in sorted(entity_names):
        left_keys = entity_candidate_keys.get(left_entity, {})
        if not left_keys:
            continue
        for right_entity in sorted(entity_names):
            if left_entity == right_entity:
                continue
            right_primary_map = entity_primary_keys.get(right_entity, {})
            right_candidate_map = entity_candidate_keys.get(right_entity, {})

            key_candidates = [f"{right_entity}_id"]
            key_candidates.extend(_normalize_identifier(value) for value in right_primary_map.values())
            left_field = None
            shared_key_lc = ""
            for candidate in key_candidates:
                if candidate in left_keys:
                    left_field = left_keys[candidate]
                    shared_key_lc = candidate
                    break
            if not left_field:
                continue

            signature = (left_entity, right_entity, shared_key_lc)
            if signature in seen_relations:
                continue
            seen_relations.add(signature)

            right_field = (
                right_primary_map.get(shared_key_lc)
                or right_candidate_map.get(shared_key_lc)
                or left_field
            )
            cardinality = "many_to_one" if shared_key_lc in right_primary_map else "unknown"
            relations.append(
                RelationSpec(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    join_keys=[
                        RelationKeyPair(left_field=left_field, right_field=right_field)
                    ],
                    cardinality=cardinality,
                    description=f"Auto-inferred relation from shared key {shared_key_lc}",
                    confidence="medium",
                    provenance="auto_schema",
                )
            )

    cross_source_anchors = _build_cross_source_anchors(source_items)

    return SemanticCatalog(
        entities=entities,
        relations=relations,
        dimensions=dimensions,
        measures=measures,
        metrics=[],
        evidence=evidence,
        knowledge_contract=knowledge_contract,
        source_items=source_items,
        cross_source_anchors=cross_source_anchors,
        routing_rules=routing_rules,
    )
