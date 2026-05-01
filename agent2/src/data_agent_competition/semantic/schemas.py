from __future__ import annotations

from typing import Any


def filter_operator_enum() -> tuple[str, ...]:
    return ("=", "!=", ">", "<", ">=", "<=", "in", "contains")


def join_type_enum() -> tuple[str, ...]:
    return ("inner", "left", "right", "outer")


def aggregation_function_enum() -> tuple[str, ...]:
    return ("identity", "sum", "count", "avg", "mean", "min", "max")


def ordering_direction_enum() -> tuple[str, ...]:
    return ("asc", "desc")


def verification_stage_enum() -> tuple[str, ...]:
    return ("logical_plan", "graph", "routing", "execution", "answer")


def verification_severity_enum() -> tuple[str, ...]:
    return ("error", "warning")


def grounded_terms_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "grounded_terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "grounding_type": {"type": "string"},
                        "source_id": {"type": ["string", "null"]},
                        "field_name": {"type": ["string", "null"]},
                        "resolved_value": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "graph_node_id": {"type": ["string", "null"]},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["term", "grounding_type", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["grounded_terms"],
        "additionalProperties": False,
    }


def logical_plan_override_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer_columns": {
                "type": "array",
                "items": {"type": "string"},
            },
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "field_name": {"type": "string"},
                        "operator": {
                            "type": "string",
                            "enum": list(filter_operator_enum()),
                        },
                        "value": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["source_id", "field_name", "operator", "value"],
                    "additionalProperties": False,
                },
            },
            "joins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "left_source_id": {"type": "string"},
                        "left_field": {"type": "string"},
                        "right_source_id": {"type": "string"},
                        "right_field": {"type": "string"},
                        "join_type": {
                            "type": "string",
                            "enum": list(join_type_enum()),
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["left_source_id", "left_field", "right_source_id", "right_field"],
                    "additionalProperties": False,
                },
            },
            "aggregations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "field_name": {"type": "string"},
                        "function": {
                            "type": "string",
                            "enum": list(aggregation_function_enum()),
                        },
                        "alias": {"type": ["string", "null"]},
                    },
                    "required": ["source_id", "field_name", "function"],
                    "additionalProperties": False,
                },
            },
            "orderings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "field_name": {"type": "string"},
                        "direction": {"type": "string", "enum": list(ordering_direction_enum())},
                    },
                    "required": ["source_id", "field_name"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer_columns", "filters", "joins", "aggregations", "orderings"],
        "additionalProperties": False,
    }


def verification_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "ok": {"type": "boolean"},
            "signature": {"type": ["string", "null"]},
            "summary": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "stage": {"type": "string", "enum": list(verification_stage_enum())},
                        "severity": {"type": "string", "enum": list(verification_severity_enum())},
                        "metadata": {
                            "type": ["object", "null"],
                            "additionalProperties": True,
                        },
                    },
                    "required": ["code", "message", "stage", "severity"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["schema_version", "ok", "signature", "summary", "issues"],
        "additionalProperties": False,
    }
