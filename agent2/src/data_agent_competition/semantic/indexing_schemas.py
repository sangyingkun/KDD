from __future__ import annotations

from typing import Any


def business_relation_type_enum() -> tuple[str, ...]:
    return (
        "metric_depends_on_field",
        "metric_scoped_to_source",
        "constraint_applies_to_field",
        "constraint_applies_to_source",
        "ambiguity_prefers_field",
        "business_alias_of_field",
        "use_case_targets_source",
        "use_case_requires_metric",
        "use_case_requires_constraint",
        "value_concept_binds_field",
        "doc_enrichment_matches_field",
    )


def knowledge_relation_extraction_schema() -> dict[str, Any]:
    relation_item = {
        "type": "object",
        "properties": {
            "relation_type": {
                "type": "string",
                "enum": list(business_relation_type_enum()),
            },
            "fact_id": {"type": "string"},
            "source_fact_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "from_node_label": {"type": "string"},
            "from_node_type": {"type": "string"},
            "to_source_id": {"type": ["string", "null"]},
            "to_field_name": {"type": ["string", "null"]},
            "to_node_label": {"type": ["string", "null"]},
            "to_node_type": {"type": ["string", "null"]},
            "canonical_relation_text": {"type": "string"},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "relation_type",
            "fact_id",
            "source_fact_ids",
            "from_node_label",
            "from_node_type",
            "canonical_relation_text",
            "rationale",
            "confidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "complete": {"type": "boolean"},
            "relations": {
                "type": "array",
                "items": relation_item,
            },
        },
        "required": ["schema_version", "complete", "relations"],
        "additionalProperties": False,
    }

