from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.semantic.catalog import SemanticCatalog
from data_agent_baseline.semantic.planner import (
    _extract_value_links,
    _find_join_path,
    _infer_expected_output_columns,
)


def validate_answer_semantics(
    catalog: SemanticCatalog,
    *,
    question: str,
    columns: list[str],
    rows: list[list[Any]],
    derivation_summary: str | None = None,
    used_entities: list[str] | None = None,
    used_measures: list[str] | None = None,
    used_metrics: list[str] | None = None,
    used_relations: list[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    lower_question = question.lower()
    derivation = (derivation_summary or "").lower()
    normalized_derivation = re.sub(r"\s+", " ", derivation)
    requested_metrics = set(used_metrics or [])

    if "total" in lower_question and len(columns) > 1:
        errors.append("Output grain appears too detailed for a total-only question.")
    else:
        checks.append("Output grain is plausible for the question.")

    expected_output_columns = _infer_expected_output_columns(catalog, question)
    normalized_columns = [re.sub(r"\s+", "_", column.strip().lower()) for column in columns]
    if expected_output_columns and normalized_columns != expected_output_columns:
        errors.append(
            "Output columns do not match the knowledge-derived semantic format: expected "
            + ", ".join(expected_output_columns)
        )
    else:
        checks.append("Output columns are plausible for the question.")

    if any(
        relation.cardinality in {"unknown", "many_to_many"} for relation in catalog.relations
    ) and ("join" in derivation or "joined" in derivation):
        warnings.append("Join path includes a relation with unknown or unsafe cardinality.")
    else:
        checks.append("No unsafe join cardinality was detected from the derivation summary.")

    relevant_metrics = [
        metric
        for metric in catalog.metrics
        if metric.name in requested_metrics or metric.name.lower() in lower_question
    ]
    if not relevant_metrics:
        relevant_metrics = [metric for metric in catalog.metrics if metric.filters and "rate" in lower_question]

    missing_filters = []
    for metric in relevant_metrics:
        for field_name, expected_value in metric.filters.items():
            if expected_value.lower() not in normalized_derivation:
                missing_filters.append(f"{field_name}={expected_value}")
    if missing_filters:
        warnings.append(
            "Derived metric may be missing required filter conditions from the semantic catalog: "
            + ", ".join(sorted(missing_filters))
        )
    else:
        checks.append("No obvious required-filter omission was detected.")

    value_links = _extract_value_links(catalog, question)
    missing_value_filters = [
        f"{item['dimension']}={item['value']}"
        for item in value_links
        if item["value"].lower() not in normalized_derivation
    ]
    if missing_value_filters:
        warnings.append(
            "Derivation summary may be missing question-implied filter values linked from semantic samples: "
            + ", ".join(sorted(missing_value_filters))
        )
    else:
        checks.append("No obvious linked filter-value omission was detected.")

    knowledge_convention_gaps: list[str] = []
    for rule in catalog.knowledge_contract.constraint_rules:
        field_name = str(rule.get("field", "")).strip()
        allowed_values = [str(value).lower() for value in rule.get("allowed_values", []) if str(value)]
        if field_name.lower() != "admission" or not {"+", "-"}.issubset(set(allowed_values)):
            continue
        if any(token in lower_question for token in ("inpatient", "outpatient", "admission")) and not any(
            value in normalized_derivation for value in ("+", "-")
        ):
            knowledge_convention_gaps.append("Admission should use '+' and '-' values from the knowledge guide.")
    if knowledge_convention_gaps:
        warnings.append(
            "Derivation may ignore knowledge-defined value conventions: "
            + "; ".join(knowledge_convention_gaps)
        )
    else:
        checks.append("No obvious knowledge-defined value convention omission was detected.")

    if any(keyword in lower_question for keyword in ("according to", "policy", "definition")) and not any(
        keyword in normalized_derivation for keyword in ("policy", "document", "doc", "read_doc", "definition")
    ):
        warnings.append(
            "Question likely requires document evidence or policy confirmation, but the derivation summary does not mention it."
        )
    else:
        checks.append("No obvious missing document-evidence step was detected.")

    if len(rows) == 0:
        warnings.append("Answer is empty; verify that the question expects no rows.")
    else:
        checks.append("Answer contains at least one row.")

    if used_entities and len(used_entities) > 1:
        anchor_entity = used_entities[0]
        recommended_join_path: list[dict[str, str]] = []
        for entity_name in used_entities[1:]:
            for edge in _find_join_path(catalog, anchor_entity=anchor_entity, target_entity=entity_name):
                if edge not in recommended_join_path:
                    recommended_join_path.append(edge)
        recommended_relation_names = {
            f"{edge['left_entity']}->{edge['right_entity']}" for edge in recommended_join_path
        }
        used_relation_names = set(used_relations or [])
        if recommended_relation_names and used_relation_names and not (recommended_relation_names & used_relation_names):
            warnings.append(
                "Used relations do not overlap with the catalog's recommended join path; review for join drift or fan-out risk."
            )
        else:
            checks.append("No obvious join-path drift was detected.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "confidence": "medium" if not errors else "low",
        "recommended_next_actions": (
            ["Review joins and metric filters before submission."] if errors or warnings else []
        ),
    }
