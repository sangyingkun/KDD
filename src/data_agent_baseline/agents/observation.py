from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_TABULAR_ACTIONS = {"execute_context_sql", "read_csv"}
_ERROR_ACTIONS = {"execute_context_sql", "execute_python"}
_MAX_TABLE_HEAD_ROWS = 3
_MAX_LIST_ITEMS = 8
_MAX_TEXT_PREVIEW_CHARS = 800


@dataclass(frozen=True, slots=True)
class RuntimeFeedback:
    signature: str
    severity: str
    summary: str
    prompt_lines: list[str]
    should_replan: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "severity": self.severity,
            "summary": self.summary,
            "prompt_lines": list(self.prompt_lines),
            "should_replan": self.should_replan,
        }


def _safe_rows(content: dict[str, Any]) -> list[list[Any]]:
    rows = content.get("rows", [])
    if not isinstance(rows, list):
        return []
    normalized: list[list[Any]] = []
    for row in rows:
        if isinstance(row, list):
            normalized.append(list(row))
    return normalized


def _trim_text(value: str, *, limit: int = _MAX_TEXT_PREVIEW_CHARS) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _summarize_tabular_content(action: str, content: dict[str, Any]) -> tuple[dict[str, Any], RuntimeFeedback | None]:
    columns = [str(item) for item in content.get("columns", []) if item is not None]
    rows = _safe_rows(content)
    row_count = int(content.get("row_count", len(rows)))
    truncated = bool(content.get("truncated", False))
    preview_rows = rows[:_MAX_TABLE_HEAD_ROWS]
    summary = {
        "path": content.get("path"),
        "columns": columns,
        "row_count": row_count,
        "column_count": len(columns),
        "truncated": truncated or row_count > len(preview_rows),
        "head_rows": preview_rows,
        "summary": f"[Execution Success] Shape: ({row_count}, {len(columns)}). Head({_MAX_TABLE_HEAD_ROWS}): {preview_rows}",
    }

    feedback: RuntimeFeedback | None = None
    if row_count == 0 and columns:
        feedback = RuntimeFeedback(
            signature="empty_result",
            severity="warning",
            summary="Query returned zero rows.",
            prompt_lines=[
                "The last query returned no rows.",
                "Before retrying, check whether the filter value format, casing, or date encoding is wrong.",
                "If the query used a join, re-check whether the join path or join direction is correct.",
            ],
            should_replan=True,
        )
    elif preview_rows and all(all(cell in {None, "", "None"} for cell in row) for row in preview_rows):
        feedback = RuntimeFeedback(
            signature="all_null_result",
            severity="warning",
            summary="Query returned rows, but the visible values are all null-like.",
            prompt_lines=[
                "The last query returned rows but the projected values are all null-like.",
                "Do not patch this by blindly adding IS NOT NULL.",
                "Re-check whether the selected field comes from the correct table or whether the join is null-producing.",
            ],
            should_replan=True,
        )
    elif row_count > 5 or truncated:
        feedback = RuntimeFeedback(
            signature="oversized_result",
            severity="info",
            summary=f"Query returned a large intermediate result with {row_count} rows.",
            prompt_lines=[
                "The last query produced a large intermediate result.",
                "Focus on shape, key columns, and whether aggregation, DISTINCT, ranking, or an extra filter is still needed.",
            ],
        )

    return summary, feedback


def _shape_mismatch_feedback(
    *,
    action: str,
    content: dict[str, Any],
    plan_snapshot: dict[str, Any] | None,
) -> RuntimeFeedback | None:
    if action not in _TABULAR_ACTIONS or not plan_snapshot:
        return None

    rows = _safe_rows(content)
    row_count = int(content.get("row_count", len(rows)))
    columns = [str(item) for item in content.get("columns", []) if item is not None]
    column_count = len(columns)

    expected_columns = [
        str(item)
        for item in plan_snapshot.get("expected_output_columns", [])
        if str(item).strip()
    ]
    required_measures = [
        str(item)
        for item in plan_snapshot.get("required_measures", [])
        if str(item).strip()
    ]
    answer_shape_hint = "table"
    if expected_columns:
        if len(expected_columns) == 1:
            answer_shape_hint = "scalar" if required_measures else "list_single_col"
        else:
            answer_shape_hint = "single_row_multi_col" if row_count <= 1 else "table"
    elif required_measures and not plan_snapshot.get("required_dimensions"):
        answer_shape_hint = "scalar"

    if answer_shape_hint == "scalar" and (row_count > 1 or column_count != 1):
        return RuntimeFeedback(
            signature="shape_mismatch",
            severity="warning",
            summary=f"Expected a scalar-like result, but observed shape ({row_count}, {column_count}).",
            prompt_lines=[
                "The observed result shape does not match the planner's scalar expectation.",
                "Check whether aggregation, ranking, or DISTINCT is still missing.",
                "Do not continue exploring detail rows if the task expects a single final value.",
            ],
            should_replan=True,
        )

    if answer_shape_hint == "single_row_multi_col" and (row_count != 1 or column_count < max(len(expected_columns), 2)):
        return RuntimeFeedback(
            signature="shape_mismatch",
            severity="warning",
            summary=f"Expected a single-row multi-column result, but observed shape ({row_count}, {column_count}).",
            prompt_lines=[
                "The current result shape is inconsistent with the expected single-row answer.",
                "Check whether the query still needs aggregation, a final filter, or a different projection set.",
            ],
            should_replan=True,
        )

    if answer_shape_hint == "list_single_col" and column_count != 1:
        return RuntimeFeedback(
            signature="shape_mismatch",
            severity="warning",
            summary=f"Expected a single-column list, but observed {column_count} columns.",
            prompt_lines=[
                "The current result should likely be a single projected column.",
                "Re-check whether extra identifier or helper columns should be removed from the final projection.",
            ],
            should_replan=True,
        )

    return None


def route_mismatch_feedback(
    *,
    action: str,
    expected_sources: list[str],
    actual_source: str | None,
    expected_source_types: list[str],
    actual_source_type: str | None,
    current_step_type: str | None,
    join_anchor: str | None,
) -> RuntimeFeedback:
    expected_sources_text = ", ".join(expected_sources[:4]) if expected_sources else "planned route sources"
    expected_types_text = ", ".join(expected_source_types[:3]) if expected_source_types else "planned source types"
    actual_source_text = actual_source or "unknown source"
    actual_type_text = actual_source_type or "unknown type"
    route_label = current_step_type or "planned route"
    prompt_lines = [
        f"The selected action `{action}` drifted away from the current {route_label}.",
        f"Prefer the planned source(s): {expected_sources_text}.",
        f"Prefer the planned source type(s): {expected_types_text}; observed {actual_type_text} at {actual_source_text}.",
    ]
    if join_anchor:
        prompt_lines.append(
            f"Keep the planned join anchor `{join_anchor}` in mind before exploring a different source."
        )
    prompt_lines.append(
        "If the planned source is genuinely insufficient, trigger replanning instead of continuing on an unplanned branch."
    )
    return RuntimeFeedback(
        signature="route_mismatch",
        severity="warning",
        summary=(
            f"Action {action} targeted {actual_source_text} ({actual_type_text}) "
            f"instead of the planned route {expected_sources_text} ({expected_types_text})."
        ),
        prompt_lines=prompt_lines,
        should_replan=True,
    )


def route_dependency_feedback(
    *,
    action: str,
    attempted_source: str | None,
    missing_dependencies: list[str],
    current_step_type: str | None,
) -> RuntimeFeedback:
    attempted_source_text = attempted_source or "unknown source"
    dependency_text = ", ".join(missing_dependencies[:4]) if missing_dependencies else "planned prerequisites"
    route_label = current_step_type or "planned route step"
    return RuntimeFeedback(
        signature="route_dependency_mismatch",
        severity="warning",
        summary=(
            f"Action {action} tried to access {attempted_source_text} before its planned dependencies "
            f"were completed: {dependency_text}."
        ),
        prompt_lines=[
            f"The selected action `{action}` is out of order for the current {route_label}.",
            f"Complete these prerequisite sources first: {dependency_text}.",
            "Do not skip the prerequisite evidence or source check and jump ahead to a dependent source.",
        ],
        should_replan=True,
    )


def route_tool_mismatch_feedback(
    *,
    action: str,
    current_step_type: str | None,
    expected_actions: list[str],
    source_ref: str | None,
    source_type: str | None,
) -> RuntimeFeedback:
    expected_actions_text = ", ".join(expected_actions[:4]) if expected_actions else "planned tool types"
    source_ref_text = source_ref or "planned source"
    source_type_text = source_type or "planned source type"
    route_label = current_step_type or "planned route step"
    return RuntimeFeedback(
        signature="route_tool_mismatch",
        severity="warning",
        summary=(
            f"Action {action} is not compatible with {source_ref_text} ({source_type_text}); "
            f"expected one of: {expected_actions_text}."
        ),
        prompt_lines=[
            f"The selected action `{action}` does not fit the current {route_label}.",
            f"For {source_ref_text} ({source_type_text}), use one of: {expected_actions_text}.",
            "Choose a tool that matches the current source type instead of forcing the wrong executor.",
        ],
        should_replan=True,
    )


def merge_runtime_feedback(
    observation: dict[str, Any],
    feedback: RuntimeFeedback | None,
) -> dict[str, Any]:
    if feedback is None:
        return observation
    updated = dict(observation)
    existing_feedback = updated.get("runtime_feedback")
    if not isinstance(existing_feedback, dict):
        updated["runtime_feedback"] = feedback.to_dict()
        return updated

    merged_prompt_lines = [
        str(line)
        for line in existing_feedback.get("prompt_lines", [])
        if str(line).strip()
    ]
    for line in feedback.prompt_lines:
        if line not in merged_prompt_lines:
            merged_prompt_lines.append(line)

    should_replan = bool(existing_feedback.get("should_replan")) or feedback.should_replan
    existing_signature = str(existing_feedback.get("signature", "")).strip()
    existing_severity = str(existing_feedback.get("severity", "")).strip() or feedback.severity
    raw_secondary = existing_feedback.get("secondary_signatures", [])
    secondary_items = raw_secondary if isinstance(raw_secondary, list) else []
    primary_signature = existing_signature or feedback.signature
    if feedback.signature.startswith("route_") and primary_signature == "execution_error":
        primary_signature = feedback.signature
    updated["runtime_feedback"] = {
        "signature": primary_signature,
        "severity": existing_severity,
        "summary": str(existing_feedback.get("summary", "")).strip() or feedback.summary,
        "prompt_lines": merged_prompt_lines,
        "should_replan": should_replan,
        "secondary_signatures": sorted(
            {
                str(item).strip()
                for item in [
                    *secondary_items,
                    feedback.signature,
                ]
                if str(item).strip() and str(item).strip() != primary_signature
            }
        ),
    }
    return updated


def _summarize_execute_python(content: dict[str, Any]) -> tuple[dict[str, Any], RuntimeFeedback | None]:
    success = bool(content.get("success"))
    output = _trim_text(content.get("output", ""))
    stderr = _trim_text(content.get("stderr", ""))
    if success:
        summary = {
            "success": True,
            "output": output,
            "stderr": stderr,
            "summary": "[Execution Success] Python code completed.",
        }
        return summary, None

    error_message = str(content.get("error", "")).strip()
    traceback_text = str(content.get("traceback", "")).strip()
    traceback_lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    final_error_line = traceback_lines[-1] if traceback_lines else error_message or "Unknown Python execution error."
    error_type = final_error_line.split(":", 1)[0] if ":" in final_error_line else "ExecutionError"
    summary = {
        "success": False,
        "error_type": error_type,
        "error_message": final_error_line,
        "stderr": stderr,
        "summary": f"[Execution Failed] Error Type: {error_type}. Message: {final_error_line}",
    }
    feedback = RuntimeFeedback(
        signature="execution_error",
        severity="error",
        summary=final_error_line,
        prompt_lines=[
            "The last execution failed.",
            "Do not make a cosmetic retry on the same path.",
            "Check whether the referenced columns, variables, or join keys are wrong before trying again.",
        ],
        should_replan=True,
    )
    return summary, feedback


def _summarize_list_content(content: dict[str, Any]) -> dict[str, Any]:
    entries = content.get("entries", [])
    if not isinstance(entries, list):
        return content
    trimmed_entries = entries[:_MAX_LIST_ITEMS]
    return {
        "root": content.get("root"),
        "entry_count": len(entries),
        "entries": trimmed_entries,
        "summary": f"Context tree has {len(entries)} entries. Showing first {len(trimmed_entries)}.",
    }


def _summarize_text_preview(content: dict[str, Any]) -> dict[str, Any]:
    preview = _trim_text(content.get("preview", ""))
    return {
        "path": content.get("path"),
        "preview": preview,
        "truncated": bool(content.get("truncated", False)) or len(str(content.get("preview", ""))) > len(preview),
    }


def _summarize_semantic_content(action: str, content: dict[str, Any]) -> dict[str, Any]:
    if action == "describe_semantics":
        return {
            "entity_names": [item.get("name") for item in content.get("entities", [])[:6]],
            "relation_count": len(content.get("relations", [])),
            "dimension_count": len(content.get("dimensions", [])),
            "measure_count": len(content.get("measures", [])),
            "evidence_count": len(content.get("evidence", [])),
        }
    if action == "link_schema_candidates":
        return {
            "query_units": content.get("query_units", [])[:4],
            "chosen_bindings": content.get("chosen_bindings", [])[:4],
            "required_sources": content.get("required_sources", []),
            "ambiguity_count": len(content.get("unresolved_ambiguities", [])),
        }
    if action == "plan_semantic_query":
        return {
            "recommended_path": content.get("recommended_path"),
            "required_sources": content.get("required_sources", []),
            "required_entities": content.get("required_entities", []),
            "required_measures": content.get("required_measures", []),
            "recommended_join_path": content.get("recommended_join_path", [])[:3],
            "replanning_guidance": content.get("replanning_guidance", []),
        }
    return content


def prune_observation(action: str, *, ok: bool, content: Any) -> dict[str, Any]:
    observation: dict[str, Any] = {"ok": ok, "tool": action}

    if not ok:
        feedback: RuntimeFeedback
        if action == "execute_python" and isinstance(content, dict):
            summarized_content, feedback = _summarize_execute_python(content)
            observation["content"] = summarized_content
        else:
            message = str(content)
            feedback = RuntimeFeedback(
                signature="execution_error",
                severity="error",
                summary=message,
                prompt_lines=[
                    "The last tool call failed.",
                    "Do not continue down the same path blindly; check field choices, join keys, or execution syntax.",
                ],
                should_replan=True,
            )
            observation["content"] = {
                "summary": f"[Execution Failed] {message}",
            }
        observation["runtime_feedback"] = feedback.to_dict()
        return observation

    feedback: RuntimeFeedback | None = None
    summarized_content: Any = content
    if isinstance(content, dict):
        if action in _TABULAR_ACTIONS and "columns" in content and "rows" in content:
            summarized_content, feedback = _summarize_tabular_content(action, content)
        elif action == "execute_python":
            summarized_content, feedback = _summarize_execute_python(content)
        elif action == "list_context":
            summarized_content = _summarize_list_content(content)
        elif action in {"read_json", "read_doc"}:
            summarized_content = _summarize_text_preview(content)
        elif action in {"describe_semantics", "link_schema_candidates", "plan_semantic_query"}:
            summarized_content = _summarize_semantic_content(action, content)
        else:
            summarized_content = content
    elif isinstance(content, list):
        summarized_content = content[:_MAX_LIST_ITEMS]

    observation["content"] = summarized_content
    if feedback is not None:
        observation["runtime_feedback"] = feedback.to_dict()
    return observation


def enrich_observation_with_plan(
    observation: dict[str, Any],
    *,
    action: str,
    raw_content: Any,
    plan_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if not observation.get("ok") or not isinstance(raw_content, dict):
        return observation

    shape_feedback = _shape_mismatch_feedback(
        action=action,
        content=raw_content,
        plan_snapshot=plan_snapshot,
    )
    if shape_feedback is None:
        return observation

    existing_feedback = observation.get("runtime_feedback")
    if isinstance(existing_feedback, dict):
        existing_signature = str(existing_feedback.get("signature", "")).strip()
        if existing_signature in {"execution_error", "empty_result", "all_null_result"}:
            return observation

    return merge_runtime_feedback(observation, shape_feedback)


def replan_feedback_message(last_observation: dict[str, Any]) -> str:
    feedback = last_observation.get("runtime_feedback", {})
    if not isinstance(feedback, dict):
        return "The current path is failing repeatedly. Re-check schema links, filters, and joins."

    signature = str(feedback.get("signature", "")).strip()
    summary = str(feedback.get("summary", "")).strip()
    if signature == "empty_result":
        return f"Repeated empty results encountered. {summary} Re-check filter values, formats, and join path."
    if signature == "all_null_result":
        return f"Repeated null-like results encountered. {summary} Re-check projected field and join direction."
    if signature == "execution_error":
        return f"Repeated execution failures encountered. {summary} Re-check field mappings, schema, and query path."
    if signature == "shape_mismatch":
        return (
            f"Repeated answer-shape mismatches encountered. {summary} "
            "Re-check the expected answer shape, missing aggregation, ranking, DISTINCT, and final projection."
        )
    if signature == "route_mismatch":
        return (
            f"Repeated route mismatches encountered. {summary} "
            "Re-check the planned source order, preferred source type, and cross-source join anchor before continuing."
        )
    if signature == "route_dependency_mismatch":
        return (
            f"Repeated route dependency mismatches encountered. {summary} "
            "Complete the prerequisite source checks before attempting the dependent source again."
        )
    if signature == "route_tool_mismatch":
        return (
            f"Repeated route tool mismatches encountered. {summary} "
            "Re-check which execution tool fits the current source type and route step."
        )
    return "The current path is failing repeatedly. Re-check schema links, filters, joins, and execution path."
