from __future__ import annotations

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.semantic.graph_types import GraphCandidateSet
from data_agent_competition.semantic.normalization import normalize_identifier
from data_agent_competition.semantic.types import SemanticSelection, SelectionRole, SourceScope, TargetGrain, TaskBundle


def resolve_target_grain(
    task: TaskBundle,
    artifact: SemanticArtifact,
    source_scopes: tuple[SourceScope, ...],
    selections: tuple[SemanticSelection, ...],
    graph_candidates: GraphCandidateSet,
    routing_metrics=(),
) -> TargetGrain:
    scoped_source_ids = {scope.source_id for scope in source_scopes}
    source_map = {source.source_id: source for source in artifact.sources}
    answer_fields = [
        selection.field_name
        for selection in selections
        if selection.role in {SelectionRole.ANSWER, SelectionRole.DIMENSION}
    ]
    grain_fields = _select_grain_fields(
        source_scopes=source_scopes,
        source_map=source_map,
        scoped_source_ids=scoped_source_ids,
        answer_fields=answer_fields,
        graph_candidates=graph_candidates,
    )
    entity = source_scopes[0].source_id if source_scopes else task.task_id
    time_grain = _resolve_time_grain(task, graph_candidates, routing_metrics, source_scopes, source_map)
    return TargetGrain(
        entity=entity,
        grain_fields=grain_fields,
        time_grain=time_grain,
        measure_scope=None,
    )


def _select_grain_fields(
    *,
    source_scopes: tuple[SourceScope, ...],
    source_map: dict[str, object],
    scoped_source_ids: set[str],
    answer_fields: list[str],
    graph_candidates: GraphCandidateSet,
) -> tuple[str, ...]:
    if _looks_measure_projection(answer_fields):
        preferred_dimension_fields: list[str] = []
        for field_name in answer_fields:
            normalized = normalize_identifier(field_name)
            if normalized in {"cost", "amount", "spent", "price", "value", "total"}:
                continue
            if normalized.endswith("_id") or normalized == "id":
                continue
            preferred_dimension_fields.append(field_name)
        if preferred_dimension_fields:
            return tuple(dict.fromkeys(preferred_dimension_fields[:1]))
    primary_key_fields: list[str] = []
    for scope in source_scopes:
        source = source_map.get(scope.source_id)
        if source is None:
            continue
        for field_name in getattr(source, "grain_hint", []) or []:
            rendered = str(field_name).strip()
            if rendered:
                primary_key_fields.append(rendered)
    if primary_key_fields:
        return tuple(dict.fromkeys(primary_key_fields))

    candidate_identifier_fields: list[str] = []
    for candidate in graph_candidates.field_candidates:
        if candidate.source_id not in scoped_source_ids or not candidate.field_name:
            continue
        semantic_tags = {normalize_identifier(str(tag)) for tag in candidate.metadata.get("semantic_tags", [])}
        if "primary_key" in semantic_tags:
            candidate_identifier_fields.append(candidate.field_name)
            continue
        field_norm = normalize_identifier(candidate.field_name)
        if field_norm.endswith("_id") or field_norm == "id":
            candidate_identifier_fields.append(candidate.field_name)
    if candidate_identifier_fields:
        return tuple(dict.fromkeys(candidate_identifier_fields))

    answer_identifier_fields = [
        field_name
        for field_name in answer_fields
        if normalize_identifier(field_name).endswith("_id") or normalize_identifier(field_name) == "id"
    ]
    if answer_identifier_fields:
        return tuple(dict.fromkeys(answer_identifier_fields))

    preferred_dimension_fields: list[str] = []
    for field_name in answer_fields:
        normalized = normalize_identifier(field_name)
        if normalized.endswith("_name") or normalized in {"name", "title", "label", "code"}:
            continue
        preferred_dimension_fields.append(field_name)
    if preferred_dimension_fields:
        return tuple(dict.fromkeys(preferred_dimension_fields[:1]))
    if answer_fields:
        return tuple(dict.fromkeys(answer_fields[:1]))
    return ()


def _looks_measure_projection(answer_fields: list[str]) -> bool:
    normalized_fields = {normalize_identifier(field_name) for field_name in answer_fields}
    has_measure = bool(normalized_fields & {"cost", "amount", "spent", "price", "value", "total"})
    has_dimension = any(
        field_name not in {"cost", "amount", "spent", "price", "value", "total"}
        and not field_name.endswith("_id")
        and field_name != "id"
        for field_name in normalized_fields
    )
    return has_measure and has_dimension


def _resolve_time_grain(
    task: TaskBundle,
    graph_candidates: GraphCandidateSet,
    routing_metrics,
    source_scopes: tuple[SourceScope, ...],
    source_map: dict[str, object],
) -> str | None:
    metric_time_grains = [
        str(metric.requires_time_grain).strip()
        for metric in routing_metrics
        if getattr(metric, "requires_time_grain", None)
    ]
    if metric_time_grains:
        return metric_time_grains[0]

    scoped_source_ids = {scope.source_id for scope in source_scopes}
    temporal_fields: list[str] = []
    for candidate in graph_candidates.field_candidates:
        if candidate.source_id not in scoped_source_ids or not candidate.field_name:
            continue
        dtype = normalize_identifier(str(candidate.metadata.get("dtype", "")))
        field_norm = normalize_identifier(candidate.field_name)
        if dtype in {"date", "datetime", "timestamp"} or any(token in field_norm for token in ("date", "time", "year", "month")):
            temporal_fields.append(field_norm)

    lowered_question = task.question.lower()
    if "month" in lowered_question and temporal_fields:
        return "month"
    if "year" in lowered_question and temporal_fields:
        return "year"
    return None
