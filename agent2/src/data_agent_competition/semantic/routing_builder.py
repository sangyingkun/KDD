from __future__ import annotations

from dataclasses import dataclass
import re

from data_agent_competition.artifacts.schema import SemanticArtifact, SourceDescriptor, SourceField
from data_agent_competition.semantic.normalization import normalize_identifier, token_overlap_score
from data_agent_competition.semantic.types import (
    AmbiguityWarning,
    AssetKind,
    GroundingResult,
    JoinEdge,
    PostSQLEnrichment,
    QuestionType,
    RoutingAnswerKind,
    RoutingFieldRef,
    RoutingFilter,
    RoutingMetric,
    SemanticRoutingSpec,
    SemanticSelection,
    SelectionRole,
    SourceScope,
    TaskBundle,
)


@dataclass(frozen=True, slots=True)
class _AnswerRequest:
    phrase: str
    normalized_phrase: str
    requested_aliases: frozenset[str]
    entity_hint: str | None
    asks_identifier: bool
    asks_measure: bool


def build_routing_spec(
    *,
    task: TaskBundle,
    artifact: SemanticArtifact,
    question_type: QuestionType,
    grounding: GroundingResult,
    graph_candidates,
    retrieval_notes: tuple[str, ...],
) -> SemanticRoutingSpec:
    target_sources = _target_sources(artifact, graph_candidates, grounding)
    target_sources = _ensure_graph_bridge_sources(artifact, target_sources, graph_candidates)
    answer_slots = _answer_slots(task.question, artifact, graph_candidates, target_sources)
    target_sources = _ensure_answer_sources(artifact, target_sources, answer_slots)
    join_path = _graph_joins(artifact, target_sources, graph_candidates, answer_slots)
    filters = _merge_filters(
        _filters_from_grounding(grounding),
        _boolean_question_filters(
            question=task.question,
            artifact=artifact,
            graph_candidates=graph_candidates,
            source_scopes=target_sources,
            answer_slots=answer_slots,
        ),
    )
    time_constraints = _time_constraints(task.question, artifact, target_sources)
    metrics = _routing_metrics(task.question, graph_candidates)
    ambiguity_warnings = _ambiguity_warnings(graph_candidates)
    post_sql_enrichments = _post_sql_enrichments(task, artifact, target_sources, answer_slots)
    return SemanticRoutingSpec(
        question_type=question_type,
        target_sources=target_sources,
        join_path=join_path,
        answer_slots=answer_slots,
        metrics=metrics,
        filters=filters,
        time_constraints=time_constraints,
        post_sql_enrichments=post_sql_enrichments,
        ambiguity_warnings=ambiguity_warnings,
        supporting_node_ids=graph_candidates.evidence_node_ids,
        supporting_edge_ids=graph_candidates.evidence_edge_ids,
        notes=("graph_routing_spec", *retrieval_notes, *grounding.notes),
    )


def routing_selections(routing_spec: SemanticRoutingSpec) -> tuple[SemanticSelection, ...]:
    selections: list[SemanticSelection] = []
    for slot in routing_spec.answer_slots:
        role = SelectionRole.ANSWER if slot.answer_kind != RoutingAnswerKind.MEASURE else SelectionRole.MEASURE
        selections.append(
            SemanticSelection(
                source_id=slot.source_id,
                field_name=slot.field_name,
                role=role,
                alias=slot.field_name,
                confidence=slot.confidence,
                rationale=slot.rationale,
                graph_node_id=slot.graph_node_id,
            )
        )
    return tuple(selections)


def _target_sources(
    artifact: SemanticArtifact,
    graph_candidates,
    grounding: GroundingResult,
) -> tuple[SourceScope, ...]:
    source_map = {source.source_id: source for source in artifact.sources}
    required_by_source: dict[str, set[str]] = {}
    for term in grounding.grounded_terms:
        if term.source_id and term.field_name:
            required_by_source.setdefault(term.source_id, set()).add(term.field_name)
    support_scores: dict[str, float] = {}
    rationales: dict[str, str] = {}

    def register_support(source_id: str | None, score: float, rationale: str) -> None:
        if not source_id or source_id not in source_map:
            return
        support_scores[source_id] = max(support_scores.get(source_id, 0.0), score)
        existing = rationales.get(source_id)
        if existing is None or score >= support_scores.get(source_id, 0.0):
            rationales[source_id] = rationale

    for candidate in graph_candidates.source_candidates:
        register_support(candidate.source_id, candidate.score + 0.35, candidate.rationale or "graph_source_candidate")
    for candidate in graph_candidates.field_candidates:
        register_support(candidate.source_id, candidate.score * 0.9, candidate.rationale or "graph_field_support")
    for candidate in (
        *graph_candidates.metric_candidates,
        *graph_candidates.constraint_candidates,
        *graph_candidates.ambiguity_candidates,
        *graph_candidates.use_case_candidates,
        *graph_candidates.value_candidates,
    ):
        for source_id in _candidate_bound_source_ids(candidate):
            register_support(source_id, candidate.score * 0.72, candidate.rationale or "graph_business_support")
    for source_id, fields in required_by_source.items():
        register_support(source_id, 0.88 + (0.02 * min(len(fields), 4)), "grounded_filter_support")

    ranked_source_ids = [
        source_id
        for source_id, _score in sorted(
            support_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    scopes: list[SourceScope] = []
    for index, source_id in enumerate(ranked_source_ids):
        source = source_map[source_id]
        scopes.append(
            SourceScope(
                source_id=source.source_id,
                source_kind=AssetKind(source.source_kind),
                asset_path=source.asset_path,
                rationale=rationales.get(source_id, "graph_source_support"),
                confidence=support_scores.get(source_id, 0.0),
                priority=index,
                required_fields=tuple(sorted(required_by_source.get(source.source_id, set()))),
            )
        )
    return tuple(scopes[:4])


def _graph_joins(
    artifact: SemanticArtifact,
    source_scopes: tuple[SourceScope, ...],
    graph_candidates,
    answer_slots: tuple[RoutingFieldRef, ...],
) -> tuple[JoinEdge, ...]:
    scoped_source_ids = list(dict.fromkeys(scope.source_id for scope in source_scopes))
    for slot in answer_slots:
        if slot.source_id not in scoped_source_ids:
            scoped_source_ids.append(slot.source_id)
    scoped_source_ids = tuple(scoped_source_ids)
    if len(scoped_source_ids) < 2:
        return ()
    allowed = set(scoped_source_ids)
    adjacency: dict[str, list] = {}
    for candidate in graph_candidates.join_candidates:
        join_edge = _join_edge_from_graph_candidate(candidate)
        if join_edge is None:
            continue
        if join_edge.left_source_id not in allowed or join_edge.right_source_id not in allowed:
            continue
        adjacency.setdefault(join_edge.left_source_id, []).append(join_edge)
        adjacency.setdefault(join_edge.right_source_id, []).append(join_edge)
    for join_edge in _artifact_join_edges(artifact):
        if join_edge.left_source_id not in allowed or join_edge.right_source_id not in allowed:
            continue
        adjacency.setdefault(join_edge.left_source_id, []).append(join_edge)
        adjacency.setdefault(join_edge.right_source_id, []).append(join_edge)
    root = scoped_source_ids[0]
    seen_sources = {root}
    selected: list[JoinEdge] = []
    frontier = [root]
    while frontier:
        current = frontier.pop(0)
        candidates = sorted(
            adjacency.get(current, []),
            key=lambda item: (-item.confidence, item.left_source_id, item.right_source_id),
        )
        for join_edge in candidates:
            neighbor = join_edge.right_source_id if join_edge.left_source_id == current else join_edge.left_source_id
            if neighbor in seen_sources:
                continue
            selected.append(join_edge)
            seen_sources.add(neighbor)
            frontier.append(neighbor)
    return tuple(selected)


def _answer_slots(
    question: str,
    artifact: SemanticArtifact,
    graph_candidates,
    source_scopes: tuple[SourceScope, ...],
) -> tuple[RoutingFieldRef, ...]:
    source_rank = {scope.source_id: index for index, scope in enumerate(source_scopes)}
    lowered_question = question.lower()
    wants_measure = any(token in lowered_question for token in ("total", "sum", "average", "avg", "count", "highest", "lowest"))
    answer_requests = _explicit_answer_requests(question)
    explicit_answer_terms = _explicit_answer_terms(question)
    filter_keys = {
        (candidate.source_id, candidate.field_name)
        for candidate in graph_candidates.value_candidates
        for candidate_source_id, candidate_field_name in [_bound_field_key(candidate)]
        if candidate_source_id and candidate_field_name
    }
    slots: list[RoutingFieldRef] = []
    if answer_requests:
        slots.extend(
            _slots_from_answer_requests(
                question=question,
                artifact=artifact,
                graph_candidates=graph_candidates,
                source_scopes=source_scopes,
                answer_requests=answer_requests,
                filter_keys=filter_keys,
            )
        )
    for candidate in graph_candidates.field_candidates:
        if not candidate.source_id or not candidate.field_name:
            continue
        if candidate.source_id not in source_rank:
            continue
        answer_kind = _answer_kind(
            question,
            candidate.field_name,
            wants_measure,
            explicit_answer_terms,
            is_filter_bound=(candidate.source_id, candidate.field_name) in filter_keys,
        )
        if answer_kind is None:
            continue
        slots.append(
            RoutingFieldRef(
                source_id=candidate.source_id,
                field_name=candidate.field_name,
                answer_kind=answer_kind,
                confidence=candidate.score,
                rationale=candidate.rationale or "graph_field_candidate",
                graph_node_id=candidate.node_id,
            )
        )
    if not answer_requests:
        slots.extend(_artifact_answer_slot_fallbacks(question, artifact, graph_candidates, source_scopes))
    slots.sort(
        key=lambda item: (
            _answer_kind_rank(item.answer_kind),
            source_rank.get(item.source_id, 999),
            -item.confidence,
            item.field_name.lower(),
        )
    )
    deduped: dict[tuple[str, str], RoutingFieldRef] = {}
    for slot in slots:
        deduped.setdefault((slot.source_id, slot.field_name), slot)
    return tuple(list(deduped.values())[:8])


def _answer_kind(
    question: str,
    field_name: str,
    wants_measure: bool,
    explicit_answer_terms: set[str],
    *,
    is_filter_bound: bool,
) -> RoutingAnswerKind | None:
    lowered_question = question.lower()
    lowered_field = field_name.lower()
    normalized_field = lowered_field.replace("_", " ")
    explicit_id_requested = "id" in explicit_answer_terms or "identifier" in explicit_answer_terms
    if explicit_answer_terms:
        if lowered_field.endswith("id") and ("id" in explicit_answer_terms or normalized_field in explicit_answer_terms):
            return RoutingAnswerKind.IDENTIFIER
        if (
            normalized_field in explicit_answer_terms
            or lowered_field in explicit_answer_terms
            or _field_answer_alias(lowered_field) in explicit_answer_terms
        ):
            return RoutingAnswerKind.ATTRIBUTE
        if is_filter_bound:
            return None
    if lowered_field.endswith("id"):
        if explicit_answer_terms and not explicit_id_requested:
            return None
        return RoutingAnswerKind.IDENTIFIER
    if wants_measure and lowered_field in {"cost", "amount", "spent", "price", "consumption", "count"}:
        return RoutingAnswerKind.MEASURE
    if is_filter_bound:
        return None
    if lowered_field in lowered_question or any(token in lowered_question for token in ("name", "type", "category", "country", "sex", "diagnosis", "disease")):
        return RoutingAnswerKind.ATTRIBUTE
    return None


def _answer_kind_rank(answer_kind: RoutingAnswerKind) -> int:
    if answer_kind == RoutingAnswerKind.IDENTIFIER:
        return 0
    if answer_kind == RoutingAnswerKind.ATTRIBUTE:
        return 1
    if answer_kind == RoutingAnswerKind.GROUP_BY:
        return 2
    return 3


def _filters_from_grounding(grounding: GroundingResult) -> tuple[RoutingFilter, ...]:
    filters: list[RoutingFilter] = []
    for term in grounding.grounded_terms:
        if not term.source_id or not term.field_name or term.resolved_value is None:
            continue
        filters.append(
            RoutingFilter(
                source_id=term.source_id,
                field_name=term.field_name,
                operator="=",
                value=term.resolved_value,
                confidence=term.confidence,
                rationale=term.grounding_type,
                graph_node_id=term.graph_node_id,
            )
        )
    deduped: dict[tuple[str, str, str], RoutingFilter] = {}
    for item in filters:
        deduped.setdefault((item.source_id, item.field_name, item.value), item)
    return tuple(deduped.values())


def _merge_filters(*groups: tuple[RoutingFilter, ...]) -> tuple[RoutingFilter, ...]:
    deduped: dict[tuple[str, str, str, str], RoutingFilter] = {}
    for group in groups:
        for item in group:
            deduped.setdefault((item.source_id, item.field_name, item.operator, item.value), item)
    return tuple(deduped.values())


def _time_constraints(
    question: str,
    artifact: SemanticArtifact,
    source_scopes: tuple[SourceScope, ...],
) -> tuple[RoutingFilter, ...]:
    lowered = question.lower()
    month_match = _explicit_month_phrase(lowered)
    year_match = re.search(r"\b(19|20)\d{2}\b", lowered)
    if month_match is None and year_match is None:
        return ()
    scoped_ids = {scope.source_id for scope in source_scopes}
    constraints: list[RoutingFilter] = []
    for source in artifact.sources:
        if source.source_id not in scoped_ids:
            continue
        for field in source.fields:
            lowered_field = field.field_name.lower()
            if "date" not in lowered_field and "time" not in lowered_field and "year" not in lowered_field:
                continue
            value = year_match.group(0) if year_match else ""
            operator = "year_equals"
            if month_match is not None and year_match is not None:
                value = f"{year_match.group(0)}-{_MONTH_INDEX[month_match]}"
                operator = "month_year_equals"
            elif month_match is not None:
                value = _MONTH_INDEX[month_match]
                operator = "month_equals"
            constraints.append(
                RoutingFilter(
                    source_id=source.source_id,
                    field_name=field.field_name,
                    operator=operator,
                    value=value,
                    confidence=0.8,
                    rationale="question_temporal_phrase",
                )
            )
            return tuple(constraints)
    return ()


def _routing_metrics(question: str, graph_candidates) -> tuple[RoutingMetric, ...]:
    metrics: list[RoutingMetric] = []
    lowered_question = question.lower()
    for candidate in graph_candidates.metric_candidates:
        if candidate.label.lower() in lowered_question or any(token in lowered_question for token in ("ratio", "utilization", "index", "rate", "average", "total")):
            source_fields = _metric_source_fields(candidate, graph_candidates)
            metrics.append(
                RoutingMetric(
                    node_id=candidate.node_id,
                    label=candidate.label,
                    formula=_string_metadata(candidate.metadata.get("formula")),
                    source_fields=source_fields,
                    requires_time_grain=_string_metadata(candidate.metadata.get("requires_time_grain")),
                    confidence=candidate.score,
                    rationale=candidate.rationale or "graph_metric_candidate",
                )
            )
    return tuple(metrics[:4])


def _metric_source_fields(candidate, graph_candidates) -> tuple[str, ...]:
    bound_refs = [
        str(item).strip()
        for item in candidate.metadata.get("bound_field_refs", [])
        if str(item).strip()
    ]
    if bound_refs:
        return tuple(dict.fromkeys(bound_refs))[:6]
    return _bind_metric_fields(candidate, graph_candidates)


def _bind_metric_fields(candidate, graph_candidates) -> tuple[str, ...]:
    field_terms = [
        str(item).strip().lower()
        for item in candidate.metadata.get("field_terms", [])
        if str(item).strip()
    ]
    if not field_terms:
        return ()
    bindings: list[str] = []
    seen: set[str] = set()
    for field_candidate in graph_candidates.field_candidates:
        if not field_candidate.source_id or not field_candidate.field_name:
            continue
        lowered_field = field_candidate.field_name.lower()
        if lowered_field not in field_terms:
            continue
        qualified = f"{field_candidate.source_id}.{field_candidate.field_name}"
        if qualified in seen:
            continue
        seen.add(qualified)
        bindings.append(qualified)
    return tuple(bindings[:6])


def _ambiguity_warnings(graph_candidates) -> tuple[AmbiguityWarning, ...]:
    warnings: list[AmbiguityWarning] = []
    for candidate in graph_candidates.ambiguity_candidates:
        warnings.append(
            AmbiguityWarning(
                node_id=candidate.node_id,
                label=candidate.label,
                message=candidate.canonical_text,
                preferred_source_id=_string_metadata(candidate.metadata.get("preferred_source_id")),
                preferred_field_name=_string_metadata(candidate.metadata.get("preferred_field_name")),
                confidence=candidate.score,
            )
        )
    return tuple(warnings[:8])


def _post_sql_enrichments(
    task: TaskBundle,
    artifact: SemanticArtifact,
    source_scopes: tuple[SourceScope, ...],
    answer_slots: tuple[RoutingFieldRef, ...],
) -> tuple[PostSQLEnrichment, ...]:
    lowered_question = task.question.lower()
    if not any(token in lowered_question for token in ("json", "doc", "background", "describe", "introduction", "summary")):
        return ()
    scoped_source_ids = {scope.source_id for scope in source_scopes}
    answer_field_names = {slot.field_name.lower() for slot in answer_slots}
    enrichments: list[PostSQLEnrichment] = []
    for source in artifact.sources:
        if source.source_id in scoped_source_ids:
            continue
        if source.source_kind != AssetKind.JSON.value:
            continue
        exact_overlap = [
            field.field_name
            for field in source.fields
            if field.field_name.lower() in answer_field_names
        ]
        name_or_id = [
            field.field_name
            for field in source.fields
            if field.field_name.lower().endswith("name") or field.field_name.lower().endswith("id")
        ]
        match_field = (exact_overlap or name_or_id[:1] or [None])[0]
        if match_field is None:
            continue
        enrichments.append(
            PostSQLEnrichment(
                source_id=source.source_id,
                asset_path=source.asset_path,
                source_kind=AssetKind(source.source_kind),
                match_field=match_field,
                purpose="post_sql_context_enrichment",
                confidence=0.82 if exact_overlap else 0.68,
                rationale="question_requests_json_context",
            )
        )
    return tuple(enrichments[:3])


def _ensure_graph_bridge_sources(
    artifact: SemanticArtifact,
    source_scopes: tuple[SourceScope, ...],
    graph_candidates,
) -> tuple[SourceScope, ...]:
    scoped_source_ids = {scope.source_id for scope in source_scopes}
    source_map = {source.source_id: source for source in artifact.sources}
    bridge_scopes = list(source_scopes)
    changed = True
    while changed and len(bridge_scopes) < 4:
        changed = False
        for candidate in graph_candidates.join_candidates:
            join_edge = _join_edge_from_graph_candidate(candidate)
            if join_edge is None:
                continue
            left_in = join_edge.left_source_id in scoped_source_ids
            right_in = join_edge.right_source_id in scoped_source_ids
            if left_in == right_in:
                continue
            missing_source_id = join_edge.right_source_id if left_in else join_edge.left_source_id
            source = source_map.get(missing_source_id)
            if source is None:
                continue
            bridge_scopes.append(
                SourceScope(
                    source_id=source.source_id,
                    source_kind=AssetKind(source.source_kind),
                    asset_path=source.asset_path,
                    rationale="bridge_source_from_graph_join",
                    confidence=join_edge.confidence,
                    priority=len(bridge_scopes),
                    required_fields=(),
                )
            )
            scoped_source_ids.add(source.source_id)
            changed = True
            if len(bridge_scopes) >= 4:
                break
    deduped: dict[str, SourceScope] = {}
    for scope in bridge_scopes:
        deduped.setdefault(scope.source_id, scope)
    return tuple(list(deduped.values())[:4])


def _ensure_answer_sources(
    artifact: SemanticArtifact,
    source_scopes: tuple[SourceScope, ...],
    answer_slots: tuple[RoutingFieldRef, ...],
) -> tuple[SourceScope, ...]:
    scoped_ids = {scope.source_id for scope in source_scopes}
    if all(slot.source_id in scoped_ids for slot in answer_slots):
        return source_scopes
    source_map = {source.source_id: source for source in artifact.sources}
    expanded = list(source_scopes)
    for slot in answer_slots:
        if slot.source_id in scoped_ids:
            continue
        source = source_map.get(slot.source_id)
        if source is None:
            continue
        expanded.append(
            SourceScope(
                source_id=source.source_id,
                source_kind=AssetKind(source.source_kind),
                asset_path=source.asset_path,
                rationale="answer_slot_required_source",
                confidence=max(slot.confidence, 0.7),
                priority=len(expanded),
                required_fields=(slot.field_name,),
            )
        )
        scoped_ids.add(slot.source_id)
        if len(expanded) >= 4:
            break
    deduped: dict[str, SourceScope] = {}
    for scope in expanded:
        existing = deduped.get(scope.source_id)
        if existing is None:
            deduped[scope.source_id] = scope
            continue
        merged_required = tuple(dict.fromkeys((*existing.required_fields, *scope.required_fields)))
        deduped[scope.source_id] = SourceScope(
            source_id=existing.source_id,
            source_kind=existing.source_kind,
            asset_path=existing.asset_path,
            rationale=existing.rationale,
            confidence=max(existing.confidence, scope.confidence),
            priority=min(existing.priority, scope.priority),
            required_fields=merged_required,
        )
    return tuple(sorted(deduped.values(), key=lambda item: (item.priority, -item.confidence, item.source_id)))[:4]


def _join_edge_from_graph_candidate(candidate) -> JoinEdge | None:
    left_source_id = _string_metadata(candidate.metadata.get("left_source_id"))
    left_field = _string_metadata(candidate.metadata.get("left_field"))
    right_source_id = _string_metadata(candidate.metadata.get("right_source_id"))
    right_field = _string_metadata(candidate.metadata.get("right_field"))
    if not all((left_source_id, left_field, right_source_id, right_field)):
        return None
    return JoinEdge(
        left_source_id=left_source_id,
        left_field=left_field,
        right_source_id=right_source_id,
        right_field=right_field,
        confidence=candidate.score,
        rationale=candidate.rationale or "graph_join_candidate",
    )


def _artifact_join_edges(artifact: SemanticArtifact) -> tuple[JoinEdge, ...]:
    return tuple(
        JoinEdge(
            left_source_id=candidate.left_source_id,
            left_field=candidate.left_field,
            right_source_id=candidate.right_source_id,
            right_field=candidate.right_field,
            confidence=candidate.confidence,
            rationale=candidate.rationale or "artifact_join_candidate",
        )
        for candidate in artifact.join_candidates
    )


def _candidate_bound_source_ids(candidate) -> tuple[str, ...]:
    source_ids: list[str] = []
    if candidate.source_id:
        source_ids.append(candidate.source_id)
    for value in candidate.metadata.get("bound_source_ids", []):
        rendered = str(value).strip()
        if rendered:
            source_ids.append(rendered)
    return tuple(dict.fromkeys(source_ids))


def _bound_field_key(candidate) -> tuple[str | None, str | None]:
    bound_refs = [
        str(item).strip()
        for item in candidate.metadata.get("bound_field_refs", [])
        if str(item).strip()
    ]
    if bound_refs:
        source_id, field_name = bound_refs[0].rsplit(".", maxsplit=1)
        return source_id, field_name
    return _string_metadata(candidate.metadata.get("source_id")), _string_metadata(candidate.metadata.get("field_name"))


def _explicit_answer_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for request in _explicit_answer_requests(question):
        terms.add(request.normalized_phrase.replace("_", " "))
        terms.update(alias.replace("_", " ") for alias in request.requested_aliases if alias)
    return {term for term in terms if term}


def _explicit_answer_requests(question: str) -> tuple[_AnswerRequest, ...]:
    answer_segment = _answer_segment(question)
    if not answer_segment:
        return ()
    requests: list[_AnswerRequest] = []
    inherited_entity_hint: str | None = None
    for raw_piece in re.split(r",|\band\b", answer_segment, flags=re.IGNORECASE):
        original_piece = raw_piece.strip(" .")
        if not original_piece:
            continue
        lowered_piece = original_piece.lower()
        uses_inherited_entity = lowered_piece.startswith("their ")
        cleaned_piece = re.sub(r"^(?:identify|list|show|return|provide|give)\s+", "", lowered_piece).strip()
        cleaned_piece = re.sub(r"^(?:their|the|a|an)\s+", "", cleaned_piece).strip(" .")
        if not cleaned_piece:
            continue
        normalized_piece = normalize_identifier(cleaned_piece)
        if not normalized_piece:
            continue
        tokens = [token for token in normalized_piece.split("_") if token]
        asks_identifier = any(token in {"id", "identifier"} for token in tokens)
        asks_measure = any(token in {"total", "sum", "average", "avg", "count", "value", "amount", "cost", "spent"} for token in tokens)
        entity_hint = _entity_hint_from_phrase(tokens)
        if entity_hint is None and uses_inherited_entity:
            entity_hint = inherited_entity_hint
        requests.append(
            _AnswerRequest(
                phrase=cleaned_piece,
                normalized_phrase=normalized_piece,
                requested_aliases=frozenset(_requested_term_aliases(normalized_piece)),
                entity_hint=entity_hint,
                asks_identifier=asks_identifier,
                asks_measure=asks_measure,
            )
        )
        if entity_hint is not None:
            inherited_entity_hint = entity_hint
    return tuple(requests)


def _answer_segment(question: str) -> str:
    lowered = question.lower()
    for marker in (" list ", " show ", " return ", " identify ", " provide ", " give "):
        wrapped = f" {lowered} "
        if marker not in wrapped:
            continue
        lowered = wrapped.split(marker, maxsplit=1)[1]
        break
    for marker in (" where ", " among ", " from ", " for "):
        wrapped = f" {lowered} "
        if marker not in wrapped:
            continue
        lowered = wrapped.split(marker, maxsplit=1)[0]
        break
    return lowered.strip(" .")


def _entity_hint_from_phrase(tokens: list[str]) -> str | None:
    if "of" in tokens:
        index = tokens.index("of")
        trailing = [token for token in tokens[index + 1 :] if token not in {"their", "the", "a", "an"}]
        if trailing:
            return trailing[-1]
    preferred = [
        token
        for token in tokens
        if token not in {"type", "value", "total", "approved", "cost", "amount", "spent", "id", "identifier"}
    ]
    if preferred:
        return preferred[-1]
    return None


def _artifact_answer_slot_fallbacks(
    question: str,
    artifact: SemanticArtifact,
    graph_candidates,
    source_scopes: tuple[SourceScope, ...],
) -> list[RoutingFieldRef]:
    explicit_terms = _explicit_answer_terms(question)
    if not explicit_terms:
        return []
    source_rank = {scope.source_id: index for index, scope in enumerate(source_scopes)}
    graph_field_index = {
        (candidate.source_id, candidate.field_name): candidate
        for candidate in graph_candidates.field_candidates
        if candidate.source_id and candidate.field_name
    }
    graph_join_sources = {
        join.left_source_id
        for join in _joins_from_candidates(graph_candidates)
    } | {
        join.right_source_id
        for join in _joins_from_candidates(graph_candidates)
    }
    fallbacks: list[RoutingFieldRef] = []
    seen_terms: set[str] = set()
    wants_measure = any(token in question.lower() for token in ("total", "sum", "average", "avg", "count", "highest", "lowest"))
    filter_keys = {
        _bound_field_key(candidate)
        for candidate in graph_candidates.value_candidates
        if all(_bound_field_key(candidate))
    }
    for requested_term in sorted(explicit_terms, key=len, reverse=True):
        match = _best_artifact_field_match(
            requested_term=requested_term,
            artifact=artifact,
            scoped_source_ids={scope.source_id for scope in source_scopes},
            graph_join_sources=graph_join_sources,
        )
        if match is None:
            continue
        source, field, score = match
        term_key = f"{source.source_id}.{field.field_name}"
        if term_key in seen_terms:
            continue
        field_candidate = graph_field_index.get((source.source_id, field.field_name))
        answer_kind = _answer_kind(
            question,
            field.field_name,
            wants_measure,
            explicit_terms,
            is_filter_bound=(source.source_id, field.field_name) in filter_keys,
        )
        if answer_kind is None:
            continue
        fallbacks.append(
            RoutingFieldRef(
                source_id=source.source_id,
                field_name=field.field_name,
                answer_kind=answer_kind,
                confidence=max(field_candidate.score if field_candidate else 0.0, score),
                rationale="artifact_schema_answer_fallback" if field_candidate is None else "artifact_schema_answer_upgrade",
                graph_node_id=field_candidate.node_id if field_candidate else f"field::{source.source_id}.{field.field_name}",
            )
        )
        seen_terms.add(term_key)
    fallbacks.sort(
        key=lambda item: (
            _answer_kind_rank(item.answer_kind),
            source_rank.get(item.source_id, 999),
            -item.confidence,
            item.field_name.lower(),
        )
    )
    return fallbacks


def _slots_from_answer_requests(
    *,
    question: str,
    artifact: SemanticArtifact,
    graph_candidates,
    source_scopes: tuple[SourceScope, ...],
    answer_requests: tuple[_AnswerRequest, ...],
    filter_keys: set[tuple[str, str]],
) -> list[RoutingFieldRef]:
    source_rank = {scope.source_id: index for index, scope in enumerate(source_scopes)}
    graph_index = {
        (candidate.source_id, candidate.field_name): candidate
        for candidate in graph_candidates.field_candidates
        if candidate.source_id and candidate.field_name
    }
    scoped_ids = {scope.source_id for scope in source_scopes}
    join_source_ids = {
        join.left_source_id
        for join in (*_joins_from_candidates(graph_candidates), *_artifact_join_edges(artifact))
    } | {
        join.right_source_id
        for join in (*_joins_from_candidates(graph_candidates), *_artifact_join_edges(artifact))
    }
    slots: list[RoutingFieldRef] = []
    for request in answer_requests:
        match = _best_request_match(
            request=request,
            artifact=artifact,
            graph_index=graph_index,
            scoped_source_ids=scoped_ids,
            join_source_ids=join_source_ids,
        )
        if match is None:
            continue
        source, field, score = match
        candidate = graph_index.get((source.source_id, field.field_name))
        answer_kind = _answer_kind(
            question,
            field.field_name,
            wants_measure=request.asks_measure,
            explicit_answer_terms=set(request.requested_aliases),
            is_filter_bound=(source.source_id, field.field_name) in filter_keys,
        )
        if request.asks_measure:
            answer_kind = RoutingAnswerKind.MEASURE
        elif request.asks_identifier:
            answer_kind = RoutingAnswerKind.IDENTIFIER
        if answer_kind is None:
            continue
        slots.append(
            RoutingFieldRef(
                source_id=source.source_id,
                field_name=field.field_name,
                answer_kind=answer_kind,
                confidence=max(candidate.score if candidate else 0.0, score),
                rationale=candidate.rationale if candidate is not None else "explicit_answer_request",
                graph_node_id=candidate.node_id if candidate is not None else f"field::{source.source_id}.{field.field_name}",
            )
        )
    slots.sort(
        key=lambda item: (
            _answer_kind_rank(item.answer_kind),
            source_rank.get(item.source_id, 999),
            -item.confidence,
            item.field_name.lower(),
        )
    )
    return slots


def _best_request_match(
    *,
    request: _AnswerRequest,
    artifact: SemanticArtifact,
    graph_index: dict[tuple[str, str], object],
    scoped_source_ids: set[str],
    join_source_ids: set[str],
) -> tuple[SourceDescriptor, SourceField, float] | None:
    best: tuple[float, SourceDescriptor, SourceField] | None = None
    for source in artifact.sources:
        for field in source.fields:
            candidate = graph_index.get((source.source_id, field.field_name))
            score = _request_field_match_score(
                request=request,
                source=source,
                field=field,
                candidate=candidate,
                scoped_source_ids=scoped_source_ids,
                join_source_ids=join_source_ids,
            )
            if score <= 0.0:
                continue
            if best is None or score > best[0]:
                best = (score, source, field)
    if best is None or best[0] < 0.78:
        return None
    return best[1], best[2], min(best[0], 0.98)


def _request_field_match_score(
    *,
    request: _AnswerRequest,
    source: SourceDescriptor,
    field: SourceField,
    candidate,
    scoped_source_ids: set[str],
    join_source_ids: set[str],
) -> float:
    field_norm = normalize_identifier(field.field_name)
    semantic_tags = {normalize_identifier(tag) for tag in field.semantic_tags}
    if not request.asks_identifier and (field_norm == "id" or field_norm.endswith("_id") or "primary_key" in semantic_tags):
        return 0.0
    source_bonus = 0.0
    if source.source_id in scoped_source_ids:
        source_bonus += 0.15
    if source.source_id in join_source_ids:
        source_bonus += 0.05
    if _source_matches_entity_hint(source, field, request.entity_hint):
        source_bonus += 0.28
    graph_bonus = min(candidate.score, 3.0) * 0.04 if candidate is not None else 0.0
    if request.asks_identifier:
        if field_norm == "id" or field_norm.endswith("_id") or "primary_key" in semantic_tags:
            return 1.0 + source_bonus + graph_bonus
        return 0.0
    if request.asks_measure:
        measure_score = _measure_field_score(field_norm, field)
        return measure_score + source_bonus + graph_bonus if measure_score > 0.0 else 0.0
    attribute_score = _attribute_field_score(request, field_norm, field)
    return attribute_score + source_bonus + graph_bonus if attribute_score > 0.0 else 0.0


def _best_artifact_field_match(
    *,
    requested_term: str,
    artifact: SemanticArtifact,
    scoped_source_ids: set[str],
    graph_join_sources: set[str],
) -> tuple[SourceDescriptor, SourceField, float] | None:
    requested_norm = normalize_identifier(requested_term)
    requested_aliases = _requested_term_aliases(requested_norm)
    best: tuple[float, SourceDescriptor, SourceField] | None = None
    for source in artifact.sources:
        source_bonus = 0.0
        if source.source_id in scoped_source_ids:
            source_bonus += 0.2
        if source.source_id in graph_join_sources:
            source_bonus += 0.08
        for field in source.fields:
            score = _artifact_field_match_score(requested_aliases, requested_term, field) + source_bonus
            if score <= 0.0:
                continue
            if best is None or score > best[0]:
                best = (score, source, field)
    if best is None or best[0] < 0.74:
        return None
    return best[1], best[2], min(best[0], 0.97)


def _artifact_field_match_score(
    requested_aliases: set[str],
    requested_term: str,
    field: SourceField,
) -> float:
    field_norm = normalize_identifier(field.field_name)
    if (field_norm == "id" or field_norm.endswith("_id")) and not ({"id", "identifier"} & requested_aliases):
        return 0.0
    field_aliases = {field_norm}
    field_aliases.update(normalize_identifier(alias) for alias in field.aliases if alias)
    field_aliases.add(_field_answer_alias(field_norm))
    best = 0.0
    for candidate in field_aliases:
        if not candidate:
            continue
        if candidate in requested_aliases or any(alias == candidate for alias in requested_aliases):
            best = max(best, 1.0)
            continue
        if any(candidate in alias or alias in candidate for alias in requested_aliases):
            best = max(best, 0.9)
            continue
        overlap = token_overlap_score(requested_term, candidate)
        if overlap > 0:
            best = max(best, min(0.55 + (0.12 * overlap), 0.88))
    semantic_tags = {normalize_identifier(tag) for tag in field.semantic_tags}
    if field_norm == "id" or "primary_key" in semantic_tags:
        if {"id", "identifier"} & requested_aliases:
            best = max(best, 1.0)
    return best


def _measure_field_score(field_norm: str, field: SourceField) -> float:
    if field_norm in {"cost", "amount", "spent", "price", "value", "total"}:
        return 1.0
    alias_norms = {normalize_identifier(alias) for alias in field.aliases if alias}
    if alias_norms & {"cost", "amount", "spent", "price", "value", "total"}:
        return 0.96
    dtype_norm = normalize_identifier(field.dtype)
    if dtype_norm in {"int", "integer", "float", "real", "double", "decimal", "number", "float64"}:
        return 0.74
    return 0.0


def _attribute_field_score(
    request: _AnswerRequest,
    field_norm: str,
    field: SourceField,
) -> float:
    requested_aliases = set(request.requested_aliases)
    base = _artifact_field_match_score(requested_aliases, request.phrase, field)
    if "type" in requested_aliases:
        semantic_rank = {
            "expense_description": 0.97,
            "description": 0.95,
            "category": 0.92,
            "type": 0.84,
            "name": 0.72,
        }
        base = max(base, semantic_rank.get(field_norm, 0.0))
    if {"disease", "diagnosis"} & requested_aliases and field_norm == "diagnosis":
        base = max(base, 0.98)
    if {"sex", "gender"} & requested_aliases and field_norm == "sex":
        base = max(base, 0.98)
    return base


def _source_matches_entity_hint(
    source: SourceDescriptor,
    field: SourceField,
    entity_hint: str | None,
) -> bool:
    if not entity_hint:
        return False
    candidates = {
        normalize_identifier(source.source_id),
        normalize_identifier(source.object_name),
        normalize_identifier(source.asset_path),
        normalize_identifier(field.field_name),
    }
    return any(entity_hint in candidate for candidate in candidates if candidate)


def _requested_term_aliases(normalized_term: str) -> set[str]:
    aliases = {normalized_term}
    token_variants = normalized_term.replace("_", " ").split()
    aliases.update(token_variants)
    alias_map = {
        "id": {"identifier", "patient_id"},
        "identifier": {"id"},
        "sex": {"gender"},
        "gender": {"sex"},
        "diagnosis": {"disease"},
        "disease": {"diagnosis"},
    }
    for token in list(aliases):
        aliases.update(alias_map.get(token, set()))
    return {alias for alias in aliases if alias}


def _boolean_question_filters(
    *,
    question: str,
    artifact: SemanticArtifact,
    graph_candidates,
    source_scopes: tuple[SourceScope, ...],
    answer_slots: tuple[RoutingFieldRef, ...],
) -> tuple[RoutingFilter, ...]:
    question_norm = normalize_identifier(question)
    answer_keys = {(slot.source_id, slot.field_name) for slot in answer_slots}
    graph_index = {
        (candidate.source_id, candidate.field_name): candidate
        for candidate in graph_candidates.field_candidates
        if candidate.source_id and candidate.field_name
    }
    scoped_ids = {scope.source_id for scope in source_scopes}
    filters: list[RoutingFilter] = []
    for source in artifact.sources:
        if source.source_id not in scoped_ids:
            continue
        for field in source.fields:
            key = (source.source_id, field.field_name)
            if key in answer_keys:
                continue
            if not _looks_boolean_field(field):
                continue
            field_aliases = {normalize_identifier(field.field_name)}
            field_aliases.update(normalize_identifier(alias) for alias in field.aliases if alias)
            matched_alias = next((alias for alias in field_aliases if alias and alias in question_norm), None)
            if matched_alias is None:
                continue
            value = "false" if f"not_{matched_alias}" in question_norm or f"non_{matched_alias}" in question_norm else "true"
            candidate = graph_index.get(key)
            filters.append(
                RoutingFilter(
                    source_id=source.source_id,
                    field_name=field.field_name,
                    operator="=",
                    value=value,
                    confidence=max(candidate.score if candidate is not None else 0.0, 0.83),
                    rationale="boolean_question_field",
                    graph_node_id=candidate.node_id if candidate is not None else f"field::{source.source_id}.{field.field_name}",
                )
            )
    deduped: dict[tuple[str, str, str, str], RoutingFilter] = {}
    for item in filters:
        deduped.setdefault((item.source_id, item.field_name, item.operator, item.value), item)
    return tuple(deduped.values())


def _looks_boolean_field(field: SourceField) -> bool:
    dtype_norm = normalize_identifier(field.dtype)
    if dtype_norm in {"bool", "boolean"}:
        return True
    sample_values = {normalize_identifier(value) for value in field.sample_values if str(value).strip()}
    if not sample_values:
        return False
    return sample_values <= {"true", "false", "yes", "no", "0", "1"}


def _joins_from_candidates(graph_candidates) -> tuple[JoinEdge, ...]:
    joins: list[JoinEdge] = []
    for candidate in graph_candidates.join_candidates:
        join = _join_edge_from_graph_candidate(candidate)
        if join is not None:
            joins.append(join)
    return tuple(joins)


def _field_answer_alias(field_name: str) -> str:
    aliases = {
        "diagnosis": "disease",
        "sex": "gender",
        "id": "identifier",
    }
    return aliases.get(field_name, field_name)


_MONTH_INDEX = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def _explicit_month_phrase(question_text: str) -> str | None:
    for month in _MONTH_INDEX:
        if re.search(rf"\b(?:in|during|from|for)\s+{month}\b", question_text):
            return month
        if re.search(rf"\b{month},\s*(?:19|20)\d{{2}}\b", question_text):
            return month
    return None


def _string_metadata(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None
