from __future__ import annotations

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.semantic.graph_types import GraphCandidateSet
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.semantic.schemas import (
    aggregation_function_enum,
    filter_operator_enum,
    join_type_enum,
    logical_plan_override_schema,
    ordering_direction_enum,
)
from data_agent_competition.semantic.types import (
    ExecutionHint,
    JoinEdge,
    LogicalAggregation,
    LogicalFilter,
    LogicalOrdering,
    LogicalPlan,
    PostSQLEnrichment,
    QuestionType,
    RoutingAnswerKind,
    SemanticRoutingSpec,
    SemanticSelection,
    SelectionRole,
    TargetGrain,
    TaskBundle,
)


def synthesize_logical_plan(
    *,
    task: TaskBundle,
    artifact: SemanticArtifact,
    routing_spec: SemanticRoutingSpec,
    target_grain: TargetGrain,
    runtime: SemanticRuntime,
    graph_candidates: GraphCandidateSet,
    retrieval_context: tuple[str, ...] = (),
) -> LogicalPlan:
    selections = _selections_from_routing(routing_spec)
    filters = _logical_filters(routing_spec)
    aggregations = _aggregations_from_routing(routing_spec, selections)
    answer_columns = _answer_columns(routing_spec, aggregations)
    answer_aliases = _answer_aliases(answer_columns)
    llm_overrides = _llm_plan_overrides(
        task=task,
        routing_spec=routing_spec,
        graph_candidates=graph_candidates,
        runtime=runtime,
        retrieval_context=retrieval_context,
    )
    if llm_overrides["answer_columns"]:
        answer_columns = llm_overrides["answer_columns"]
        answer_aliases = _answer_aliases(answer_columns)
    if llm_overrides["filters"]:
        filters = llm_overrides["filters"]
    if llm_overrides["aggregations"]:
        aggregations = llm_overrides["aggregations"]
    joins = llm_overrides["joins"] or routing_spec.join_path
    orderings = llm_overrides["orderings"]
    return LogicalPlan(
        task_id=task.task_id,
        question_type=routing_spec.question_type,
        target_grain=target_grain,
        sources=routing_spec.target_sources,
        joins=joins,
        selections=selections,
        filters=tuple(_deduplicate_filters(list(filters))),
        aggregations=aggregations,
        orderings=orderings,
        answer_columns=answer_columns,
        execution_hint=_execution_hint(routing_spec.target_sources),
        limit=None,
        notes=tuple(artifact.notes[:3]) + routing_spec.notes,
        verification_focus=(
            "routing_spec_completeness",
            "answer_shape",
            "target_grain",
            "join_plausibility",
            "graph_consistency",
        ),
        answer_aliases=answer_aliases,
        post_sql_enrichments=routing_spec.post_sql_enrichments,
    )


def _selections_from_routing(routing_spec: SemanticRoutingSpec) -> tuple[SemanticSelection, ...]:
    selections: list[SemanticSelection] = []
    for slot in routing_spec.answer_slots:
        role = SelectionRole.ANSWER
        if slot.answer_kind == RoutingAnswerKind.MEASURE:
            role = SelectionRole.MEASURE
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
    for routing_filter in (*routing_spec.filters, *routing_spec.time_constraints):
        selections.append(
            SemanticSelection(
                source_id=routing_filter.source_id,
                field_name=routing_filter.field_name,
                role=SelectionRole.FILTER,
                alias=routing_filter.field_name,
                confidence=routing_filter.confidence,
                rationale=routing_filter.rationale,
                graph_node_id=routing_filter.graph_node_id,
            )
        )
    for join in routing_spec.join_path:
        selections.append(
            SemanticSelection(
                source_id=join.left_source_id,
                field_name=join.left_field,
                role=SelectionRole.JOIN_KEY,
                alias=join.left_field,
                confidence=join.confidence,
                rationale=join.rationale,
            )
        )
        selections.append(
            SemanticSelection(
                source_id=join.right_source_id,
                field_name=join.right_field,
                role=SelectionRole.JOIN_KEY,
                alias=join.right_field,
                confidence=join.confidence,
                rationale=join.rationale,
            )
        )
    return tuple(_dedupe_selections(selections))


def _logical_filters(routing_spec: SemanticRoutingSpec) -> tuple[LogicalFilter, ...]:
    return tuple(
        LogicalFilter(
            source_id=item.source_id,
            field_name=item.field_name,
            operator=item.operator,
            value=item.value,
            rationale=item.rationale,
        )
        for item in (*routing_spec.filters, *routing_spec.time_constraints)
    )


def _aggregations_from_routing(
    routing_spec: SemanticRoutingSpec,
    selections: tuple[SemanticSelection, ...],
) -> tuple[LogicalAggregation, ...]:
    aggregations: list[LogicalAggregation] = []
    for metric in routing_spec.metrics:
        if metric.source_fields:
            source_id, field_name = metric.source_fields[0].split(".", maxsplit=1)
            aggregations.append(
                LogicalAggregation(
                    source_id=source_id,
                    field_name=field_name,
                    function="sum" if _looks_aggregate(metric.label) else "identity",
                    alias=metric.label,
                )
            )
    if aggregations:
        return tuple(aggregations)
    measure_slots = [selection for selection in selections if selection.role == SelectionRole.MEASURE]
    if not measure_slots:
        return ()
    function = "count" if routing_spec.question_type == QuestionType.AGGREGATION and any(
        slot.field_name.lower() == "count" for slot in measure_slots
    ) else "sum"
    measure = measure_slots[0]
    return (
        LogicalAggregation(
            source_id=measure.source_id,
            field_name=measure.field_name,
            function=function,
            alias=None,
        ),
    )


def _answer_columns(
    routing_spec: SemanticRoutingSpec,
    aggregations: tuple[LogicalAggregation, ...],
) -> tuple[str, ...]:
    qualified = [
        f"{slot.source_id}.{slot.field_name}"
        for slot in routing_spec.answer_slots
        if slot.answer_kind in {RoutingAnswerKind.IDENTIFIER, RoutingAnswerKind.ATTRIBUTE, RoutingAnswerKind.GROUP_BY}
    ]
    for aggregation in aggregations:
        qualified.append(f"{aggregation.source_id}.{aggregation.field_name}")
    if qualified:
        return tuple(dict.fromkeys(qualified))
    fallback = [f"{slot.source_id}.{slot.field_name}" for slot in routing_spec.answer_slots[:3]]
    return tuple(dict.fromkeys(fallback))


def _answer_aliases(answer_columns: tuple[str, ...]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for column in answer_columns:
        field_name = column.split(".")[-1]
        aliases[column] = "Disease" if field_name.lower() == "diagnosis" else field_name
    return aliases


def _llm_plan_overrides(
    *,
    task: TaskBundle,
    routing_spec: SemanticRoutingSpec,
    graph_candidates: GraphCandidateSet,
    runtime: SemanticRuntime,
    retrieval_context: tuple[str, ...],
) -> dict[str, tuple]:
    payload = runtime.llm_client.call_structured(
        system_prompt=(
            "You are a constrained logical planner. "
            "Use only the routing spec, listed fields, listed joins, and listed metrics. "
            "Do not invent sources, fields, joins, filters, or aggregations. "
            "If uncertain, return empty arrays."
        ),
        user_prompt=_planner_prompt(task, routing_spec, retrieval_context),
        function_name="synthesize_logical_plan",
        schema=logical_plan_override_schema(),
    )
    if not payload:
        return {
            "answer_columns": (),
            "filters": (),
            "joins": (),
            "aggregations": (),
            "orderings": (),
        }

    valid_fields = {
        (candidate.source_id, candidate.field_name)
        for candidate in graph_candidates.field_candidates
        if candidate.source_id and candidate.field_name
    }
    valid_joins = {
        (join.left_source_id, join.left_field, join.right_source_id, join.right_field)
        for join in routing_spec.join_path
    }
    return {
        "answer_columns": tuple(
            value
            for value in (str(item).strip() for item in payload.get("answer_columns", []))
            if "." in value and tuple(value.rsplit(".", 1)) in valid_fields
        ),
        "filters": tuple(
            LogicalFilter(
                source_id=str(item["source_id"]).strip(),
                field_name=str(item["field_name"]).strip(),
                operator=str(item["operator"]).strip(),
                value=str(item["value"]),
                rationale=str(item.get("rationale", "")).strip(),
            )
            for item in payload.get("filters", [])
            if (
                (str(item.get("source_id", "")).strip(), str(item.get("field_name", "")).strip()) in valid_fields
                and str(item.get("operator", "")).strip() in filter_operator_enum()
            )
        ),
        "joins": tuple(
            JoinEdge(
                left_source_id=str(item["left_source_id"]).strip(),
                left_field=str(item["left_field"]).strip(),
                right_source_id=str(item["right_source_id"]).strip(),
                right_field=str(item["right_field"]).strip(),
                join_type=str(item.get("join_type", "inner")).strip() or "inner",
                confidence=0.95,
                rationale=str(item.get("rationale", "")).strip() or "llm_join",
            )
            for item in payload.get("joins", [])
            if (
                str(item.get("left_source_id", "")).strip(),
                str(item.get("left_field", "")).strip(),
                str(item.get("right_source_id", "")).strip(),
                str(item.get("right_field", "")).strip(),
            )
            in valid_joins
            and str(item.get("join_type", "inner")).strip() in join_type_enum()
        ),
        "aggregations": tuple(
            LogicalAggregation(
                source_id=str(item["source_id"]).strip(),
                field_name=str(item["field_name"]).strip(),
                function=str(item["function"]).strip().lower(),
                alias=_nullable(item.get("alias")),
            )
            for item in payload.get("aggregations", [])
            if (
                (str(item.get("source_id", "")).strip(), str(item.get("field_name", "")).strip()) in valid_fields
                and str(item.get("function", "")).strip().lower() in aggregation_function_enum()
            )
        ),
        "orderings": tuple(
            LogicalOrdering(
                source_id=str(item["source_id"]).strip(),
                field_name=str(item["field_name"]).strip(),
                direction=str(item.get("direction", "asc")).strip() or "asc",
            )
            for item in payload.get("orderings", [])
            if (
                (str(item.get("source_id", "")).strip(), str(item.get("field_name", "")).strip()) in valid_fields
                and str(item.get("direction", "asc")).strip() in ordering_direction_enum()
            )
        ),
    }


def _planner_prompt(
    task: TaskBundle,
    routing_spec: SemanticRoutingSpec,
    retrieval_context: tuple[str, ...],
) -> str:
    sources = "\n".join(
        f"- {scope.source_id} ({scope.source_kind.value})"
        for scope in routing_spec.target_sources
    )
    fields = "\n".join(
        f"- {slot.source_id}.{slot.field_name} kind={slot.answer_kind.value}"
        for slot in routing_spec.answer_slots
    )
    filters = "\n".join(
        f"- {item.source_id}.{item.field_name} {item.operator} {item.value}"
        for item in (*routing_spec.filters, *routing_spec.time_constraints)
    )
    joins = "\n".join(
        f"- {join.left_source_id}.{join.left_field} = {join.right_source_id}.{join.right_field}"
        for join in routing_spec.join_path
    )
    metrics = "\n".join(
        f"- {metric.label}: formula={metric.formula or 'n/a'} fields={list(metric.source_fields)}"
        for metric in routing_spec.metrics
    )
    return (
        f"Question: {task.question}\n"
        "Retrieved graph context:\n"
        f"{chr(10).join(retrieval_context) if retrieval_context else '- none'}\n"
        "Routing sources:\n"
        f"{sources if sources else '- none'}\n"
        "Routing answer slots:\n"
        f"{fields if fields else '- none'}\n"
        "Routing filters:\n"
        f"{filters if filters else '- none'}\n"
        "Routing joins:\n"
        f"{joins if joins else '- none'}\n"
        "Routing metrics:\n"
        f"{metrics if metrics else '- none'}\n"
        "Rules:\n"
        "- answer_columns must come from routing answer slots\n"
        "- joins must come from routing joins\n"
        "- filters must come from routing filters\n"
        "- aggregations must use routing metrics or measure slots\n"
        "- do not invent fields, joins, or metrics"
    )


def _execution_hint(scoped_sources) -> ExecutionHint:
    kinds = {scope.source_kind.value for scope in scoped_sources}
    if kinds == {"db"}:
        return ExecutionHint.SQL_ONLY
    if len(kinds) == 1 and kinds <= {"csv", "json", "doc", "knowledge"}:
        return ExecutionHint.PYTHON_ONLY
    return ExecutionHint.HYBRID


def _deduplicate_filters(filters: list[LogicalFilter]) -> list[LogicalFilter]:
    deduped: dict[tuple[str, str, str, str], LogicalFilter] = {}
    for item in filters:
        deduped[(item.source_id, item.field_name, item.operator, item.value)] = item
    return list(deduped.values())


def _dedupe_selections(selections: list[SemanticSelection]) -> list[SemanticSelection]:
    best: dict[tuple[str, str, SelectionRole], SemanticSelection] = {}
    for selection in selections:
        key = (selection.source_id, selection.field_name, selection.role)
        previous = best.get(key)
        if previous is None or selection.confidence > previous.confidence:
            best[key] = selection
    return list(best.values())


def _looks_aggregate(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in ("total", "sum", "average", "avg", "count", "ratio", "utilization"))


def _nullable(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None
