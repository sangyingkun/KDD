from __future__ import annotations

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.semantic.business_grounder import ground_business_context
from data_agent_competition.semantic.grain_resolver import resolve_target_grain
from data_agent_competition.semantic.lightrag_query_adapter import query_semantic_graph_with_lightrag
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.semantic.logical_planner import synthesize_logical_plan
from data_agent_competition.semantic.question_typer import classify_question
from data_agent_competition.semantic.routing_builder import build_routing_spec, routing_selections
from data_agent_competition.semantic.types import LogicalPlan, SemanticRoutingSpec, TaskBundle
from data_agent_competition.semantic.verifier import VerificationResult, verify_logical_plan


def analyze_task_semantics(
    task: TaskBundle,
    artifact: SemanticArtifact,
    runtime: SemanticRuntime,
) -> tuple[SemanticRoutingSpec, LogicalPlan, VerificationResult]:
    question_type = classify_question(task)
    lightrag_result = query_semantic_graph_with_lightrag(
        task=task,
        artifact=artifact,
        embedding_provider=runtime.embedding_provider,
    )
    graph_candidates = lightrag_result.candidates
    retrieval_context = lightrag_result.retrieval.context_sections
    grounding = ground_business_context(
        task,
        artifact,
        question_type,
        runtime,
        graph_candidates,
        retrieval_context,
    )
    routing_spec = build_routing_spec(
        task=task,
        artifact=artifact,
        question_type=question_type,
        grounding=grounding,
        graph_candidates=graph_candidates,
        retrieval_notes=lightrag_result.retrieval.notes,
    )
    target_grain = resolve_target_grain(
        task,
        artifact,
        routing_spec.target_sources,
        routing_selections(routing_spec),
        graph_candidates,
        routing_spec.metrics,
    )
    logical_plan = synthesize_logical_plan(
        task=task,
        artifact=artifact,
        routing_spec=routing_spec,
        target_grain=target_grain,
        runtime=runtime,
        graph_candidates=graph_candidates,
        retrieval_context=retrieval_context,
    )
    verification = verify_logical_plan(
        task=task,
        plan=logical_plan,
        artifact=artifact,
        routing_spec=routing_spec,
    )
    return routing_spec, logical_plan, verification
