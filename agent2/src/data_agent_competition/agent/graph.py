from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from data_agent_competition.agent.replan_policy import decide_replan
from data_agent_competition.agent.state import ControllerState
from data_agent_competition.execution.executor import execute_physical_plan
from data_agent_competition.execution.physical_planner import build_physical_plan
from data_agent_competition.runtime.config import CompetitionConfig
from data_agent_competition.semantic.artifact_loader import load_task_artifact
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.semantic.pipeline import analyze_task_semantics
from data_agent_competition.semantic.verifier import verify_execution_outcome

NodeHandler = Callable[[ControllerState, CompetitionConfig], ControllerState]


def run_controller_graph(
    state: ControllerState,
    config: CompetitionConfig,
    runtime: SemanticRuntime,
) -> ControllerState:
    if _langgraph_available():
        state.orchestration_backend = "langgraph"
        return _run_langgraph(state, config, runtime)
    state.orchestration_backend = "sequential"
    return _run_sequential(state, config, runtime)


def _run_sequential(
    state: ControllerState,
    config: CompetitionConfig,
    runtime: SemanticRuntime,
) -> ControllerState:
    state = _load_artifact(state, config, runtime)
    if state.failure_reason is not None:
        return _finish(state, config)

    while True:
        if _step_budget_exceeded(state, config):
            return _finish(state, config)
        state = _analyze_semantics(state, config, runtime)
        if _step_budget_exceeded(state, config):
            return _finish(state, config)
        state = _plan_execution(state, config)
        if _step_budget_exceeded(state, config):
            return _finish(state, config)
        state = _execute_plan(state, config)
        if _step_budget_exceeded(state, config):
            return _finish(state, config)
        state = _verify_result(state, config)
        if _step_budget_exceeded(state, config):
            return _finish(state, config)
        state = _bounded_replan(state, config)
        if state.status != "retrying_semantics":
            break
    return _finish(state, config)


def _run_langgraph(
    state: ControllerState,
    config: CompetitionConfig,
    runtime: SemanticRuntime,
) -> ControllerState:
    try:
        from langgraph.errors import GraphRecursionError
        from langgraph.graph import END, StateGraph
    except Exception:
        state.orchestration_backend = "sequential"
        return _run_sequential(state, config, runtime)

    graph = StateGraph(ControllerState)
    graph.add_node("load_artifact", lambda current: _load_artifact(current, config, runtime))
    graph.add_node("analyze_semantics", lambda current: _analyze_semantics(current, config, runtime))
    graph.add_node("plan_execution", lambda current: _plan_execution(current, config))
    graph.add_node("execute_plan", lambda current: _execute_plan(current, config))
    graph.add_node("verify_result", lambda current: _verify_result(current, config))
    graph.add_node("bounded_replan", lambda current: _bounded_replan(current, config))
    graph.add_node("finish", lambda current: _finish(current, config))

    graph.set_entry_point("load_artifact")
    graph.add_conditional_edges(
        "load_artifact",
        lambda current: "finish" if current.failure_reason is not None else "analyze_semantics",
    )
    graph.add_edge("analyze_semantics", "plan_execution")
    graph.add_edge("plan_execution", "execute_plan")
    graph.add_edge("execute_plan", "verify_result")
    graph.add_edge("verify_result", "bounded_replan")
    graph.add_conditional_edges(
        "bounded_replan",
        lambda current: "analyze_semantics" if current.status == "retrying_semantics" else "finish",
    )
    graph.add_edge("finish", END)
    compiled = graph.compile()
    try:
        result = compiled.invoke(
            state,
            config={"recursion_limit": max(int(config.runtime.graph_recursion_limit), 1)},
        )
    except GraphRecursionError:
        state.failure_reason = (
            f"controller hit LangGraph recursion limit ({config.runtime.graph_recursion_limit})"
        )
        state.status = "controller_recursion_limit"
        state.record_step(
            "graph_recursion_limit",
            status=state.status,
            failure_reason=state.failure_reason,
            recursion_limit=config.runtime.graph_recursion_limit,
        )
        return _finish(state, config)
    if isinstance(result, ControllerState):
        return result
    if isinstance(result, dict):
        return _state_from_mapping(result, fallback=state)
    raise TypeError(f"Unexpected LangGraph state type: {type(result)!r}")


def _load_artifact(
    state: ControllerState,
    config: CompetitionConfig,
    runtime: SemanticRuntime,
) -> ControllerState:
    artifact_result = load_task_artifact(state.task, runtime=runtime)
    artifact = artifact_result.artifact
    state.semantic_artifact = artifact
    state.semantic_artifact_mode = artifact_result.mode
    state.semantic_artifact_path = artifact_result.artifact_path
    state.status = "artifact_loaded"
    state.record_step(
        "load_artifact",
        status=state.status,
        artifact_mode=artifact_result.mode,
        artifact_path=artifact_result.artifact_path,
        asset_count=len(artifact.assets),
        source_count=len(artifact.sources),
        knowledge_fact_count=len(artifact.knowledge_facts),
    )
    return state


def _analyze_semantics(
    state: ControllerState,
    config: CompetitionConfig,
    runtime: SemanticRuntime,
) -> ControllerState:
    if state.semantic_artifact is None:
        state.failure_reason = "semantic artifact not loaded"
        state.status = "semantic_failed"
        return state
    semantic_routing, logical_plan, logical_verification = analyze_task_semantics(
        state.task,
        state.semantic_artifact,
        runtime,
    )
    state.semantic_attempts += 1
    state.semantic_routing = semantic_routing
    state.logical_plan = logical_plan
    state.logical_verification = logical_verification
    state.status = "semantic_planned"
    state.record_step(
        "analyze_semantics",
        status=state.status,
        semantic_attempts=state.semantic_attempts,
        semantic_routing=semantic_routing.to_dict(),
        logical_verification=logical_verification.to_dict(),
        llm_enabled=runtime.llm_client.enabled,
        llm_disabled_reason=runtime.llm_client.disabled_reason,
    )
    return state


def _plan_execution(state: ControllerState, config: CompetitionConfig) -> ControllerState:
    if state.logical_plan is None:
        state.failure_reason = "logical plan missing"
        state.status = "planning_failed"
        return state
    state.physical_plan = build_physical_plan(state.logical_plan)
    state.status = "physical_planned"
    state.record_step(
        "plan_execution",
        status=state.status,
        stage_count=len(state.physical_plan.stages),
        answer_columns=list(state.physical_plan.answer_columns),
    )
    return state


def _execute_plan(state: ControllerState, config: CompetitionConfig) -> ControllerState:
    if state.semantic_artifact is None or state.logical_plan is None or state.physical_plan is None:
        state.failure_reason = "execution prerequisites missing"
        state.status = "execution_failed"
        return state
    execution_result, answer_table = execute_physical_plan(
        state.task,
        state.semantic_artifact,
        state.logical_plan,
        state.physical_plan,
    )
    state.execution_result = execution_result
    state.answer_table = answer_table
    state.failure_reason = execution_result.failure_reason
    state.status = "executed" if execution_result.succeeded else "execution_failed"
    state.record_step(
        "execute_plan",
        status=state.status,
        execution=execution_result.to_dict(),
    )
    return state


def _verify_result(state: ControllerState, config: CompetitionConfig) -> ControllerState:
    state.final_verification = verify_execution_outcome(
        logical_plan=state.logical_plan,
        execution_result=state.execution_result,
        answer_table=state.answer_table,
    )
    if state.final_verification.ok and state.failure_reason is None:
        state.status = "verified"
    else:
        state.status = "verification_failed"
    state.record_step(
        "verify_result",
        status=state.status,
        final_verification=state.final_verification.to_dict(),
    )
    return state


def _bounded_replan(state: ControllerState, config: CompetitionConfig) -> ControllerState:
    decision = decide_replan(
        logical_plan=state.logical_plan,
        logical_verification=state.logical_verification,
        final_verification=state.final_verification,
        execution_result=state.execution_result,
        semantic_attempts=state.semantic_attempts,
        max_semantic_retries=config.agent.max_semantic_retries,
    )
    state.failure_signature = decision.signature
    state.record_step(
        "bounded_replan",
        status=state.status,
        decision=asdict(decision),
    )
    if decision.should_replan:
        state.status = "retrying_semantics"
        state.failure_reason = None
        state.semantic_routing = None
        state.logical_plan = None
        state.logical_verification = None
        state.physical_plan = None
        state.execution_result = None
        state.answer_table = None
        state.final_verification = None
        return state
    if state.failure_reason is None and state.final_verification is not None and not state.final_verification.ok:
        primary_issue = state.final_verification.primary_issue
        state.failure_reason = primary_issue.message if primary_issue is not None else "verification failed"
    return state


def _finish(state: ControllerState, config: CompetitionConfig) -> ControllerState:
    if state.failure_reason is None and state.execution_result is not None and state.execution_result.succeeded:
        state.status = "succeeded"
    elif state.failure_reason is None and state.status not in {"succeeded", "verified"}:
        state.failure_reason = "controller finished without a successful execution result"
        state.status = "failed"
    else:
        state.status = "failed" if state.failure_reason is not None else state.status
    state.record_step(
        "finish",
        status=state.status,
        failure_reason=state.failure_reason,
        orchestration_backend=state.orchestration_backend,
    )
    return state


def _step_budget_exceeded(state: ControllerState, config: CompetitionConfig) -> bool:
    limit = max(int(config.runtime.graph_recursion_limit), 1)
    if len(state.trace_steps) < limit:
        return False
    state.failure_reason = f"controller step budget exhausted ({limit})"
    state.status = "controller_step_limit"
    state.record_step(
        "controller_step_limit",
        status=state.status,
        failure_reason=state.failure_reason,
        recursion_limit=limit,
    )
    return True


def _langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
    except Exception:
        return False
    return True


def _state_from_mapping(payload: dict, *, fallback: ControllerState) -> ControllerState:
    state = ControllerState(task=payload.get("task", fallback.task))
    for field_name in (
        "semantic_artifact",
        "semantic_artifact_mode",
        "semantic_artifact_path",
        "semantic_routing",
        "logical_plan",
        "logical_verification",
        "physical_plan",
        "execution_result",
        "answer_table",
        "final_verification",
        "status",
        "failure_reason",
        "failure_signature",
        "semantic_attempts",
        "trace_steps",
        "orchestration_backend",
    ):
        if field_name in payload:
            setattr(state, field_name, payload[field_name])
    return state
