from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.observation import (
    enrich_observation_with_plan,
    merge_runtime_feedback,
    prune_observation,
    replan_feedback_message,
    route_dependency_feedback,
    route_mismatch_feedback,
    route_tool_mismatch_feedback,
)
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import PROFILE_AGENT_CORE, ToolExecutionResult, ToolRegistry

logger = logging.getLogger("dabench.react")

_SOURCE_BOUND_ACTIONS = {
    "execute_context_sql",
    "inspect_sqlite_schema",
    "read_csv",
    "read_doc",
    "read_json",
}

_ROUTE_ACTIONS_BY_SOURCE_TYPE = {
    "sqlite": {"inspect_sqlite_schema", "execute_context_sql"},
    "csv": {"read_csv"},
    "json": {"read_json"},
    "document": {"read_doc"},
}


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 20
    circuit_breaker_threshold: int = 2
    enable_function_calling: bool = True
    allow_text_fallback_when_tools_missing: bool = False


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    @staticmethod
    def _maybe_update_plan_snapshot(
        state: AgentRuntimeState,
        *,
        action: str,
        ok: bool,
        content: Any,
    ) -> None:
        if action != "plan_semantic_query" or not ok or not isinstance(content, dict):
            return
        state.latest_plan_snapshot = dict(content)
        routing_plan = content.get("routing_plan", [])
        if isinstance(routing_plan, list):
            state.latest_routing_plan = [
                dict(step)
                for step in routing_plan
                if isinstance(step, dict)
            ]
        else:
            state.latest_routing_plan = []
        state.completed_route_sources = []

    def _bootstrap_semantic_context(self, task: PublicTask, state: AgentRuntimeState) -> None:
        bootstrap_actions = (
            ("describe_semantics", {"max_items_per_section": 8, "include_evidence": True}),
            ("link_schema_candidates", {"question": task.question}),
            ("plan_semantic_query", {"question": task.question}),
        )
        for action, action_input in bootstrap_actions:
            try:
                tool_result = self.tools.execute(task, action, action_input)
            except Exception as exc:
                state.steps.append(
                    StepRecord(
                        step_index=len(state.steps) + 1,
                        thought="Semantic bootstrap failed.",
                        action=action,
                        action_input=action_input,
                        raw_response=json.dumps({
                            "thought": "Semantic bootstrap failed.",
                            "action": action,
                            "action_input": action_input,
                        }),
                        observation=prune_observation(action, ok=False, content=str(exc)),
                        ok=False,
                    )
                )
                continue
            observation = prune_observation(action, ok=tool_result.ok, content=tool_result.content)
            self._maybe_update_plan_snapshot(
                state,
                action=action,
                ok=tool_result.ok,
                content=tool_result.content,
            )
            state.steps.append(
                StepRecord(
                    step_index=len(state.steps) + 1,
                    thought="Bootstrap semantic context before free-form exploration.",
                    action=action,
                    action_input=action_input,
                    raw_response=json.dumps({
                        "thought": "Bootstrap semantic context before free-form exploration.",
                        "action": action,
                        "action_input": action_input,
                    }),
                    observation=observation,
                    ok=tool_result.ok,
                )
            )

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(PROFILE_AGENT_CORE),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    @staticmethod
    def _build_raw_response_from_tool_call(action: str, action_input: dict[str, Any]) -> str:
        return json.dumps(
            {
                "thought": "Structured tool call provided by the model.",
                "action": action,
                "action_input": action_input,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _action_source_ref(action: str, action_input: dict[str, Any]) -> str | None:
        if action not in _SOURCE_BOUND_ACTIONS:
            return None
        source_ref = action_input.get("path")
        if source_ref is None:
            return None
        source_text = str(source_ref).strip()
        return source_text or None

    @staticmethod
    def _action_source_type(action: str, action_input: dict[str, Any]) -> str | None:
        if action == "execute_context_sql" or action == "inspect_sqlite_schema":
            return "sqlite"
        source_ref = ReActAgent._action_source_ref(action, action_input)
        if source_ref is None:
            return None
        lowered_source = source_ref.lower()
        if lowered_source.endswith(".csv"):
            return "csv"
        if lowered_source.endswith(".json"):
            return "json"
        if lowered_source.endswith((".md", ".txt")):
            return "document"
        if lowered_source.endswith((".db", ".sqlite")):
            return "sqlite"
        if action == "read_doc":
            return "document"
        return None

    @staticmethod
    def _pending_route_steps(state: AgentRuntimeState) -> list[dict[str, Any]]:
        pending_steps: list[dict[str, Any]] = []
        completed = set(state.completed_route_sources)
        for step in state.latest_routing_plan:
            if not isinstance(step, dict):
                continue
            source_ref = str(step.get("source_ref", "")).strip()
            step_type = str(step.get("step_type", "")).strip()
            if step_type not in {"knowledge_check", "source_access"}:
                continue
            if source_ref and source_ref in completed:
                continue
            pending_steps.append(step)
        return pending_steps

    def _route_feedback_for_action(
        self,
        state: AgentRuntimeState,
        *,
        action: str,
        action_input: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if action not in _SOURCE_BOUND_ACTIONS or not state.latest_routing_plan:
            return None, None

        actual_source = self._action_source_ref(action, action_input)
        actual_source_type = self._action_source_type(action, action_input)
        if actual_source is None:
            return None, None

        pending_steps = self._pending_route_steps(state)
        if not pending_steps:
            return None, None

        current_step = pending_steps[0]
        expected_source = str(current_step.get("source_ref", "")).strip()
        expected_source_type = str(current_step.get("source_type", "")).strip()
        missing_dependencies = [
            str(item).strip()
            for item in current_step.get("depends_on", [])
            if str(item).strip() and str(item).strip() not in state.completed_route_sources
        ]
        if missing_dependencies:
            feedback = route_dependency_feedback(
                action=action,
                attempted_source=actual_source,
                missing_dependencies=missing_dependencies,
                current_step_type=str(current_step.get("step_type", "")).strip() or None,
            )
            return current_step, {
                "action": action,
                "planned_step": dict(current_step),
                "actual_source": actual_source,
                "actual_source_type": actual_source_type,
                "feedback": feedback,
            }

        expected_actions = sorted(
            _ROUTE_ACTIONS_BY_SOURCE_TYPE.get(expected_source_type, set())
        )
        if expected_actions and action not in expected_actions:
            feedback = route_tool_mismatch_feedback(
                action=action,
                current_step_type=str(current_step.get("step_type", "")).strip() or None,
                expected_actions=expected_actions,
                source_ref=expected_source or actual_source,
                source_type=expected_source_type or actual_source_type,
            )
            return current_step, {
                "action": action,
                "planned_step": dict(current_step),
                "actual_source": actual_source,
                "actual_source_type": actual_source_type,
                "feedback": feedback,
            }

        if expected_source and actual_source == expected_source:
            return current_step, None
        if expected_source_type and actual_source_type == expected_source_type and not expected_source:
            return current_step, None

        feedback = route_mismatch_feedback(
            action=action,
            expected_sources=[expected_source] if expected_source else [],
            actual_source=actual_source,
            expected_source_types=[expected_source_type] if expected_source_type else [],
            actual_source_type=actual_source_type,
            current_step_type=str(current_step.get("step_type", "")).strip() or None,
            join_anchor=str(current_step.get("join_anchor", "")).strip() or None,
        )
        return current_step, {
            "action": action,
            "planned_step": dict(current_step),
            "actual_source": actual_source,
            "actual_source_type": actual_source_type,
            "feedback": feedback,
        }

    @staticmethod
    def _mark_route_progress(
        state: AgentRuntimeState,
        *,
        action: str,
        action_input: dict[str, Any],
        route_step: dict[str, Any] | None,
    ) -> None:
        if action not in {"execute_context_sql", "read_csv", "read_doc", "read_json"}:
            return
        matched_step = route_step if isinstance(route_step, dict) else None
        if matched_step is None:
            return
        source_ref = str(matched_step.get("source_ref", "")).strip()
        if source_ref and source_ref not in state.completed_route_sources:
            state.completed_route_sources.append(source_ref)

    def _dead_end_signature(self, action: str, observation: dict[str, Any]) -> str | None:
        if action not in {"execute_context_sql", "execute_python", "read_csv", "inspect_sqlite_schema", "read_doc", "read_json"}:
            return None
        runtime_feedback = observation.get("runtime_feedback")
        if not isinstance(runtime_feedback, dict):
            return None
        signature = str(runtime_feedback.get("signature", "")).strip()
        secondary_signatures = {
            str(item).strip()
            for item in runtime_feedback.get("secondary_signatures", [])
            if str(item).strip()
        }
        effective_signatures = {signature, *secondary_signatures}
        tracked_signatures = {
            "execution_error",
            "empty_result",
            "all_null_result",
            "shape_mismatch",
            "route_mismatch",
            "route_dependency_mismatch",
            "route_tool_mismatch",
        }
        tracked = sorted(effective_signatures & tracked_signatures)
        if not tracked:
            return None
        return f"{action}:{'|'.join(tracked)}"

    def _trigger_replan(self, task: PublicTask, state: AgentRuntimeState, trigger_observation: dict[str, Any]) -> None:
        feedback = replan_feedback_message(trigger_observation)
        action = "plan_semantic_query"
        action_input = {"question": task.question, "feedback": feedback}
        try:
            tool_result = self.tools.execute(task, action, action_input)
            observation = prune_observation(action, ok=tool_result.ok, content=tool_result.content)
            self._maybe_update_plan_snapshot(
                state,
                action=action,
                ok=tool_result.ok,
                content=tool_result.content,
            )
        except Exception as exc:
            observation = prune_observation(action, ok=False, content=str(exc))
            tool_result = None

        state.steps.append(
            StepRecord(
                step_index=len(state.steps) + 1,
                thought="The current path failed repeatedly, so I am forcing a replanning step.",
                action=action,
                action_input=action_input,
                raw_response=json.dumps(
                    {
                        "thought": "The current path failed repeatedly, so I need a new plan.",
                        "action": action,
                        "action_input": action_input,
                    }
                ),
                observation=observation,
                ok=bool(tool_result.ok) if tool_result is not None else False,
            )
        )
        state.repeated_dead_end_count = 0
        state.last_dead_end_signature = None

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        try:
            self._bootstrap_semantic_context(task, state)
            run_start = time.perf_counter()
            logger.info(f"{'='*60}")
            logger.info(f"Task started | task_id={task.task_id} | difficulty={task.difficulty} | max_steps={self.config.max_steps}")
            logger.info(f"Question: {task.question[:150]}")
            logger.info(f"{'='*60}")

            for step_index in range(1, self.config.max_steps + 1):
                step_start = time.perf_counter()
                logger.info(f"\nStep {step_index}/{self.config.max_steps} starting...")
                try:
                    messages = self._build_messages(task, state)
                    agent_tools = self.tools.to_openai_tools_format(PROFILE_AGENT_CORE)
                    current_action = "__error__"
                    current_action_input: dict[str, Any] = {}
                    current_thought = ""
                    raw_response = ""
                    structured_mode = self.config.enable_function_calling and hasattr(self.model, "complete_with_tools")
                    if structured_mode:
                        turn = self.model.complete_with_tools(messages, agent_tools)  # type: ignore[attr-defined]
                        raw_response = turn.raw_response or turn.text_response or ""
                        if turn.tool_calls:
                            if len(turn.tool_calls) > 1:
                                logger.warning("Model returned %s tool calls; using the first one only.", len(turn.tool_calls))
                            tool_call = turn.tool_calls[0]
                            current_action = tool_call.name
                            current_action_input = dict(tool_call.arguments)
                            current_thought = "Structured tool call provided by the model."
                            raw_response = self._build_raw_response_from_tool_call(current_action, current_action_input)
                            logger.info(
                                "Structured tool call | action=%s | input=%s",
                                current_action,
                                json.dumps(current_action_input, ensure_ascii=False)[:300],
                            )
                            if current_action not in self.tools.get_specs_for_profile(PROFILE_AGENT_CORE):
                                raise ValueError(f"Tool '{current_action}' is not available in the agent_core profile.")
                        else:
                            if not self.config.allow_text_fallback_when_tools_missing:
                                current_action = "__missing_tool_call__"
                                current_thought = "The model did not follow the required tool-calling protocol."
                                raise ValueError(
                                    "Tool-capable model returned no structured tool call. "
                                    "Text fallback is disabled for this run."
                                )
                            raw_response = turn.text_response or turn.raw_response or ""
                            model_step = parse_model_step(raw_response)
                            current_action = model_step.action
                            current_action_input = dict(model_step.action_input)
                            current_thought = model_step.thought
                            logger.warning(
                                "Structured tool call missing; using compatibility text fallback for action %s.",
                                model_step.action,
                            )
                            logger.info(f"Thought: {model_step.thought[:200]}")
                            logger.info(f"Action: {model_step.action} | input={json.dumps(model_step.action_input, ensure_ascii=False)[:300]}")
                    else:
                        raw_response = self.model.complete(messages)
                        model_step = parse_model_step(raw_response)
                        current_action = model_step.action
                        current_action_input = dict(model_step.action_input)
                        current_thought = model_step.thought
                        logger.info(f"Thought: {model_step.thought[:200]}")
                        logger.info(f"Action: {model_step.action} | input={json.dumps(model_step.action_input, ensure_ascii=False)[:300]}")

                    matched_route_step, route_guard = self._route_feedback_for_action(
                        state,
                        action=current_action,
                        action_input=current_action_input,
                    )
                    if route_guard is not None:
                        feedback = route_guard["feedback"]
                        logger.warning(
                            "Route guard blocked tool execution | action=%s | actual_source=%s | planned_source=%s",
                            current_action,
                            route_guard.get("actual_source"),
                            route_guard["planned_step"].get("source_ref"),
                        )
                        observation = merge_runtime_feedback(
                            prune_observation(current_action, ok=False, content=feedback.summary),
                            feedback,
                        )
                        step_record = StepRecord(
                            step_index=step_index,
                            thought=current_thought,
                            action=current_action,
                            action_input=current_action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                        state.steps.append(step_record)
                        dead_end_signature = self._dead_end_signature(current_action, observation)
                        if dead_end_signature is None:
                            state.repeated_dead_end_count = 0
                            state.last_dead_end_signature = None
                        elif dead_end_signature == state.last_dead_end_signature:
                            state.repeated_dead_end_count += 1
                        else:
                            state.last_dead_end_signature = dead_end_signature
                            state.repeated_dead_end_count = 1

                        if state.repeated_dead_end_count >= self.config.circuit_breaker_threshold:
                            logger.warning(
                                "Circuit breaker triggered after repeated dead-end signature %s",
                                dead_end_signature,
                            )
                            state.route_replan_count += 1
                            self._trigger_replan(task, state, observation)
                        continue

                    tool_result = self.tools.execute(task, current_action, current_action_input)

                    step_elapsed = time.perf_counter() - step_start
                    if tool_result.ok:
                        result_preview = str(tool_result.content)[:300]
                        if len(str(tool_result.content)) > 300:
                            result_preview += "..."
                        logger.info(f"Tool result (OK) | elapsed={step_elapsed:.1f}s | preview={result_preview}")
                    else:
                        logger.warning(f"Tool execution failed | elapsed={step_elapsed:.1f}s | error={str(tool_result.content)[:200]}")

                    observation = prune_observation(current_action, ok=tool_result.ok, content=tool_result.content)
                    self._maybe_update_plan_snapshot(
                        state,
                        action=current_action,
                        ok=tool_result.ok,
                        content=tool_result.content,
                    )
                    observation = enrich_observation_with_plan(
                        observation,
                        action=current_action,
                        raw_content=tool_result.content,
                        plan_snapshot=state.latest_plan_snapshot,
                    )
                    self._mark_route_progress(
                        state,
                        action=current_action,
                        action_input=current_action_input,
                        route_step=matched_route_step,
                    )
                    step_record = StepRecord(
                        step_index=step_index,
                        thought=current_thought,
                        action=current_action,
                        action_input=current_action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                    state.steps.append(step_record)
                    dead_end_signature = self._dead_end_signature(current_action, observation)
                    if dead_end_signature is None:
                        state.repeated_dead_end_count = 0
                        state.last_dead_end_signature = None
                    elif dead_end_signature == state.last_dead_end_signature:
                        state.repeated_dead_end_count += 1
                    else:
                        state.last_dead_end_signature = dead_end_signature
                        state.repeated_dead_end_count = 1

                    if state.repeated_dead_end_count >= self.config.circuit_breaker_threshold:
                        logger.warning(
                            "Circuit breaker triggered after repeated dead-end signature %s",
                            dead_end_signature,
                        )
                        self._trigger_replan(task, state, observation)

                    if tool_result.is_terminal:
                        state.answer = tool_result.answer
                        logger.info(f"\nTask completed | total_steps={step_index} | total_elapsed={time.perf_counter() - run_start:.1f}s")
                        break
                except Exception as exc:
                    step_elapsed = time.perf_counter() - step_start
                    logger.error(f"Step {step_index} exception | elapsed={step_elapsed:.1f}s | error={exc}")
                    current_action = locals().get("current_action", "__error__")
                    current_action_input = locals().get("current_action_input", {})
                    observation = prune_observation(current_action, ok=False, content=str(exc))
                    step_record = StepRecord(
                        step_index=step_index,
                        thought="",
                        action=current_action,
                        action_input=current_action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                    state.steps.append(step_record)
                    dead_end_signature = self._dead_end_signature(current_action, observation)
                    if dead_end_signature is None:
                        state.repeated_dead_end_count = 0
                        state.last_dead_end_signature = None
                    elif dead_end_signature == state.last_dead_end_signature:
                        state.repeated_dead_end_count += 1
                    else:
                        state.last_dead_end_signature = dead_end_signature
                        state.repeated_dead_end_count = 1

                    if state.repeated_dead_end_count >= self.config.circuit_breaker_threshold:
                        logger.warning(
                            "Circuit breaker triggered after repeated dead-end signature %s",
                            dead_end_signature,
                        )
                        self._trigger_replan(task, state, observation)

            if state.answer is None and state.failure_reason is None:
                state.failure_reason = "Agent did not submit an answer within max_steps."
                logger.warning(f"Max steps {self.config.max_steps} reached without answer | total_elapsed={time.perf_counter() - run_start:.1f}s")

            return AgentRunResult(
                task_id=task.task_id,
                answer=state.answer,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
            )
        finally:
            if hasattr(self.tools, "cleanup_task_runtime"):
                self.tools.cleanup_task_runtime(task.task_id)
