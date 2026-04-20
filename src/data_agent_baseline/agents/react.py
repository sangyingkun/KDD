from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

logger = logging.getLogger("dabench.react")


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


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
def _bootstrap_semantic_context(self, task: PublicTask, state: AgentRuntimeState) -> None:
    bootstrap_actions = (
        ("describe_semantics", {"max_items_per_section": 8, "include_evidence": True}),
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
                    observation={"ok": False, "tool": action, "error": str(exc)},
                    ok=False,
                )
            )
            continue
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
                observation={
                    "ok": tool_result.ok,
                    "tool": action,
                    "content": tool_result.content,
                },
                ok=tool_result.ok,
            )
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

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
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

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        self._bootstrap_semantic_context(task, state)
        run_start = time.perf_counter()
        logger.info(f"{'='*60}")
        logger.info(f"🚀 开始执行任务 | task_id={task.task_id} | difficulty={task.difficulty} | max_steps={self.config.max_steps}")
        logger.info(f"📝 问题: {task.question[:150]}")
        logger.info(f"{'='*60}")

        for step_index in range(1, self.config.max_steps + 1):
            step_start = time.perf_counter()
            logger.info(f"\n🔄 Step {step_index}/{self.config.max_steps} 开始...")
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
                logger.info(f"💭 Thought: {model_step.thought[:200]}")
                logger.info(f"🔧 Action: {model_step.action} | input={json.dumps(model_step.action_input, ensure_ascii=False)[:300]}")

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)

                step_elapsed = time.perf_counter() - step_start
                if tool_result.ok:
                    result_preview = str(tool_result.content)[:300]
                    if len(str(tool_result.content)) > 300:
                        result_preview += "..."
                    logger.info(f"📋 Tool结果 (OK) | 耗时={step_elapsed:.1f}s | 预览={result_preview}")
                else:
                    logger.warning(f"⚠️ Tool执行失败 | 耗时={step_elapsed:.1f}s | 错误={str(tool_result.content)[:200]}")

                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    logger.info(f"\n🏁 任务完成 | 共 {step_index} 步 | 总耗时={time.perf_counter() - run_start:.1f}s")
                    break
            except Exception as exc:
                step_elapsed = time.perf_counter() - step_start
                logger.error(f"❌ Step {step_index} 异常 | 耗时={step_elapsed:.1f}s | 错误={exc}")
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
            logger.warning(f"⚠️ 达到最大步数 {self.config.max_steps}，未提交答案 | 总耗时={time.perf_counter() - run_start:.1f}s")

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
