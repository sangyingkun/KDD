from __future__ import annotations

import json
import logging
from typing import Any

from openai import APIError, OpenAI

from data_agent_competition.runtime.config import AgentConfig

logger = logging.getLogger("agent2.semantic.llm")


class SemanticLLMClient:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._disabled_reason: str | None = None
        if not config.semantic_llm_enabled:
            self._disabled_reason = "semantic_llm_disabled_by_config"
        elif not config.api_key:
            self._disabled_reason = "missing_api_key"

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def call_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        function_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.api_base.rstrip("/"),
            timeout=float(self._config.semantic_llm_timeout_seconds),
        )
        request_kwargs = self._request_kwargs(
            model=self._config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            function_name=function_name,
            schema=schema,
        )
        try:
            response = client.chat.completions.create(**request_kwargs)
        except APIError as exc:
            logger.warning("Semantic LLM request failed: %s", exc)
            return None
        choices = response.choices or []
        if not choices:
            return None
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if function is None or getattr(function, "name", None) != function_name:
                continue
            arguments_text = getattr(function, "arguments", "{}")
            try:
                payload = json.loads(arguments_text)
            except json.JSONDecodeError:
                return None
            if isinstance(payload, dict):
                return payload
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return _parse_json_content(content)
        return None

    def _request_kwargs(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        function_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.semantic_llm_max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": f"Return structured semantic output for {function_name}.",
                        "parameters": schema,
                    },
                }
            ],
        }
        if _is_dashscope_compatible(self._config.api_base):
            request_kwargs["tool_choice"] = "auto"
            request_kwargs["extra_body"] = {"enable_thinking": False}
            return request_kwargs
        request_kwargs["tool_choice"] = {"type": "function", "function": {"name": function_name}}
        return request_kwargs


def _parse_json_content(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _is_dashscope_compatible(api_base: str) -> bool:
    lowered = api_base.rstrip("/").lower()
    return "dashscope.aliyuncs.com/compatible-mode/v1" in lowered
