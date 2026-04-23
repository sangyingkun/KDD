from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import APIError, OpenAI

logger = logging.getLogger("dabench.model")


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelTurn:
    raw_response: str
    text_response: str | None = None
    tool_calls: list[ModelToolCall] = field(default_factory=list)


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError

    def complete_with_tools(self, messages: list[ModelMessage], tools: list[dict[str, Any]]) -> ModelTurn:
        raise NotImplementedError


def _message_payloads(messages: list[ModelMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _extract_tool_calls(message: Any) -> list[ModelToolCall]:
    tool_calls: list[ModelToolCall] = []
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    for index, tool_call in enumerate(raw_tool_calls, start=1):
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None)
        if not isinstance(name, str) or not name.strip():
            continue
        arguments_text = getattr(function, "arguments", "{}")
        if not isinstance(arguments_text, str):
            arguments_text = "{}"
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments_text}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        call_id = getattr(tool_call, "id", None)
        tool_calls.append(
            ModelToolCall(
                id=str(call_id) if call_id else f"tool_call_{index}",
                name=name,
                arguments=arguments,
            )
        )
    return tool_calls


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def _client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        return OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=120.0,
        )

    def complete(self, messages: list[ModelMessage]) -> str:
        client = self._client()
        msg_tokens = sum(len(m.content.split()) for m in messages)
        logger.info(f"Calling model | model={self.model} | messages={len(messages)} | approx_tokens={msg_tokens}")

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=_message_payloads(messages),
                temperature=self.temperature,
                max_tokens=8192,
            )
        except APIError as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"Model call failed | elapsed={elapsed:.1f}s | error={exc}")
            raise RuntimeError(f"Model request failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")

        preview = content[:200].replace("\n", " ")
        if len(content) > 200:
            preview += "..."
        logger.info(f"Model response received | elapsed={elapsed:.1f}s | length={len(content)}chars | preview={preview}")
        return content

    def complete_with_tools(self, messages: list[ModelMessage], tools: list[dict[str, Any]]) -> ModelTurn:
        client = self._client()
        msg_tokens = sum(len(m.content.split()) for m in messages)
        logger.info(
            "Calling tool-enabled model | model=%s | messages=%s | tools=%s | approx_tokens=%s",
            self.model,
            len(messages),
            len(tools),
            msg_tokens,
        )

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=_message_payloads(messages),
                temperature=self.temperature,
                max_tokens=8192,
                tools=tools,
                tool_choice="auto",
            )
        except APIError as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"Tool-enabled model call failed | elapsed={elapsed:.1f}s | error={exc}")
            raise RuntimeError(f"Model request failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        message = choices[0].message
        text = message.content if isinstance(message.content, str) else None
        tool_calls = _extract_tool_calls(message)
        preview = (text or "")[:200].replace("\n", " ")
        if text and len(text) > 200:
            preview += "..."
        logger.info(
            "Tool-enabled model response received | elapsed=%.1fs | text_len=%s | tool_calls=%s | preview=%s",
            elapsed,
            len(text) if text is not None else 0,
            len(tool_calls),
            preview,
        )
        return ModelTurn(raw_response=text or "", text_response=text, tool_calls=tool_calls)


class ScriptedModelAdapter:
    def __init__(
        self,
        responses: list[str] | None = None,
        tool_turns: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._tool_turns = [list(turn) for turn in (tool_turns or [])]

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)

    def complete_with_tools(self, messages: list[ModelMessage], tools: list[dict[str, Any]]) -> ModelTurn:
        del messages, tools
        if not self._tool_turns:
            raise RuntimeError("No scripted tool turns remaining.")
        raw_turn = self._tool_turns.pop(0)
        if len(raw_turn) == 1 and isinstance(raw_turn[0], dict) and "text_response" in raw_turn[0]:
            text_response = raw_turn[0].get("text_response")
            return ModelTurn(
                raw_response=str(text_response or ""),
                text_response=str(text_response) if text_response is not None else None,
                tool_calls=[],
            )
        tool_calls: list[ModelToolCall] = []
        for index, item in enumerate(raw_turn, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            tool_calls.append(
                ModelToolCall(
                    id=str(item.get("id", f"tool_call_{index}")),
                    name=name,
                    arguments=dict(arguments),
                )
            )
        return ModelTurn(raw_response="", text_response=None, tool_calls=tool_calls)
