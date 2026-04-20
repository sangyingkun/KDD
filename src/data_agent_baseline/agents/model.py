from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


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

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=120.0,
        )

        msg_tokens = sum(len(m.content.split()) for m in messages)
        logger.info(f"Calling model | model={self.model} | messages={len(messages)} | approx_tokens={msg_tokens}")

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=self.temperature,
                max_tokens=8192
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


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
