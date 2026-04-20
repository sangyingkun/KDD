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

        msg_tokens = sum(len(m.content.split()) for m in messages)  # 粗略 token 估算
        logger.info(f"📋 调用模型 | model={self.model} | 消息数={len(messages)} | 粗略tokens≈{msg_tokens}")

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=self.temperature
            )
        except APIError as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"❌ 模型调用失败 | 耗时={elapsed:.1f}s | 错误={exc}")
            raise RuntimeError(f"Model request failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")

        # 打印响应摘要（截断到 200 字符）
        preview = content[:200].replace("\n", " ")
        if len(content) > 200:
            preview += "..."
        logger.info(f"✅ 模型响应 | 耗时={elapsed:.1f}s | 响应长度={len(content)}字符 | 预览={preview}")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
