from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml


def _agent2_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


AGENT2_ROOT = _agent2_root()
REPO_ROOT = _repo_root()


def _default_dataset_root() -> Path:
    return AGENT2_ROOT / "public" / "input"


def _default_output_root() -> Path:
    return AGENT2_ROOT / "artifacts" / "runs"


def _default_semantic_cache_root() -> Path:
    return AGENT2_ROOT / ".cache" / "semantic"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    output_dir: Path = field(default_factory=_default_output_root)
    run_id: str | None = None
    semantic_cache_dir: Path = field(default_factory=_default_semantic_cache_root)
    max_workers: int = 1
    task_timeout_seconds: int = 600
    graph_recursion_limit: int = 24


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "qwen3.6-35b-a3b"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    embedding_enabled: bool = True
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_model_path: str = ""
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_local_files_only: bool = True
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    enable_function_calling: bool = True
    allow_text_fallback_when_tools_missing: bool = False
    enable_stateful_python_session: bool = True
    python_session_timeout_seconds: int = 30
    max_semantic_retries: int = 1
    semantic_llm_enabled: bool = True
    semantic_llm_timeout_seconds: int = 45
    semantic_llm_max_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class CompetitionConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (AGENT2_ROOT / candidate).resolve()


def load_competition_config(config_path: Path) -> CompetitionConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_payload = payload.get("dataset", {})
    runtime_payload = payload.get("run", payload.get("runtime", {}))
    agent_payload = payload.get("agent", {})

    dataset_defaults = DatasetConfig()
    runtime_defaults = RuntimeConfig()
    agent_defaults = AgentConfig()

    raw_run_id = runtime_payload.get("run_id")
    run_id = runtime_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    return CompetitionConfig(
        dataset=DatasetConfig(
            root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path)
        ),
        runtime=RuntimeConfig(
            output_dir=_path_value(runtime_payload.get("output_dir"), runtime_defaults.output_dir),
            run_id=run_id,
            semantic_cache_dir=_path_value(
                runtime_payload.get("semantic_cache_dir"),
                runtime_defaults.semantic_cache_dir,
            ),
            max_workers=int(runtime_payload.get("max_workers", runtime_defaults.max_workers)),
            task_timeout_seconds=int(
                runtime_payload.get("task_timeout_seconds", runtime_defaults.task_timeout_seconds)
            ),
            graph_recursion_limit=int(
                runtime_payload.get("graph_recursion_limit", runtime_defaults.graph_recursion_limit)
            ),
        ),
        agent=AgentConfig(
            model=str(agent_payload.get("model", agent_defaults.model)),
            api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
            api_key=_resolve_agent_api_key(str(agent_payload.get("api_key", agent_defaults.api_key))),
            embedding_enabled=bool(
                agent_payload.get("embedding_enabled", agent_defaults.embedding_enabled)
            ),
            embedding_model_name=str(
                agent_payload.get("embedding_model_name", agent_defaults.embedding_model_name)
            ),
            embedding_model_path=str(
                agent_payload.get("embedding_model_path", agent_defaults.embedding_model_path)
            ),
            embedding_dimension=int(
                agent_payload.get("embedding_dimension", agent_defaults.embedding_dimension)
            ),
            embedding_device=str(
                agent_payload.get("embedding_device", agent_defaults.embedding_device)
            ),
            embedding_batch_size=int(
                agent_payload.get("embedding_batch_size", agent_defaults.embedding_batch_size)
            ),
            embedding_local_files_only=bool(
                agent_payload.get(
                    "embedding_local_files_only",
                    agent_defaults.embedding_local_files_only,
                )
            ),
            embedding_query_prefix=str(
                agent_payload.get(
                    "embedding_query_prefix",
                    agent_defaults.embedding_query_prefix,
                )
            ),
            embedding_document_prefix=str(
                agent_payload.get(
                    "embedding_document_prefix",
                    agent_defaults.embedding_document_prefix,
                )
            ),
            max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
            temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
            enable_function_calling=bool(
                agent_payload.get("enable_function_calling", agent_defaults.enable_function_calling)
            ),
            allow_text_fallback_when_tools_missing=bool(
                agent_payload.get(
                    "allow_text_fallback_when_tools_missing",
                    agent_defaults.allow_text_fallback_when_tools_missing,
                )
            ),
            enable_stateful_python_session=bool(
                agent_payload.get(
                    "enable_stateful_python_session",
                    agent_defaults.enable_stateful_python_session,
                )
            ),
            python_session_timeout_seconds=int(
                agent_payload.get(
                    "python_session_timeout_seconds",
                    agent_defaults.python_session_timeout_seconds,
                )
            ),
            max_semantic_retries=int(
                agent_payload.get("max_semantic_retries", agent_defaults.max_semantic_retries)
            ),
            semantic_llm_enabled=bool(
                agent_payload.get("semantic_llm_enabled", agent_defaults.semantic_llm_enabled)
            ),
            semantic_llm_timeout_seconds=int(
                agent_payload.get(
                    "semantic_llm_timeout_seconds",
                    agent_defaults.semantic_llm_timeout_seconds,
                )
            ),
            semantic_llm_max_tokens=int(
                agent_payload.get("semantic_llm_max_tokens", agent_defaults.semantic_llm_max_tokens)
            ),
        ),
    )


def _resolve_agent_api_key(configured_api_key: str) -> str:
    if configured_api_key:
        return configured_api_key
    for env_name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""
