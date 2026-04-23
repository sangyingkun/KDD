from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    enable_function_calling: bool = True
    allow_text_fallback_when_tools_missing: bool = False
    enable_stateful_python_session: bool = True
    python_session_timeout_seconds: int = 30


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    enable_dense_linking: bool = True
    enable_global_scope: bool = False
    embedding_provider: str = "local"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_model_path: str = ""
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_local_files_only: bool = True
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    retrieval_top_k: int = 8
    lexical_top_k: int = 8
    final_candidate_top_k: int = 12


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()
    retrieval_defaults = RetrievalConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})
    retrieval_payload = payload.get("retrieval", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
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
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    retrieval_config = RetrievalConfig(
        enable_dense_linking=bool(
            retrieval_payload.get("enable_dense_linking", retrieval_defaults.enable_dense_linking)
        ),
        enable_global_scope=bool(
            retrieval_payload.get("enable_global_scope", retrieval_defaults.enable_global_scope)
        ),
        embedding_provider=str(
            retrieval_payload.get("embedding_provider", retrieval_defaults.embedding_provider)
        ),
        embedding_model_name=str(
            retrieval_payload.get("embedding_model_name", retrieval_defaults.embedding_model_name)
        ),
        embedding_model_path=str(
            retrieval_payload.get("embedding_model_path", retrieval_defaults.embedding_model_path)
        ),
        embedding_api_base=str(
            retrieval_payload.get("embedding_api_base", retrieval_defaults.embedding_api_base)
        ),
        embedding_api_key=str(
            retrieval_payload.get("embedding_api_key", retrieval_defaults.embedding_api_key)
        ),
        embedding_dimension=int(
            retrieval_payload.get("embedding_dimension", retrieval_defaults.embedding_dimension)
        ),
        embedding_device=str(
            retrieval_payload.get("embedding_device", retrieval_defaults.embedding_device)
        ),
        embedding_batch_size=int(
            retrieval_payload.get("embedding_batch_size", retrieval_defaults.embedding_batch_size)
        ),
        embedding_local_files_only=bool(
            retrieval_payload.get(
                "embedding_local_files_only",
                retrieval_defaults.embedding_local_files_only,
            )
        ),
        embedding_query_prefix=str(
            retrieval_payload.get("embedding_query_prefix", retrieval_defaults.embedding_query_prefix)
        ),
        embedding_document_prefix=str(
            retrieval_payload.get("embedding_document_prefix", retrieval_defaults.embedding_document_prefix)
        ),
        retrieval_top_k=int(retrieval_payload.get("retrieval_top_k", retrieval_defaults.retrieval_top_k)),
        lexical_top_k=int(retrieval_payload.get("lexical_top_k", retrieval_defaults.lexical_top_k)),
        final_candidate_top_k=int(
            retrieval_payload.get("final_candidate_top_k", retrieval_defaults.final_candidate_top_k)
        ),
    )
    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        run=run_config,
        retrieval=retrieval_config,
    )
