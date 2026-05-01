from __future__ import annotations

import json
from pathlib import Path

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.runtime.config import AGENT2_ROOT
from data_agent_competition.semantic.artifact_builder import ARTIFACT_VERSION_MARKER, build_semantic_artifact
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.semantic.types import ArtifactLoadResult, TaskBundle


class ArtifactLoadError(RuntimeError):
    """Raised when the semantic layer cannot build its internal snapshot."""


def load_task_artifact(
    task: TaskBundle,
    runtime: SemanticRuntime | None = None,
) -> ArtifactLoadResult:
    artifact_path = _static_artifact_path(task.task_id)
    if artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact = SemanticArtifact.model_validate(payload)
            if _artifact_is_current(artifact):
                return ArtifactLoadResult(
                    artifact=artifact,
                    mode="static_committed",
                    artifact_path=str(artifact_path),
                )
            if runtime is not None:
                rebuilt_artifact = build_semantic_artifact(
                    task,
                    runtime=None,
                    enable_llm_indexing=False,
                )
                return ArtifactLoadResult(
                    artifact=rebuilt_artifact,
                    mode="runtime_rules_rebuilt_from_legacy_static",
                    artifact_path=str(artifact_path),
                )
            return ArtifactLoadResult(
                artifact=artifact,
                mode="static_committed_legacy",
                artifact_path=str(artifact_path),
            )
        except Exception as exc:  # noqa: BLE001
            raise ArtifactLoadError(f"Failed to load static semantic artifact for {task.task_id}: {exc}") from exc
    try:
        return ArtifactLoadResult(
            artifact=build_semantic_artifact(
                task,
                runtime=None,
                enable_llm_indexing=False,
            ),
            mode="runtime_internal_rules_only",
            artifact_path=None,
        )
    except Exception as exc:  # noqa: BLE001
        raise ArtifactLoadError(f"Failed to build semantic snapshot for {task.task_id}: {exc}") from exc


def _static_artifact_path(task_id: str) -> Path:
    return AGENT2_ROOT / "competition_artifacts" / task_id / "semantic_artifact.json"


def _artifact_is_current(artifact: SemanticArtifact) -> bool:
    return ARTIFACT_VERSION_MARKER in artifact.notes
