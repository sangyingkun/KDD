from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_competition.agent.controller import run_controller
from data_agent_competition.benchmark.dataset import DABenchPublicDataset
from data_agent_competition.runtime.config import CompetitionConfig
from data_agent_competition.runtime.result_writer import (
    TaskRunArtifacts,
    create_run_output_dir,
    write_run_summary,
    write_task_outputs,
)
from data_agent_competition.runtime.task_loader import load_task_bundle
from data_agent_competition.semantic.artifact_loader import ArtifactLoadError


@dataclass(frozen=True, slots=True)
class PendingKernelResult:
    task_id: str
    trace_payload: dict[str, Any]
    failure_reason: str | None


def run_single_task(
    *,
    task_id: str,
    config: CompetitionConfig,
    run_output_dir: Path,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    bundle = load_task_bundle(task_id, config)
    trace_payload: dict[str, Any] = {
        "task_id": bundle.task_id,
        "question": bundle.question,
        "status": "loaded",
        "task_bundle": bundle.to_dict(),
        "semantic_snapshot_mode": None,
        "semantic_snapshot_path": None,
        "failure_reason": None,
    }
    failure_reason: str | None = None

    try:
        controller_state = run_controller(bundle, config)
        artifact = controller_state.semantic_artifact
        trace_payload["semantic_snapshot_mode"] = controller_state.semantic_artifact_mode
        trace_payload["semantic_snapshot_path"] = controller_state.semantic_artifact_path
        if artifact is not None:
            trace_payload["semantic_snapshot_summary"] = {
                "asset_count": len(artifact.assets),
                "source_count": len(artifact.sources),
                "knowledge_fact_count": len(artifact.knowledge_facts),
                "join_candidate_count": len(artifact.join_candidates),
                "doc_chunk_count": len(artifact.doc_chunks),
            }
        if controller_state.logical_plan is not None:
            trace_payload["logical_plan"] = controller_state.logical_plan.to_dict()
        if controller_state.semantic_routing is not None:
            trace_payload["semantic_routing"] = controller_state.semantic_routing.to_dict()
        if controller_state.logical_verification is not None:
            trace_payload["logical_verification"] = controller_state.logical_verification.to_dict()
        if controller_state.physical_plan is not None:
            trace_payload["physical_plan"] = controller_state.physical_plan.to_dict()
        if controller_state.execution_result is not None:
            trace_payload["execution"] = controller_state.execution_result.to_dict()
        if controller_state.final_verification is not None:
            trace_payload["final_verification"] = controller_state.final_verification.to_dict()
        trace_payload["controller"] = {
            "schema_version": controller_state.trace_schema_version,
            "status": controller_state.status,
            "semantic_attempts": controller_state.semantic_attempts,
            "failure_signature": controller_state.failure_signature,
            "orchestration_backend": controller_state.orchestration_backend,
            "steps": controller_state.trace_steps,
        }
        trace_payload["status"] = controller_state.status
        failure_reason = controller_state.failure_reason
        trace_payload["failure_reason"] = failure_reason
        trace_payload["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
        return write_task_outputs(
            task_id=task_id,
            run_output_dir=run_output_dir,
            trace_payload=trace_payload,
            answer_table=controller_state.answer_table,
            failure_reason=failure_reason,
        )
    except ArtifactLoadError as exc:
        trace_payload["status"] = "artifact_missing"
        failure_reason = str(exc)

    trace_payload["failure_reason"] = failure_reason
    trace_payload["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return write_task_outputs(
        task_id=task_id,
        run_output_dir=run_output_dir,
        trace_payload=trace_payload,
        answer_table=None,
        failure_reason=failure_reason,
    )


def run_many_tasks(
    *,
    task_ids: list[str],
    config: CompetitionConfig,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    run_id, run_output_dir = create_run_output_dir(config.runtime.output_dir, run_id=config.runtime.run_id)
    effective_workers = max(int(config.runtime.max_workers), 1)
    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1 or len(task_ids) <= 1:
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(task_id=task_id, config=config, run_output_dir=run_output_dir)
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]
    write_run_summary(run_output_dir, run_id=run_id, task_artifacts=task_artifacts)
    return run_output_dir, task_artifacts


def run_benchmark(
    *,
    config: CompetitionConfig,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    dataset = DABenchPublicDataset(config.dataset.root_path)
    task_ids = [task.task_id for task in dataset.iter_tasks()]
    if limit is not None:
        task_ids = task_ids[:limit]
    return run_many_tasks(task_ids=task_ids, config=config, progress_callback=progress_callback)
