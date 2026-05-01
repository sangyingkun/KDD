from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AnswerTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()
    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def write_task_outputs(
    *,
    task_id: str,
    run_output_dir: Path,
    trace_payload: dict[str, Any],
    answer_table: AnswerTable | None,
    failure_reason: str | None,
) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = task_output_dir / "trace.json"
    trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prediction_csv_path: Path | None = None
    if answer_table is not None:
        prediction_csv_path = task_output_dir / "prediction.csv"
        with prediction_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(answer_table.columns)
            for row in answer_table.rows:
                writer.writerow(row)

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=answer_table is not None and failure_reason is None,
        failure_reason=failure_reason,
    )


def write_run_summary(
    run_output_dir: Path,
    *,
    run_id: str,
    task_artifacts: list[TaskRunArtifacts],
) -> Path:
    summary_path = run_output_dir / "summary.json"
    payload = {
        "run_id": run_id,
        "task_count": len(task_artifacts),
        "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
        "tasks": [artifact.to_dict() for artifact in task_artifacts],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path
