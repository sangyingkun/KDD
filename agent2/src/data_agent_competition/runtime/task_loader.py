from __future__ import annotations

import json
from pathlib import Path

from data_agent_competition.adapters.asset_inventory import build_asset_inventory
from data_agent_competition.runtime.config import CompetitionConfig
from data_agent_competition.semantic.types import TaskBundle


class TaskLoadError(RuntimeError):
    """Raised when a benchmark task directory is malformed."""


def load_task_bundle(task_id: str, config: CompetitionConfig) -> TaskBundle:
    task_dir = config.dataset.root_path / task_id
    task_json_path = task_dir / "task.json"
    context_dir = task_dir / "context"

    if not task_json_path.exists():
        raise TaskLoadError(f"Missing task.json for {task_id}: {task_json_path}")
    if not context_dir.is_dir():
        raise TaskLoadError(f"Missing context directory for {task_id}: {context_dir}")

    payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    expected_keys = {"task_id", "difficulty", "question"}
    if set(payload) != expected_keys:
        raise TaskLoadError(
            f"Unexpected task.json keys for {task_id}: expected {sorted(expected_keys)}, got {sorted(payload)}"
        )
    if payload["task_id"] != task_id:
        raise TaskLoadError(
            f"task_id mismatch for {task_id}: task.json contains {payload['task_id']!r}"
        )

    inventory = build_asset_inventory(context_dir)
    return TaskBundle(
        task_id=str(payload["task_id"]),
        difficulty=str(payload["difficulty"]),
        question=str(payload["question"]),
        task_dir=task_dir,
        context_dir=context_dir,
        assets=inventory.assets,
    )
