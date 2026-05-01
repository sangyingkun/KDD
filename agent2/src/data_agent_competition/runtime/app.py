from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_competition.benchmark.dataset import DABenchPublicDataset
from data_agent_competition.io.context_filesystem import list_context_tree
from data_agent_competition.runtime.config import AGENT2_ROOT, load_competition_config
from data_agent_competition.runtime.runner import TaskRunArtifacts, create_run_output_dir, run_benchmark, run_single_task

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = AGENT2_ROOT
CONFIGS_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console(force_terminal=True)


def _setup_logging() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    rich_handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
    rich_handler.setLevel(logging.INFO)

    log_file = ARTIFACTS_DIR / "dabench-agent2.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger("agent2")
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local Agent2 competition kernel."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    competition_config = load_competition_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(competition_config.dataset.root_path)

    table = Table(title="Agent2 Competition Kernel Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(competition_config.dataset.root_path), _status_value(competition_config.dataset.root_path))
    table.add_row("config_path", str(config_path), _status_value(config_path))
    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(f"{difficulty}={count}" for difficulty, count in sorted(counts.items()))
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    competition_config = load_competition_config(config)
    dataset = DABenchPublicDataset(competition_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    _setup_logging()
    competition_config = load_competition_config(config)
    try:
        _, run_output_dir = create_run_output_dir(competition_config.runtime.output_dir, run_id=competition_config.runtime.run_id)
    except Exception as exc:
        msg = str(exc).encode("ascii", "replace").decode("ascii")
        console.print(f"[red]Error: {msg}[/red]")
        raise typer.Exit(1) from exc
    artifacts = run_single_task(task_id=task_id, config=competition_config, run_output_dir=run_output_dir)
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
) -> None:
    _setup_logging()
    competition_config = load_competition_config(config)
    dataset = DABenchPublicDataset(competition_config.dataset.root_path)
    task_total = len(dataset.iter_tasks())
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = competition_config.runtime.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )
        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact: TaskRunArtifacts) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=competition_config,
                limit=limit,
                progress_callback=on_task_complete,
            )
        except Exception as exc:
            msg = str(exc).encode("ascii", "replace").decode("ascii")
            console.print(f"[red]Error: {msg}[/red]")
            raise typer.Exit(1) from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
