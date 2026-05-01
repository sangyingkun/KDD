from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

SCRIPT_ROOT = Path(__file__).resolve().parent
AGENT2_ROOT = SCRIPT_ROOT.parent
SRC_ROOT = AGENT2_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_agent_competition.runtime.config import AGENT2_ROOT, load_competition_config
from data_agent_competition.runtime.task_loader import load_task_bundle
from data_agent_competition.semantic.artifact_builder import build_semantic_artifact
from data_agent_competition.semantic.embedding import build_embedding_provider
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.llm.client import SemanticLLMClient

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def build(
    task_id: list[str] = typer.Option(..., "--task-id", help="Task ids to inspect."),
    config: Path = typer.Option(
        AGENT2_ROOT / "configs" / "competition.example.yaml",
        exists=True,
        dir_okay=False,
        help="Agent2 config path.",
    ),
) -> None:
    competition_config = load_competition_config(config)
    runtime = SemanticRuntime(
        llm_client=SemanticLLMClient(competition_config.agent),
        embedding_provider=build_embedding_provider(competition_config.agent),
    )
    for one_task_id in task_id:
        bundle = load_task_bundle(one_task_id, competition_config)
        artifact = build_semantic_artifact(
            bundle,
            runtime=runtime,
            enable_llm_indexing=True,
        )
        output_path = AGENT2_ROOT / "competition_artifacts" / one_task_id / "semantic_artifact.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        typer.echo(str(output_path))


if __name__ == "__main__":
    app()
