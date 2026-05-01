from __future__ import annotations

from data_agent_competition.agent.graph import run_controller_graph
from data_agent_competition.agent.state import ControllerState
from data_agent_competition.llm.client import SemanticLLMClient
from data_agent_competition.runtime.config import CompetitionConfig
from data_agent_competition.semantic.embedding import build_embedding_provider
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.semantic.types import TaskBundle


def run_controller(task: TaskBundle, config: CompetitionConfig) -> ControllerState:
    state = ControllerState(task=task)
    runtime = SemanticRuntime(
        llm_client=SemanticLLMClient(config.agent),
        embedding_provider=build_embedding_provider(config.agent),
    )
    return run_controller_graph(state, config, runtime)
