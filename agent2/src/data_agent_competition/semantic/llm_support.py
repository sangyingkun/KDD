from __future__ import annotations

from dataclasses import dataclass

from data_agent_competition.semantic.embedding import EmbeddingProvider
from data_agent_competition.llm.client import SemanticLLMClient


@dataclass(frozen=True, slots=True)
class SemanticRuntime:
    llm_client: SemanticLLMClient
    embedding_provider: EmbeddingProvider
