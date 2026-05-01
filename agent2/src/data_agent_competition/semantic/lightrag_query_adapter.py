from __future__ import annotations

import asyncio

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.semantic.embedding import EmbeddingProvider
from data_agent_competition.semantic.lightrag_store_adapter import build_lightrag_task_stores
from data_agent_competition.semantic.types import TaskBundle
from data_agent_competition.vendor.lightrag_core import (
    LightRAGSemanticResult,
    build_query_param,
    build_semantic_candidates,
    query_task_semantic_graph,
)


def query_semantic_graph_with_lightrag(
    *,
    task: TaskBundle,
    artifact: SemanticArtifact,
    embedding_provider: EmbeddingProvider,
) -> LightRAGSemanticResult:
    query_param = build_query_param(task)
    stores = build_lightrag_task_stores(artifact, embedding_provider)
    retrieval = asyncio.run(
        query_task_semantic_graph(
            question=task.question,
            artifact=artifact,
            stores=stores,
            query_param=query_param,
        )
    )
    candidates = build_semantic_candidates(question=task.question, artifact=artifact, retrieval=retrieval)
    return LightRAGSemanticResult(
        retrieval=retrieval,
        candidates=candidates,
    )
