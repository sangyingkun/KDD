from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from data_agent_competition.artifacts.schema import RetrievalDocument, SemanticArtifact
from data_agent_competition.semantic.embedding import EmbeddingProvider
from data_agent_competition.semantic.normalization import token_overlap_score
from data_agent_competition.vendor.lightrag_core.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)
from data_agent_competition.vendor.lightrag_core.operate import LightRAGQueryStores


def build_lightrag_task_stores(
    artifact: SemanticArtifact,
    embedding_provider: EmbeddingProvider,
) -> LightRAGQueryStores:
    entity_docs, relation_docs, chunk_docs = _partition_documents(artifact)
    chunk_payload = {
        chunk.chunk_id: {
            "id": chunk.chunk_id,
            "content": chunk.text,
            "asset_path": chunk.asset_path,
            "section": chunk.section,
            "keywords": list(chunk.keywords),
        }
        for chunk in artifact.doc_chunks
    }
    return LightRAGQueryStores(
        graph=ArtifactGraphStorage.from_artifact(artifact),
        entities_vdb=ArtifactVectorStorage.from_documents(
            namespace="entities",
            documents=entity_docs,
            embedding_provider=embedding_provider,
        ),
        relations_vdb=ArtifactVectorStorage.from_documents(
            namespace="relations",
            documents=relation_docs,
            embedding_provider=embedding_provider,
        ),
        chunks_vdb=ArtifactVectorStorage.from_documents(
            namespace="chunks",
            documents=chunk_docs,
            embedding_provider=embedding_provider,
        ),
        text_chunks=ArtifactKVStorage(
            namespace="chunks",
            workspace=artifact.task_id,
            global_config={},
            data=chunk_payload,
        ),
    )


def _partition_documents(
    artifact: SemanticArtifact,
) -> tuple[tuple[RetrievalDocument, ...], tuple[RetrievalDocument, ...], tuple[RetrievalDocument, ...]]:
    entity_docs: list[RetrievalDocument] = []
    relation_docs: list[RetrievalDocument] = []
    chunk_docs: list[RetrievalDocument] = []
    for document in artifact.retrieval_documents:
        node_type = document.metadata.get("node_type", "")
        if document.doc_id.startswith("graph-edge::"):
            relation_docs.append(document)
            continue
        if node_type in {"source", "field", "value_concept", "ambiguity"}:
            entity_docs.append(document)
            continue
        if node_type in {"join", "metric", "constraint", "use_case"}:
            relation_docs.append(document)
            continue
        chunk_docs.append(document)
    for chunk in artifact.doc_chunks:
        chunk_docs.append(
            RetrievalDocument(
                doc_id=f"chunk::{chunk.chunk_id}",
                scope="task",
                doc_type="chunk",
                text=chunk.text,
                source_ref=chunk.chunk_id,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "asset_path": chunk.asset_path,
                    "section": chunk.section,
                },
                evidence_refs=[chunk.chunk_id],
                confidence=0.45,
            )
        )
    return tuple(entity_docs), tuple(relation_docs), tuple(chunk_docs)


@dataclass(kw_only=True)
class ArtifactVectorStorage(BaseVectorStorage):
    documents: tuple[RetrievalDocument, ...]

    @classmethod
    def from_documents(
        cls,
        *,
        namespace: str,
        documents: tuple[RetrievalDocument, ...],
        embedding_provider: EmbeddingProvider,
    ) -> ArtifactVectorStorage:
        return cls(
            namespace=namespace,
            workspace="task",
            global_config={},
            embedding_func=embedding_provider,
            documents=documents,
        )

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        return {"status": "success", "message": "data dropped"}

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.embedding_func.score_documents(
            query=query,
            documents=self.documents,
            limit=max(top_k * 3, top_k),
        )
        score_by_id = {item.doc_id: item.score for item in scores}
        ranked: list[dict[str, Any]] = []
        for document in self.documents:
            lexical = float(token_overlap_score(query, document.text))
            dense = score_by_id.get(document.doc_id, 0.0)
            total = (lexical * 0.45) + (dense * 0.55)
            if total <= 0:
                continue
            ranked.append(
                {
                    "id": document.doc_id,
                    "score": total,
                    "text": document.text,
                    "source_ref": document.source_ref,
                    "metadata": dict(document.metadata),
                    "evidence_refs": list(document.evidence_refs),
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["id"]))
        return ranked[:top_k]

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        return None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        for document in self.documents:
            if document.doc_id == id:
                return {
                    "id": document.doc_id,
                    "text": document.text,
                    "source_ref": document.source_ref,
                    "metadata": dict(document.metadata),
                }
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        id_set = set(ids)
        return [
            {
                "id": document.doc_id,
                "text": document.text,
                "source_ref": document.source_ref,
                "metadata": dict(document.metadata),
            }
            for document in self.documents
            if document.doc_id in id_set
        ]


@dataclass(kw_only=True)
class ArtifactKVStorage(BaseKVStorage):
    data: dict[str, dict[str, Any]]

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        self.data.clear()
        return {"status": "success", "message": "data dropped"}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return self.data.get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self.data[id] for id in ids if id in self.data]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return {key for key in keys if key not in self.data}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        self.data.update(data)

    async def delete(self, ids: list[str]) -> None:
        for id in ids:
            self.data.pop(id, None)

    async def is_empty(self) -> bool:
        return not self.data


@dataclass(kw_only=True)
class ArtifactGraphStorage(BaseGraphStorage):
    nodes: dict[str, dict[str, Any]]
    edges: dict[tuple[str, str], dict[str, Any]]
    adjacency: dict[str, list[tuple[str, str]]]
    node_labels: dict[str, str]

    @classmethod
    def from_artifact(cls, artifact: SemanticArtifact) -> ArtifactGraphStorage:
        nodes = {
            node.node_id: {
                "node_id": node.node_id,
                "label": node.label,
                "node_type": node.node_type,
                "canonical_text": node.canonical_text,
                **dict(node.metadata),
            }
            for node in artifact.graph_nodes
        }
        edges: dict[tuple[str, str], dict[str, Any]] = {}
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in artifact.graph_edges:
            key = _edge_key(edge.source_node_id, edge.target_node_id)
            edges[key] = {
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "weight": edge.weight,
            }
            adjacency[edge.source_node_id].append((edge.source_node_id, edge.target_node_id))
            adjacency[edge.target_node_id].append((edge.source_node_id, edge.target_node_id))
        return cls(
            namespace="graph",
            workspace=artifact.task_id,
            global_config={},
            nodes=nodes,
            edges=edges,
            adjacency=dict(adjacency),
            node_labels={node_id: str(payload.get("label", "")) for node_id, payload in nodes.items()},
        )

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        return {"status": "success", "message": "data dropped"}

    async def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return _edge_key(source_node_id, target_node_id) in self.edges

    async def node_degree(self, node_id: str) -> int:
        return len(self.adjacency.get(node_id, []))

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes.get(node_id)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, Any] | None:
        return self.edges.get(_edge_key(source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        return list(self.adjacency.get(source_node_id, []))

    async def get_all_nodes(self) -> list[dict[str, Any]]:
        return list(self.nodes.values())

    async def get_all_edges(self) -> list[dict[str, Any]]:
        return list(self.edges.values())

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        ranked = sorted(
            self.node_labels.items(),
            key=lambda item: (-len(self.adjacency.get(item[0], [])), item[1]),
        )
        return [label for _, label in ranked[:limit]]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        ranked = sorted(
            (
                (node_id, label, token_overlap_score(query, label))
                for node_id, label in self.node_labels.items()
            ),
            key=lambda item: (-item[2], item[1]),
        )
        return [label for _, label, score in ranked if score > 0][:limit]


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))
