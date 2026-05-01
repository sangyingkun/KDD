from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    canonical_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class GraphEdge(BaseModel):
    edge_id: str
    edge_type: str
    source_node_id: str
    target_node_id: str
    weight: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)


class ArtifactAsset(BaseModel):
    asset_id: str
    relative_path: str
    kind: str
    format: str
    description: str = ""
    size_bytes: int = 0


class KnowledgeFact(BaseModel):
    fact_id: str
    statement: str
    evidence: list[str] = Field(default_factory=list)
    source_path: str = ""
    tags: list[str] = Field(default_factory=list)


class SourceField(BaseModel):
    field_name: str
    dtype: str = ""
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    sample_values: list[str] = Field(default_factory=list)
    semantic_tags: list[str] = Field(default_factory=list)
    nullable: bool = True


class SourceDescriptor(BaseModel):
    source_id: str
    source_kind: str
    asset_path: str
    object_name: str
    description: str = ""
    grain_hint: list[str] = Field(default_factory=list)
    fields: list[SourceField] = Field(default_factory=list)


class JoinCandidate(BaseModel):
    left_source_id: str
    left_field: str
    right_source_id: str
    right_field: str
    rationale: str = ""
    confidence: float = 0.0


class DocumentChunk(BaseModel):
    chunk_id: str
    asset_path: str
    text: str
    section: str = ""
    keywords: list[str] = Field(default_factory=list)


class RetrievalDocument(BaseModel):
    doc_id: str
    scope: str
    doc_type: str
    text: str
    source_ref: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class SemanticArtifact(BaseModel):
    task_id: str
    assets: list[ArtifactAsset] = Field(default_factory=list)
    knowledge_facts: list[KnowledgeFact] = Field(default_factory=list)
    sources: list[SourceDescriptor] = Field(default_factory=list)
    join_candidates: list[JoinCandidate] = Field(default_factory=list)
    doc_chunks: list[DocumentChunk] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    retrieval_documents: list[RetrievalDocument] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
