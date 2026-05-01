from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.semantic.graph_types import (
    GraphEdgeHit,
    GraphNodeHit,
    GraphRetrievalResult,
)
from data_agent_competition.semantic.normalization import token_overlap_score
from data_agent_competition.vendor.lightrag_core.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    QueryParam,
)


LOCAL_EDGE_TYPES = {
    "has_field",
    "alias_of",
    "maps_value",
    "uses_field",
    "uses_source",
    "disambiguates",
}
GLOBAL_EDGE_TYPES = {
    "join_on",
    "constrains",
    "uses_field",
    "uses_source",
    "co_occurs_with",
    "evidenced_by",
}
MIX_EDGE_TYPES = LOCAL_EDGE_TYPES | GLOBAL_EDGE_TYPES


@dataclass(frozen=True, slots=True)
class LightRAGQueryStores:
    graph: BaseGraphStorage
    entities_vdb: BaseVectorStorage
    relations_vdb: BaseVectorStorage
    chunks_vdb: BaseVectorStorage
    text_chunks: BaseKVStorage


async def query_task_semantic_graph(
    *,
    question: str,
    artifact: SemanticArtifact,
    stores: LightRAGQueryStores,
    query_param: QueryParam,
) -> GraphRetrievalResult:
    if query_param.mode == "local":
        return await _mode_retrieval(
            question=question,
            artifact=artifact,
            stores=stores,
            query_param=query_param,
            use_entities=True,
            use_relations=False,
            use_chunks=False,
            allowed_edge_types=LOCAL_EDGE_TYPES,
        )
    if query_param.mode == "global":
        return await _mode_retrieval(
            question=question,
            artifact=artifact,
            stores=stores,
            query_param=query_param,
            use_entities=False,
            use_relations=True,
            use_chunks=False,
            allowed_edge_types=GLOBAL_EDGE_TYPES,
        )
    if query_param.mode == "hybrid":
        return await _mode_retrieval(
            question=question,
            artifact=artifact,
            stores=stores,
            query_param=query_param,
            use_entities=True,
            use_relations=True,
            use_chunks=False,
            allowed_edge_types=MIX_EDGE_TYPES,
        )
    return await _mode_retrieval(
        question=question,
        artifact=artifact,
        stores=stores,
        query_param=query_param,
        use_entities=True,
        use_relations=True,
        use_chunks=True,
        allowed_edge_types=MIX_EDGE_TYPES,
    )


async def _mode_retrieval(
    *,
    question: str,
    artifact: SemanticArtifact,
    stores: LightRAGQueryStores,
    query_param: QueryParam,
    use_entities: bool,
    use_relations: bool,
    use_chunks: bool,
    allowed_edge_types: set[str],
) -> GraphRetrievalResult:
    node_index = {node.node_id: node for node in artifact.graph_nodes}
    edge_index = {edge.edge_id: edge for edge in artifact.graph_edges}
    seed_nodes: dict[str, GraphNodeHit] = {}
    seed_edges: dict[str, GraphEdgeHit] = {}
    chunk_hits: list[dict] = []

    if use_entities:
        entity_hits = await stores.entities_vdb.query(question, top_k=query_param.top_k)
        _ingest_vector_hits(entity_hits, node_index, edge_index, seed_nodes, seed_edges, question, "local_seed")
    if use_relations:
        relation_hits = await stores.relations_vdb.query(question, top_k=query_param.top_k)
        _ingest_vector_hits(relation_hits, node_index, edge_index, seed_nodes, seed_edges, question, "global_seed")
    if use_chunks:
        chunk_hits = await stores.chunks_vdb.query(question, top_k=query_param.chunk_top_k)
        _ingest_chunk_hits(chunk_hits, artifact, node_index, seed_nodes, question)

    expanded_nodes = dict(seed_nodes)
    expanded_edges = dict(seed_edges)
    frontier = deque((node_id, 0) for node_id in seed_nodes)
    visited = {node_id: 0 for node_id in seed_nodes}
    while frontier:
        current_node_id, hop = frontier.popleft()
        if hop >= 2:
            continue
        node_edges = await stores.graph.get_node_edges(current_node_id) or []
        current_hit = expanded_nodes[current_node_id]
        for left_id, right_id in node_edges:
            edge_payload = await stores.graph.get_edge(left_id, right_id)
            if edge_payload is None or edge_payload.get("edge_type") not in allowed_edge_types:
                continue
            neighbor_id = right_id if left_id == current_node_id else left_id
            neighbor_payload = await stores.graph.get_node(neighbor_id)
            if neighbor_payload is None:
                continue
            score = max(current_hit.score * 0.72, 0.05)
            edge_id = str(edge_payload["edge_id"])
            existing_edge = expanded_edges.get(edge_id)
            if existing_edge is None or score > existing_edge.score:
                expanded_edges[edge_id] = GraphEdgeHit(
                    edge_id=edge_id,
                    edge_type=str(edge_payload["edge_type"]),
                    source_node_id=str(edge_payload["source_node_id"]),
                    target_node_id=str(edge_payload["target_node_id"]),
                    score=score,
                    hop_distance=hop + 1,
                    rationale=f"graph_expand::{query_param.mode}",
                )
            lexical = float(
                token_overlap_score(question, str(neighbor_payload.get("label", "")))
                + token_overlap_score(question, str(neighbor_payload.get("canonical_text", "")))
            )
            candidate_score = max(score + (lexical * 0.12), 0.02)
            previous = expanded_nodes.get(neighbor_id)
            if previous is None or candidate_score > previous.score:
                expanded_nodes[neighbor_id] = GraphNodeHit(
                    node_id=neighbor_id,
                    node_type=str(neighbor_payload.get("node_type", "")),
                    label=str(neighbor_payload.get("label", "")),
                    canonical_text=str(neighbor_payload.get("canonical_text", "")),
                    score=candidate_score,
                    lexical_score=lexical,
                    dense_score=0.0,
                    hop_distance=hop + 1,
                    source_id=_nullable(neighbor_payload.get("source_id")),
                    field_name=_nullable(neighbor_payload.get("field_name")),
                    resolved_value=_nullable(neighbor_payload.get("resolved_value")),
                    rationale=f"graph_expand::{query_param.mode}",
                )
            if neighbor_id not in visited or hop + 1 < visited[neighbor_id]:
                visited[neighbor_id] = hop + 1
                frontier.append((neighbor_id, hop + 1))

    ranked_nodes = tuple(
        sorted(
            expanded_nodes.values(),
            key=lambda item: (-item.score, item.hop_distance, item.node_id),
        )[: max(query_param.top_k * 3, 24)]
    )
    ranked_edges = tuple(
        sorted(
            expanded_edges.values(),
            key=lambda item: (-item.score, item.hop_distance, item.edge_id),
        )[: max(query_param.top_k * 4, 32)]
    )
    notes = [
        f"lightrag_mode={query_param.mode}",
        f"hl_keywords={len(query_param.hl_keywords)}",
        f"ll_keywords={len(query_param.ll_keywords)}",
    ]
    if chunk_hits:
        notes.append(f"chunk_hits={len(chunk_hits)}")
    context_sections = await _build_context_sections(
        artifact=artifact,
        stores=stores,
        node_hits=ranked_nodes,
        edge_hits=ranked_edges,
        chunk_hits=chunk_hits,
    )
    return GraphRetrievalResult(
        seed_node_ids=tuple(seed_nodes.keys()),
        expanded_node_ids=tuple(expanded_nodes.keys()),
        node_hits=ranked_nodes,
        edge_hits=ranked_edges,
        notes=tuple(notes),
        context_sections=context_sections,
    )


def _ingest_vector_hits(
    hits: list[dict],
    node_index: dict[str, object],
    edge_index: dict[str, object],
    seed_nodes: dict[str, GraphNodeHit],
    seed_edges: dict[str, GraphEdgeHit],
    question: str,
    rationale: str,
) -> None:
    for hit in hits:
        metadata = hit.get("metadata", {})
        node_id = str(metadata.get("node_id", "")).strip()
        if node_id and node_id in node_index:
            _promote_seed_node(seed_nodes, node_index[node_id], hit, question, rationale)
            continue
        edge_id = str(metadata.get("edge_id", "")).strip()
        if edge_id and edge_id in edge_index:
            edge = edge_index[edge_id]
            seed_edges[edge_id] = GraphEdgeHit(
                edge_id=edge.edge_id,
                edge_type=edge.edge_type,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                score=float(hit.get("score", 0.0)),
                hop_distance=0,
                rationale=rationale,
            )
            if edge.source_node_id in node_index:
                _promote_seed_node(seed_nodes, node_index[edge.source_node_id], hit, question, rationale)
            if edge.target_node_id in node_index:
                _promote_seed_node(seed_nodes, node_index[edge.target_node_id], hit, question, rationale)


def _ingest_chunk_hits(
    chunk_hits: list[dict],
    artifact: SemanticArtifact,
    node_index: dict[str, object],
    seed_nodes: dict[str, GraphNodeHit],
    question: str,
) -> None:
    evidence_to_nodes: dict[str, list[str]] = {}
    for node in artifact.graph_nodes:
        for evidence_ref in node.evidence_refs:
            evidence_to_nodes.setdefault(evidence_ref, []).append(node.node_id)
    for hit in chunk_hits:
        metadata = hit.get("metadata", {})
        chunk_id = str(metadata.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        for node_id in evidence_to_nodes.get(chunk_id, []):
            node = node_index.get(node_id)
            if node is None:
                continue
            _promote_seed_node(seed_nodes, node, hit, question, "mix_chunk_seed")


def _promote_seed_node(
    seed_nodes: dict[str, GraphNodeHit],
    node,
    hit: dict,
    question: str,
    rationale: str,
) -> None:
    lexical = float(
        token_overlap_score(question, node.label)
        + token_overlap_score(question, node.canonical_text)
    )
    candidate = GraphNodeHit(
        node_id=node.node_id,
        node_type=node.node_type,
        label=node.label,
        canonical_text=node.canonical_text,
        score=float(hit.get("score", 0.0)),
        lexical_score=lexical,
        dense_score=float(hit.get("score", 0.0)),
        hop_distance=0,
        source_id=_nullable(node.metadata.get("source_id")),
        field_name=_nullable(node.metadata.get("field_name")),
        resolved_value=_nullable(node.metadata.get("resolved_value")),
        rationale=rationale,
    )
    previous = seed_nodes.get(node.node_id)
    if previous is None or candidate.score > previous.score:
        seed_nodes[node.node_id] = candidate


async def _build_context_sections(
    *,
    artifact: SemanticArtifact,
    stores: LightRAGQueryStores,
    node_hits: tuple[GraphNodeHit, ...],
    edge_hits: tuple[GraphEdgeHit, ...],
    chunk_hits: list[dict],
) -> tuple[str, ...]:
    node_index = {node.node_id: node for node in artifact.graph_nodes}
    sections: list[str] = []

    entity_lines: list[str] = []
    for hit in node_hits[:8]:
        node = node_index.get(hit.node_id)
        if node is None:
            continue
        entity_lines.append(f"{node.node_type}: {node.canonical_text}")
    if entity_lines:
        sections.append("Entities\n" + "\n".join(f"- {line}" for line in entity_lines))

    relation_lines: list[str] = []
    for hit in edge_hits[:8]:
        left = node_index.get(hit.source_node_id)
        right = node_index.get(hit.target_node_id)
        if left is None or right is None:
            continue
        relation_lines.append(
            f"{left.label} --{hit.edge_type}--> {right.label}"
        )
    if relation_lines:
        sections.append("Relations\n" + "\n".join(f"- {line}" for line in relation_lines))

    chunk_ids = [
        str(hit.get("metadata", {}).get("chunk_id", "")).strip()
        for hit in chunk_hits[:4]
        if str(hit.get("metadata", {}).get("chunk_id", "")).strip()
    ]
    if chunk_ids:
        chunk_payloads = await stores.text_chunks.get_by_ids(chunk_ids)
        chunk_lines = []
        for payload in chunk_payloads[:4]:
            content = " ".join(str(payload.get("content", "")).split())
            if not content:
                continue
            chunk_lines.append(content[:320])
        if chunk_lines:
            sections.append("Evidence\n" + "\n".join(f"- {line}" for line in chunk_lines))
    return tuple(sections)


def _nullable(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None
