from __future__ import annotations

from data_agent_competition.artifacts.schema import GraphEdge, GraphNode
from data_agent_competition.semantic.graph_types import GraphEdgeType, GraphNodeType
from data_agent_competition.semantic.indexing_validator import ValidatedBusinessRelation
from data_agent_competition.semantic.normalization import normalize_identifier


def merge_business_relations_into_graph(
    *,
    graph_nodes: list[GraphNode],
    graph_edges: list[GraphEdge],
    relations: tuple[ValidatedBusinessRelation, ...],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    if not relations:
        return graph_nodes, graph_edges
    node_ids = {node.node_id for node in graph_nodes}
    edge_ids = {edge.edge_id for edge in graph_edges}
    business_node_index = {
        (node.node_type, normalize_identifier(node.label)): node
        for node in graph_nodes
        if node.node_type
        in {
            GraphNodeType.METRIC.value,
            GraphNodeType.CONSTRAINT.value,
            GraphNodeType.VALUE_CONCEPT.value,
            GraphNodeType.AMBIGUITY.value,
            GraphNodeType.USE_CASE.value,
        }
    }
    field_node_index = {
        (str(node.metadata.get("source_id", "")), str(node.metadata.get("field_name", ""))): node.node_id
        for node in graph_nodes
        if node.node_type == GraphNodeType.FIELD.value
    }
    source_node_index = {
        str(node.metadata.get("source_id", "")): node.node_id
        for node in graph_nodes
        if node.node_type == GraphNodeType.SOURCE.value
    }

    for relation in relations:
        relation_node = _ensure_business_node(
            graph_nodes=graph_nodes,
            node_ids=node_ids,
            business_node_index=business_node_index,
            label=relation.from_node_label,
            node_type=_relation_node_type(relation.relation_type),
            relation=relation,
        )
        target_ref = _target_ref(
            relation,
            graph_nodes=graph_nodes,
            node_ids=node_ids,
            business_node_index=business_node_index,
            field_node_index=field_node_index,
            source_node_index=source_node_index,
        )
        if target_ref is None:
            continue
        edge = _relation_edge(relation, relation_node.node_id, target_ref)
        if edge.edge_id in edge_ids:
            continue
        graph_edges.append(edge)
        edge_ids.add(edge.edge_id)
    return graph_nodes, graph_edges


def _ensure_business_node(
    *,
    graph_nodes: list[GraphNode],
    node_ids: set[str],
    business_node_index: dict[tuple[str, str], GraphNode],
    label: str,
    node_type: str,
    relation: ValidatedBusinessRelation,
) -> GraphNode:
    normalized_label = normalize_identifier(label)
    existing = business_node_index.get((node_type, normalized_label))
    if existing is not None:
        _merge_node_metadata(existing, relation)
        return existing
    suffix = normalized_label or relation.fact_id
    node_id = f"{node_type}::llm::{suffix}"
    metadata: dict[str, object] = {
        "layer": "business",
        "fact_id": relation.fact_id,
        "relation_types": [relation.relation_type],
        "llm_indexed": True,
        "source_fact_ids": list(relation.source_fact_ids),
        "rationale": relation.rationale,
    }
    candidate = GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        canonical_text=relation.canonical_relation_text,
        metadata=metadata,
        evidence_refs=list(relation.source_fact_ids),
        confidence=relation.confidence,
    )
    _update_binding_metadata(candidate, relation)
    if candidate.node_id not in node_ids:
        graph_nodes.append(candidate)
        node_ids.add(candidate.node_id)
    business_node_index[(node_type, normalized_label)] = candidate
    return candidate


def _relation_edge(
    relation: ValidatedBusinessRelation,
    source_node_id: str,
    target_node_id: str,
) -> GraphEdge:
    edge_type = _relation_edge_type(relation.relation_type)
    return GraphEdge(
        edge_id=f"{edge_type}::{source_node_id}::{target_node_id}",
        edge_type=edge_type,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        weight=relation.confidence,
        evidence_refs=list(relation.source_fact_ids),
    )


def _target_ref(
    relation: ValidatedBusinessRelation,
    *,
    graph_nodes: list[GraphNode],
    node_ids: set[str],
    business_node_index: dict[tuple[str, str], GraphNode],
    field_node_index: dict[tuple[str, str], str],
    source_node_index: dict[str, str],
) -> str | None:
    if relation.to_source_id and relation.to_field_name:
        return field_node_index.get((relation.to_source_id, relation.to_field_name))
    if relation.to_source_id:
        return source_node_index.get(relation.to_source_id)
    if relation.to_node_label and relation.to_node_type:
        node = _ensure_business_node(
            graph_nodes=graph_nodes,
            node_ids=node_ids,
            business_node_index=business_node_index,
            label=relation.to_node_label,
            node_type=relation.to_node_type,
            relation=relation,
        )
        return node.node_id
    return None


def _relation_node_type(relation_type: str) -> str:
    if relation_type.startswith("metric_"):
        return GraphNodeType.METRIC.value
    if relation_type.startswith("constraint_"):
        return GraphNodeType.CONSTRAINT.value
    if relation_type.startswith("ambiguity_") or relation_type.startswith("business_alias"):
        return GraphNodeType.AMBIGUITY.value
    if relation_type.startswith("use_case_"):
        return GraphNodeType.USE_CASE.value
    return GraphNodeType.VALUE_CONCEPT.value


def _relation_edge_type(relation_type: str) -> str:
    if relation_type == "metric_depends_on_field":
        return GraphEdgeType.USES_FIELD.value
    if relation_type == "constraint_applies_to_field":
        return GraphEdgeType.CONSTRAINS.value
    if relation_type in {"metric_scoped_to_source", "constraint_applies_to_source", "use_case_targets_source"}:
        return GraphEdgeType.USES_SOURCE.value
    if relation_type == "ambiguity_prefers_field":
        return GraphEdgeType.DISAMBIGUATES.value
    if relation_type == "business_alias_of_field":
        return GraphEdgeType.ALIAS_OF.value
    if relation_type in {"value_concept_binds_field", "doc_enrichment_matches_field"}:
        return GraphEdgeType.MAPS_VALUE.value
    return GraphEdgeType.CO_OCCURS_WITH.value


def _merge_node_metadata(node: GraphNode, relation: ValidatedBusinessRelation) -> None:
    relation_types = [
        value
        for value in node.metadata.get("relation_types", [])
        if isinstance(value, str) and value.strip()
    ]
    if relation.relation_type not in relation_types:
        relation_types.append(relation.relation_type)
    node.metadata["relation_types"] = relation_types
    source_fact_ids = [
        value
        for value in node.metadata.get("source_fact_ids", [])
        if isinstance(value, str) and value.strip()
    ]
    for fact_id in relation.source_fact_ids:
        if fact_id not in source_fact_ids:
            source_fact_ids.append(fact_id)
    node.metadata["source_fact_ids"] = source_fact_ids
    if relation.rationale:
        existing_rationales = [
            value
            for value in node.metadata.get("rationales", [])
            if isinstance(value, str) and value.strip()
        ]
        if relation.rationale not in existing_rationales:
            existing_rationales.append(relation.rationale)
        node.metadata["rationales"] = existing_rationales
    _update_binding_metadata(node, relation)
    node.confidence = max(node.confidence, relation.confidence)
    for fact_id in relation.source_fact_ids:
        if fact_id not in node.evidence_refs:
            node.evidence_refs.append(fact_id)


def _update_binding_metadata(node: GraphNode, relation: ValidatedBusinessRelation) -> None:
    bound_field_refs = [
        value
        for value in node.metadata.get("bound_field_refs", [])
        if isinstance(value, str) and value.strip()
    ]
    if relation.to_source_id and relation.to_field_name:
        qualified = f"{relation.to_source_id}.{relation.to_field_name}"
        if qualified not in bound_field_refs:
            bound_field_refs.append(qualified)
    if bound_field_refs:
        node.metadata["bound_field_refs"] = bound_field_refs
    bound_source_ids = [
        value
        for value in node.metadata.get("bound_source_ids", [])
        if isinstance(value, str) and value.strip()
    ]
    if relation.to_source_id and relation.to_source_id not in bound_source_ids:
        bound_source_ids.append(relation.to_source_id)
    if bound_source_ids:
        node.metadata["bound_source_ids"] = bound_source_ids
    if len(bound_field_refs) == 1 and "." in bound_field_refs[0]:
        source_id, field_name = bound_field_refs[0].split(".", maxsplit=1)
        node.metadata.setdefault("source_id", source_id)
        node.metadata.setdefault("field_name", field_name)
    elif len(bound_source_ids) == 1:
        node.metadata.setdefault("source_id", bound_source_ids[0])
