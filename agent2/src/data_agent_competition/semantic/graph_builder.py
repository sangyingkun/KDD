from __future__ import annotations

import re
from collections import defaultdict

from data_agent_competition.artifacts.schema import (
    DocumentChunk,
    GraphEdge,
    GraphNode,
    JoinCandidate,
    KnowledgeFact,
    RetrievalDocument,
    SourceDescriptor,
)
from data_agent_competition.semantic.graph_types import GraphEdgeType, GraphNodeType
from data_agent_competition.semantic.normalization import normalize_identifier

SECTION_HEADINGS: tuple[tuple[str, str], ...] = (
    ("1. introduction", "introduction"),
    ("2. core entities & fields", "core_entities"),
    ("3. metric definitions", "metrics"),
    ("4. constraints & conventions", "constraints"),
    ("5. exemplar use cases", "use_cases"),
    ("6. ambiguity resolution", "ambiguity"),
)

VALUE_MAPPING_PATTERN = re.compile(
    r"'([^']+)'\s+(?:indicating|means?|for|being)\s+([^.;]+?)(?=(?:\s+and\s+'|[.;]|$))",
    re.IGNORECASE,
)
FIELD_BULLET_PATTERN = re.compile(r"^\*\*([^(:]+)\(([^)]+)\)\:\*\*\s*(.+)$")
SQL_BLOCK_PATTERN = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def build_task_semantic_graph(
    *,
    sources: list[SourceDescriptor],
    knowledge_facts: list[KnowledgeFact],
    join_candidates: list[JoinCandidate],
    doc_chunks: list[DocumentChunk],
) -> tuple[list[GraphNode], list[GraphEdge], list[RetrievalDocument]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    node_ids: set[str] = set()
    edge_ids: set[str] = set()

    for node in _source_and_field_nodes(sources):
        _append_node(nodes, node_ids, node)
    for edge in _source_and_field_edges(sources):
        _append_edge(edges, edge_ids, edge)
    for node in _join_nodes(join_candidates):
        _append_node(nodes, node_ids, node)
    for edge in _join_edges(join_candidates):
        _append_edge(edges, edge_ids, edge)

    field_index = {
        (node.metadata.get("source_id"), node.metadata.get("field_name")): node.node_id
        for node in nodes
        if node.node_type == GraphNodeType.FIELD.value
    }
    source_nodes = {
        node.metadata.get("source_id"): node
        for node in nodes
        if node.node_type == GraphNodeType.SOURCE.value
    }
    knowledge_nodes, knowledge_edges = _knowledge_graph(knowledge_facts, field_index, source_nodes)
    for node in knowledge_nodes:
        _append_node(nodes, node_ids, node)
    for edge in knowledge_edges:
        _append_edge(edges, edge_ids, edge)

    for chunk_node, chunk_edges in _connect_doc_chunks(doc_chunks, nodes):
        _append_node(nodes, node_ids, chunk_node)
        for edge in chunk_edges:
            _append_edge(edges, edge_ids, edge)

    retrieval_documents = build_graph_retrieval_documents(nodes=nodes, edges=edges)
    return nodes, edges, retrieval_documents


def build_graph_retrieval_documents(
    *,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> list[RetrievalDocument]:
    adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_node_id].append(edge)
        adjacency[edge.target_node_id].append(edge)

    documents: list[RetrievalDocument] = []
    for node in nodes:
        neighbor_labels: list[str] = []
        for edge in adjacency.get(node.node_id, [])[:8]:
            other_id = edge.target_node_id if edge.source_node_id == node.node_id else edge.source_node_id
            neighbor_labels.append(other_id)
        metadata = {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "layer": str(node.metadata.get("layer", "")),
            "source_id": str(node.metadata.get("source_id", "")),
            "field_name": str(node.metadata.get("field_name", "")),
            "resolved_value": str(node.metadata.get("resolved_value", "")),
        }
        documents.append(
            RetrievalDocument(
                doc_id=f"graph::{node.node_id}",
                scope="task",
                doc_type=node.node_type,
                text=(
                    f"{node.label}. {node.canonical_text}. "
                    f"Metadata: {_render_metadata(node.metadata)}. "
                    f"Connected graph ids: {', '.join(neighbor_labels) if neighbor_labels else 'none'}."
                ),
                source_ref=node.node_id,
                metadata=metadata,
                evidence_refs=list(node.evidence_refs),
                confidence=node.confidence,
            )
        )
    for edge in edges:
        documents.append(
            RetrievalDocument(
                doc_id=f"graph-edge::{edge.edge_id}",
                scope="task",
                doc_type="relation",
                text=(
                    f"{edge.edge_type} between {edge.source_node_id} and {edge.target_node_id}. "
                    f"Evidence refs: {', '.join(edge.evidence_refs) if edge.evidence_refs else 'none'}."
                ),
                source_ref=edge.edge_id,
                metadata={
                    "edge_id": edge.edge_id,
                    "edge_type": edge.edge_type,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                },
                evidence_refs=list(edge.evidence_refs),
                confidence=edge.weight,
            )
        )
    return documents


def _source_and_field_nodes(sources: list[SourceDescriptor]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for source in sources:
        source_node_id = f"source::{source.source_id}"
        nodes.append(
            GraphNode(
                node_id=source_node_id,
                node_type=GraphNodeType.SOURCE.value,
                label=source.object_name,
                canonical_text=f"{source.object_name} in {source.asset_path}",
                metadata={
                    "layer": "physical",
                    "source_id": source.source_id,
                    "object_name": source.object_name,
                    "asset_path": source.asset_path,
                    "source_kind": source.source_kind,
                    "grain_hint": list(source.grain_hint),
                },
                evidence_refs=[source.source_id],
                confidence=0.95,
            )
        )
        for field in source.fields:
            nodes.append(
                GraphNode(
                    node_id=f"field::{source.source_id}.{field.field_name}",
                    node_type=GraphNodeType.FIELD.value,
                    label=field.field_name,
                    canonical_text=f"{source.object_name}.{field.field_name}",
                    metadata={
                        "layer": "physical",
                        "source_id": source.source_id,
                        "field_name": field.field_name,
                        "dtype": field.dtype,
                        "aliases": list(field.aliases),
                        "semantic_tags": list(field.semantic_tags),
                        "sample_values": list(field.sample_values[:5]),
                    },
                    evidence_refs=[source.source_id],
                    confidence=0.9,
                )
            )
            for alias in field.aliases:
                alias_norm = normalize_identifier(alias)
                field_norm = normalize_identifier(field.field_name)
                if not alias_norm or alias_norm == field_norm:
                    continue
                nodes.append(
                    GraphNode(
                        node_id=_alias_node_id(source.source_id, field.field_name, alias_norm),
                        node_type=GraphNodeType.AMBIGUITY.value,
                        label=alias,
                        canonical_text=f"Alias '{alias}' refers to {source.object_name}.{field.field_name}",
                        metadata={
                            "layer": "business",
                            "ambiguity_kind": "field_alias",
                            "source_id": source.source_id,
                            "field_name": field.field_name,
                            "alias": alias,
                            "preferred_source_id": source.source_id,
                            "preferred_field_name": field.field_name,
                        },
                        evidence_refs=[source.source_id],
                        confidence=0.78,
                    )
                )
    return nodes


def _source_and_field_edges(sources: list[SourceDescriptor]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for source in sources:
        source_node_id = f"source::{source.source_id}"
        for field in source.fields:
            field_node_id = f"field::{source.source_id}.{field.field_name}"
            edges.append(
                GraphEdge(
                    edge_id=f"has_field::{source.source_id}::{field.field_name}",
                    edge_type=GraphEdgeType.HAS_FIELD.value,
                    source_node_id=source_node_id,
                    target_node_id=field_node_id,
                    weight=1.0,
                    evidence_refs=[source.source_id],
                )
            )
            for alias in field.aliases:
                alias_norm = normalize_identifier(alias)
                field_norm = normalize_identifier(field.field_name)
                if not alias_norm or alias_norm == field_norm:
                    continue
                alias_node_id = _alias_node_id(source.source_id, field.field_name, alias_norm)
                edges.append(
                    GraphEdge(
                        edge_id=f"alias_of::{alias_node_id}::{field_node_id}",
                        edge_type=GraphEdgeType.ALIAS_OF.value,
                        source_node_id=alias_node_id,
                        target_node_id=field_node_id,
                        weight=0.85,
                        evidence_refs=[source.source_id],
                    )
                )
    return edges


def _join_nodes(join_candidates: list[JoinCandidate]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for join in join_candidates:
        nodes.append(
            GraphNode(
                node_id=_join_node_id(join),
                node_type=GraphNodeType.JOIN.value,
                label=f"{join.left_field}={join.right_field}",
                canonical_text=(
                    f"{join.left_source_id}.{join.left_field} -> "
                    f"{join.right_source_id}.{join.right_field}"
                ),
                metadata={
                    "layer": "physical",
                    "left_source_id": join.left_source_id,
                    "left_field": join.left_field,
                    "right_source_id": join.right_source_id,
                    "right_field": join.right_field,
                    "rationale": join.rationale,
                },
                evidence_refs=[join.left_source_id, join.right_source_id],
                confidence=join.confidence,
            )
        )
    return nodes


def _join_edges(join_candidates: list[JoinCandidate]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for join in join_candidates:
        join_id = _join_node_id(join)
        edges.append(
            GraphEdge(
                edge_id=f"join_on::{join_id}::left",
                edge_type=GraphEdgeType.JOIN_ON.value,
                source_node_id=join_id,
                target_node_id=f"field::{join.left_source_id}.{join.left_field}",
                weight=join.confidence,
                evidence_refs=[join.left_source_id],
            )
        )
        edges.append(
            GraphEdge(
                edge_id=f"join_on::{join_id}::right",
                edge_type=GraphEdgeType.JOIN_ON.value,
                source_node_id=join_id,
                target_node_id=f"field::{join.right_source_id}.{join.right_field}",
                weight=join.confidence,
                evidence_refs=[join.right_source_id],
            )
        )
    return edges


def _knowledge_graph(
    knowledge_facts: list[KnowledgeFact],
    field_index: dict[tuple[object, object], str],
    source_nodes: dict[object, GraphNode],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    current_section = "introduction"
    current_source_hint: str | None = None
    current_use_case_node_id: str | None = None
    field_node_lookup = {
        field_node_id: (str(source_id), str(field_name))
        for (source_id, field_name), field_node_id in field_index.items()
        if source_id and field_name
    }

    for fact in knowledge_facts:
        statement = fact.statement.strip()
        if not statement:
            continue
        lowered = normalize_identifier(statement).replace("_", " ")
        section = _resolve_section(lowered, current_section)
        if section != current_section:
            current_section = section
            current_source_hint = None
            if section != "use_cases":
                current_use_case_node_id = None
            continue
        source_heading = _source_heading(statement)
        if source_heading is not None:
            current_source_hint = source_heading
            continue

        node = _knowledge_node(
            fact=fact,
            section=current_section,
            source_hint=current_source_hint,
            use_case_node_id=current_use_case_node_id,
        )
        if node is None:
            continue
        node_edges = _knowledge_edges(
            node=node,
            statement=statement,
            section=current_section,
            field_index=field_index,
            source_nodes=source_nodes,
            active_use_case_node_id=current_use_case_node_id,
            source_hint=current_source_hint,
        )
        _annotate_node_bindings(node=node, edges=node_edges, field_node_lookup=field_node_lookup)
        nodes.append(node)
        nodes.extend(_knowledge_auxiliary_nodes(node=node, statement=statement, section=current_section))
        if node.node_type == GraphNodeType.USE_CASE.value:
            current_use_case_node_id = node.node_id
        edges.extend(node_edges)
    return nodes, edges


def _knowledge_node(
    *,
    fact: KnowledgeFact,
    section: str,
    source_hint: str | None,
    use_case_node_id: str | None,
) -> GraphNode | None:
    statement = fact.statement.strip()
    node_type = {
        "metrics": GraphNodeType.METRIC,
        "constraints": GraphNodeType.CONSTRAINT,
        "ambiguity": GraphNodeType.AMBIGUITY,
        "use_cases": GraphNodeType.USE_CASE,
        "core_entities": GraphNodeType.FIELD if _match_field_bullet(statement) else GraphNodeType.CONSTRAINT,
        "introduction": GraphNodeType.CONSTRAINT,
    }.get(section, GraphNodeType.CONSTRAINT)

    if VALUE_MAPPING_PATTERN.search(statement):
        node_type = GraphNodeType.VALUE_CONCEPT
    if statement.startswith("```") or statement == "SQL:":
        return None

    label = statement[:96]
    metadata: dict[str, object] = {
        "layer": "business",
        "source_path": fact.source_path,
        "section": section,
    }
    if source_hint:
        metadata["source_hint"] = source_hint
    sql_match = SQL_BLOCK_PATTERN.search(statement)
    if sql_match:
        metadata["sql"] = sql_match.group(1).strip()
    if section == "core_entities":
        field_match = _match_field_bullet(statement)
        if field_match:
            metadata["field_name_hint"] = field_match.group(1).strip()
            metadata["dtype_hint"] = field_match.group(2).strip()
            metadata["field_description"] = field_match.group(3).strip()
    if section == "metrics":
        metadata.update(_metric_metadata(statement))
    if section == "constraints":
        metadata.update(_constraint_metadata(statement))
    if section == "ambiguity":
        metadata.update(_ambiguity_metadata(statement))
    if use_case_node_id and section == "use_cases":
        metadata["parent_use_case_id"] = use_case_node_id

    return GraphNode(
        node_id=f"{node_type.value}::{fact.fact_id}",
        node_type=node_type.value,
        label=label,
        canonical_text=statement,
        metadata=metadata,
        evidence_refs=[fact.fact_id],
        confidence=_knowledge_confidence(section, statement),
    )


def _knowledge_edges(
    *,
    node: GraphNode,
    statement: str,
    section: str,
    field_index: dict[tuple[object, object], str],
    source_nodes: dict[object, GraphNode],
    active_use_case_node_id: str | None,
    source_hint: str | None,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    normalized_statement = normalize_identifier(statement)

    if section == "core_entities" and node.node_type == GraphNodeType.FIELD.value:
        for field_node_id in _bound_field_targets(node, field_index, source_nodes):
            edges.append(
                GraphEdge(
                    edge_id=f"{GraphEdgeType.ALIAS_OF.value}::{node.node_id}::{field_node_id}",
                    edge_type=GraphEdgeType.ALIAS_OF.value,
                    source_node_id=node.node_id,
                    target_node_id=field_node_id,
                    weight=0.92,
                    evidence_refs=list(node.evidence_refs),
                )
            )
    else:
        for (source_id, field_name), field_node_id in field_index.items():
            if not source_id or not field_name:
                continue
            field_norm = normalize_identifier(str(field_name))
            if not field_norm or field_norm not in normalized_statement:
                continue
            edge_type = GraphEdgeType.USES_FIELD.value
            if section == "constraints":
                edge_type = GraphEdgeType.CONSTRAINS.value
            edges.append(
                GraphEdge(
                    edge_id=f"{edge_type}::{node.node_id}::{field_node_id}",
                    edge_type=edge_type,
                    source_node_id=node.node_id,
                    target_node_id=field_node_id,
                    weight=0.85,
                    evidence_refs=list(node.evidence_refs),
                )
            )

    for source_id, source_node in source_nodes.items():
        if not source_id:
            continue
        if source_hint and _source_hint_matches(source_hint, str(source_id), source_node):
            edges.append(
                GraphEdge(
                    edge_id=f"uses_source::{node.node_id}::{source_node.node_id}",
                    edge_type=GraphEdgeType.USES_SOURCE.value,
                    source_node_id=node.node_id,
                    target_node_id=source_node.node_id,
                    weight=0.9,
                    evidence_refs=list(node.evidence_refs),
                )
            )
            continue
        if _source_matches_statement(normalized_statement, str(source_id), source_node):
            edges.append(
                GraphEdge(
                    edge_id=f"uses_source::{node.node_id}::{source_node.node_id}",
                    edge_type=GraphEdgeType.USES_SOURCE.value,
                    source_node_id=node.node_id,
                    target_node_id=source_node.node_id,
                    weight=0.8,
                    evidence_refs=list(node.evidence_refs),
                )
            )

    preferred_source_id = _string_metadata(node.metadata.get("preferred_source_id"))
    preferred_field_name = _string_metadata(node.metadata.get("preferred_field_name"))
    preferred_field_node_id = None
    if preferred_source_id and preferred_field_name:
        preferred_field_node_id = field_index.get((preferred_source_id, preferred_field_name))
        if preferred_field_node_id is not None:
            edges.append(
                GraphEdge(
                    edge_id=f"disambiguates::{node.node_id}::{preferred_field_node_id}",
                    edge_type=GraphEdgeType.DISAMBIGUATES.value,
                    source_node_id=node.node_id,
                    target_node_id=preferred_field_node_id,
                    weight=0.94,
                    evidence_refs=list(node.evidence_refs),
                )
            )

    if section == "ambiguity" and " vs " in statement.lower() and preferred_field_node_id is None:
        target_node_id = _ambiguity_target_node_id(node.node_id)
        if target_node_id:
            edges.append(
                GraphEdge(
                    edge_id=f"disambiguates::{node.node_id}::{target_node_id}",
                    edge_type=GraphEdgeType.DISAMBIGUATES.value,
                    source_node_id=node.node_id,
                    target_node_id=target_node_id,
                    weight=0.9,
                    evidence_refs=list(node.evidence_refs),
                )
            )

    for match in VALUE_MAPPING_PATTERN.finditer(statement):
        value = match.group(1).strip()
        concept = normalize_identifier(match.group(2))
        value_target_id = _value_target_node_id(node.node_id, concept, value)
        edges.append(
            GraphEdge(
                edge_id=f"maps_value::{node.node_id}::{concept}::{value}",
                edge_type=GraphEdgeType.MAPS_VALUE.value,
                source_node_id=node.node_id,
                target_node_id=value_target_id,
                weight=0.95,
                evidence_refs=list(node.evidence_refs),
            )
        )

    if section == "use_cases" and active_use_case_node_id and active_use_case_node_id != node.node_id:
        edges.append(
            GraphEdge(
                edge_id=f"co_occurs_with::{active_use_case_node_id}::{node.node_id}",
                edge_type=GraphEdgeType.CO_OCCURS_WITH.value,
                source_node_id=active_use_case_node_id,
                target_node_id=node.node_id,
                weight=0.7,
                evidence_refs=list(node.evidence_refs),
            )
        )
    return edges


def _connect_doc_chunks(
    doc_chunks: list[DocumentChunk],
    graph_nodes: list[GraphNode],
) -> list[tuple[GraphNode, list[GraphEdge]]]:
    results: list[tuple[GraphNode, list[GraphEdge]]] = []
    knowledge_nodes = [
        node
        for node in graph_nodes
        if node.node_type
        in {
            GraphNodeType.METRIC.value,
            GraphNodeType.CONSTRAINT.value,
            GraphNodeType.VALUE_CONCEPT.value,
            GraphNodeType.AMBIGUITY.value,
            GraphNodeType.USE_CASE.value,
        }
    ]
    for chunk in doc_chunks:
        chunk_node = GraphNode(
            node_id=f"chunk::{chunk.chunk_id}",
            node_type=GraphNodeType.CONSTRAINT.value,
            label=chunk.text[:96],
            canonical_text=chunk.text,
            metadata={"layer": "business", "asset_path": chunk.asset_path, "section": chunk.section or ""},
            evidence_refs=[chunk.chunk_id],
            confidence=0.45,
        )
        chunk_norm = normalize_identifier(chunk.text)
        chunk_edges: list[GraphEdge] = []
        for node in knowledge_nodes[:24]:
            label_norm = normalize_identifier(node.label)
            if not label_norm or label_norm not in chunk_norm:
                continue
            chunk_edges.append(
                GraphEdge(
                    edge_id=f"evidenced_by::{node.node_id}::{chunk_node.node_id}",
                    edge_type=GraphEdgeType.EVIDENCED_BY.value,
                    source_node_id=node.node_id,
                    target_node_id=chunk_node.node_id,
                    weight=0.65,
                    evidence_refs=[chunk.chunk_id],
                )
            )
        results.append((chunk_node, chunk_edges))
    return results


def _resolve_section(normalized_statement: str, current_section: str) -> str:
    for heading, section in SECTION_HEADINGS:
        if heading in normalized_statement:
            return section
    return current_section


def _source_heading(statement: str) -> str | None:
    stripped = statement.strip()
    if not stripped.startswith("### "):
        return None
    heading = stripped.removeprefix("### ").strip()
    if heading.lower().startswith("example ") or heading.lower().startswith("use case "):
        return None
    return heading


def _knowledge_confidence(section: str, statement: str) -> float:
    if section == "use_cases" and "sql" in statement.lower():
        return 0.72
    if section == "ambiguity":
        return 0.92
    if section == "constraints":
        return 0.88
    if section == "metrics":
        return 0.9
    if section == "core_entities":
        return 0.86
    return 0.55


def _join_node_id(join: JoinCandidate) -> str:
    return f"join::{join.left_source_id}.{join.left_field}::{join.right_source_id}.{join.right_field}"


def _alias_node_id(source_id: str, field_name: str, alias_norm: str) -> str:
    return f"ambiguity::{source_id}.{field_name}::{alias_norm}"


def _ambiguity_target_node_id(parent_node_id: str) -> str:
    return f"ambiguity-target::{parent_node_id}"


def _value_target_node_id(parent_node_id: str, concept: str, value: str) -> str:
    return f"value-target::{parent_node_id}::{concept}::{normalize_identifier(value)}"


def _knowledge_auxiliary_nodes(
    *,
    node: GraphNode,
    statement: str,
    section: str,
) -> list[GraphNode]:
    auxiliary_nodes: list[GraphNode] = []
    if section == "ambiguity":
        metadata = _ambiguity_metadata(statement)
        if metadata:
            auxiliary_nodes.append(
                GraphNode(
                    node_id=_ambiguity_target_node_id(node.node_id),
                    node_type=GraphNodeType.AMBIGUITY.value,
                    label=f"{metadata.get('left_option', '')} vs {metadata.get('right_option', '')}".strip(),
                    canonical_text=statement,
                    metadata={
                        "layer": "business",
                        "ambiguity_kind": "resolution_target",
                        "source_id": str(node.metadata.get("source_id", "")),
                        "field_name": str(node.metadata.get("field_name", "")),
                        "preferred_source_id": str(metadata.get("preferred_source_id", "")),
                        "preferred_field_name": str(metadata.get("preferred_field_name", "")),
                        **metadata,
                    },
                    evidence_refs=list(node.evidence_refs),
                    confidence=node.confidence,
                )
            )
    for match in VALUE_MAPPING_PATTERN.finditer(statement):
        value = match.group(1).strip()
        concept = normalize_identifier(match.group(2))
        auxiliary_nodes.append(
            GraphNode(
                node_id=_value_target_node_id(node.node_id, concept, value),
                node_type=GraphNodeType.VALUE_CONCEPT.value,
                label=value,
                canonical_text=f"{concept} => {value}",
                metadata={
                    "layer": "business",
                    "concept": concept,
                    "source_id": str(node.metadata.get("source_id", "")),
                    "field_name": str(node.metadata.get("field_name", "")),
                    "resolved_value": value,
                },
                evidence_refs=list(node.evidence_refs),
                confidence=node.confidence,
            )
        )
    return auxiliary_nodes


def _metric_metadata(statement: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    normalized = normalize_identifier(statement)
    field_refs = sorted(
        {
            token
            for token in re.findall(r"\b[a-z][a-z0-9_]*\b", normalized)
            if token not in {"the", "and", "for", "with", "from", "into", "must", "should", "using"}
        }
    )
    formula = None
    if "divide" in normalized:
        formula = "divide"
    elif "ratio" in normalized:
        formula = "ratio"
    elif "average" in normalized or "avg" in normalized:
        formula = "average"
    elif "sum" in normalized or "total" in normalized:
        formula = "sum"
    if formula:
        metadata["formula"] = formula
    if field_refs:
        metadata["field_terms"] = field_refs[:8]
    if "year" in normalized:
        metadata["requires_time_grain"] = "year"
    elif "month" in normalized:
        metadata["requires_time_grain"] = "month"
    return metadata


def _constraint_metadata(statement: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    normalized = normalize_identifier(statement)
    if "must" in normalized or "should" in normalized:
        metadata["constraint_type"] = "business_rule"
    if "date" in normalized or "year" in normalized or "month" in normalized:
        metadata["constraint_scope"] = "temporal"
    return metadata


def _ambiguity_metadata(statement: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    cleaned = statement.replace("**", "")
    if " vs. " in cleaned and ":" in cleaned:
        left, remainder = cleaned.split(" vs. ", maxsplit=1)
        right, preferred = remainder.split(":", maxsplit=1)
        metadata["left_option"] = left.strip()
        metadata["right_option"] = right.strip()
        metadata["preferred_interpretation"] = preferred.strip()
        preferred_norm = normalize_identifier(preferred)
        if "." in preferred_norm:
            preferred_source_id, preferred_field_name = preferred_norm.rsplit(".", maxsplit=1)
            metadata["preferred_source_id"] = preferred_source_id
            metadata["preferred_field_name"] = preferred_field_name
    return metadata


def _render_metadata(metadata: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in metadata.items():
        if value in ("", None, [], {}):
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "none"


def _match_field_bullet(statement: str) -> re.Match[str] | None:
    cleaned = statement.strip()
    if cleaned.startswith("- "):
        cleaned = cleaned[2:].strip()
    return FIELD_BULLET_PATTERN.match(cleaned)


def _bound_field_targets(
    node: GraphNode,
    field_index: dict[tuple[object, object], str],
    source_nodes: dict[object, GraphNode],
) -> list[str]:
    field_name_hint = _string_metadata(node.metadata.get("field_name_hint"))
    source_hint = _string_metadata(node.metadata.get("source_hint"))
    if not field_name_hint:
        return []
    field_norm = normalize_identifier(field_name_hint)
    targets: list[str] = []
    for (source_id, field_name), field_node_id in field_index.items():
        if not source_id or not field_name:
            continue
        if normalize_identifier(str(field_name)) != field_norm:
            continue
        if source_hint and not _source_hint_matches(source_hint, str(source_id), source_nodes.get(source_id)):
            continue
        targets.append(field_node_id)
    return targets


def _annotate_node_bindings(
    *,
    node: GraphNode,
    edges: list[GraphEdge],
    field_node_lookup: dict[str, tuple[str, str]],
) -> None:
    bound_fields: list[tuple[str, str]] = []
    bound_sources: list[str] = []
    for edge in edges:
        target_field = field_node_lookup.get(edge.target_node_id)
        if target_field is not None:
            bound_fields.append(target_field)
            bound_sources.append(target_field[0])
        if edge.edge_type == GraphEdgeType.USES_SOURCE.value and edge.target_node_id.startswith("source::"):
            bound_sources.append(edge.target_node_id.removeprefix("source::"))
    if bound_fields:
        node.metadata["bound_field_refs"] = list(dict.fromkeys(f"{source_id}.{field_name}" for source_id, field_name in bound_fields))
    if bound_sources:
        node.metadata["bound_source_ids"] = list(dict.fromkeys(bound_sources))
    if len(bound_fields) == 1:
        source_id, field_name = bound_fields[0]
        node.metadata.setdefault("source_id", source_id)
        node.metadata.setdefault("field_name", field_name)
    elif len(bound_sources) == 1:
        node.metadata.setdefault("source_id", bound_sources[0])


def _source_hint_matches(source_hint: str, source_id: str, source_node: GraphNode | None) -> bool:
    hint_norm = normalize_identifier(source_hint)
    if not hint_norm:
        return False
    candidates = {
        normalize_identifier(source_id),
        normalize_identifier(source_id.split("::")[-1]),
    }
    if source_node is not None:
        candidates.add(normalize_identifier(source_node.label))
        candidates.add(normalize_identifier(str(source_node.metadata.get("object_name", ""))))
        candidates.add(normalize_identifier(str(source_node.metadata.get("asset_path", ""))))
    return any(
        candidate
        and (candidate == hint_norm or hint_norm == candidate.split("_")[-1] or hint_norm in candidate)
        for candidate in candidates
    )


def _source_matches_statement(normalized_statement: str, source_id: str, source_node: GraphNode | None) -> bool:
    candidates = {
        normalize_identifier(source_id),
        normalize_identifier(source_id.split("::")[-1]),
    }
    if source_node is not None:
        candidates.add(normalize_identifier(source_node.label))
        candidates.add(normalize_identifier(str(source_node.metadata.get("object_name", ""))))
    return any(candidate and candidate in normalized_statement for candidate in candidates)


def _string_metadata(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _append_node(nodes: list[GraphNode], node_ids: set[str], node: GraphNode) -> None:
    if node.node_id in node_ids:
        return
    nodes.append(node)
    node_ids.add(node.node_id)


def _append_edge(edges: list[GraphEdge], edge_ids: set[str], edge: GraphEdge) -> None:
    if edge.edge_id in edge_ids:
        return
    edges.append(edge)
    edge_ids.add(edge.edge_id)
