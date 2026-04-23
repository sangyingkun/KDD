from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from data_agent_baseline.semantic.builder import _normalize_identifier
from data_agent_baseline.semantic.catalog import SemanticCatalog
from data_agent_baseline.semantic.embedding import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    doc_id: str
    scope: str
    doc_type: str
    text: str
    metadata: dict[str, Any]
    source_ref: str
    confidence: str
    evidence_refs: list[str]


@dataclass(slots=True)
class TaskRetrievalIndex:
    documents: list[RetrievalDocument]
    embeddings: np.ndarray
    by_id: dict[str, RetrievalDocument]


def _join_text(parts: list[str]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _compact_values(values: list[str], max_values: int = 6) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        compacted.append(text)
        if len(compacted) >= max_values:
            break
    return compacted


def _field_knowledge_hints(catalog: SemanticCatalog, field_name: str) -> list[str]:
    normalized_name = _normalize_identifier(field_name)
    hints: list[str] = []

    for rule in catalog.knowledge_contract.entity_field_rules:
        rule_field = _normalize_identifier(str(rule.get("field", "")))
        description = str(rule.get("description", "")).strip()
        if rule_field == normalized_name and description:
            hints.append(description)

    for rule in catalog.knowledge_contract.constraint_rules:
        rule_field = _normalize_identifier(str(rule.get("field", "")))
        raw_text = str(rule.get("raw_text", "")).strip()
        allowed_values = [str(value).strip() for value in rule.get("allowed_values", []) if str(value).strip()]
        if rule_field != normalized_name:
            continue
        if raw_text:
            hints.append(raw_text)
        elif allowed_values:
            hints.append(f"Allowed values: {', '.join(allowed_values[:6])}")

    for constraint in catalog.knowledge_contract.output_constraints:
        fields = [_normalize_identifier(str(field)) for field in constraint.get("fields", []) if str(field).strip()]
        concept = str(constraint.get("concept", "")).strip()
        if normalized_name in fields and concept:
            hints.append(f"Output constraint concept: {concept}")

    return _compact_values(hints, max_values=4)


def _field_document_metadata(spec, source_ref: str, knowledge_hints: list[str]) -> dict[str, Any]:
    data_type = getattr(spec, "data_type", getattr(spec, "value_type", "unknown"))
    return {
        "entity": spec.entity,
        "name": spec.name,
        "field_ref": spec.field_ref,
        "source": source_ref,
        "data_type": data_type,
        "semantic_type": getattr(spec, "semantic_type", "measure"),
        "aliases": list(getattr(spec, "aliases", [])),
        "description": str(getattr(spec, "description", "")).strip(),
        "sample_values": list(getattr(spec, "sample_values", [])),
        "value_range": getattr(spec, "value_range", None),
        "format_pattern": getattr(spec, "format_pattern", None),
        "knowledge_hints": list(knowledge_hints),
    }


def _entity_document(doc_id: str, entity) -> RetrievalDocument:
    text = _join_text(
        [
            f"entity: {entity.name}",
            f"aliases: {', '.join(entity.aliases) if entity.aliases else 'none'}",
            f"sources: {', '.join(entity.sources) if entity.sources else 'none'}",
            f"primary keys: {', '.join(entity.primary_keys) if entity.primary_keys else 'none'}",
            f"candidate keys: {', '.join(entity.candidate_keys) if entity.candidate_keys else 'none'}",
            f"description: {entity.description}",
        ]
    )
    source_ref = entity.sources[0] if entity.sources else entity.name
    return RetrievalDocument(
        doc_id=doc_id,
        scope="task",
        doc_type="entity",
        text=text,
        metadata={"entity": entity.name, "sources": list(entity.sources)},
        source_ref=source_ref,
        confidence=entity.confidence,
        evidence_refs=[],
    )


def _field_document(doc_id: str, *, doc_type: str, spec, catalog: SemanticCatalog) -> RetrievalDocument:
    sample_values = _compact_values(list(getattr(spec, "sample_values", [])))
    source_ref = spec.field_ref.split("::", 1)[0]
    knowledge_hints = _field_knowledge_hints(catalog, spec.name)
    description = str(getattr(spec, "description", "")).strip() or f"{spec.name} field on {spec.entity}"
    value_range = getattr(spec, "value_range", None) or "none"
    format_pattern = getattr(spec, "format_pattern", None) or "none"
    aliases = list(getattr(spec, "aliases", []))
    data_type = getattr(spec, "data_type", getattr(spec, "value_type", "unknown"))
    text = _join_text(
        [
            f"entity: {spec.entity}",
            f"{doc_type}: {spec.name}",
            f"field_ref: {spec.field_ref}",
            f"data_type: {data_type}",
            f"semantic_type: {getattr(spec, 'semantic_type', 'measure')}",
            f"aliases: {', '.join(aliases) if aliases else 'none'}",
            f"description: {description}",
            f"value_range: {value_range}",
            f"format_pattern: {format_pattern}",
            f"sample_values: {', '.join(sample_values) if sample_values else 'none'}",
            f"knowledge_hints: {' | '.join(knowledge_hints) if knowledge_hints else 'none'}",
            f"source: {source_ref}",
        ]
    )
    return RetrievalDocument(
        doc_id=doc_id,
        scope="task",
        doc_type=doc_type,
        text=text,
        metadata=_field_document_metadata(spec, source_ref, knowledge_hints),
        source_ref=source_ref,
        confidence=spec.confidence,
        evidence_refs=[],
    )


def _relation_document(doc_id: str, relation) -> RetrievalDocument:
    join_keys = ", ".join(
        f"{pair.left_field}->{pair.right_field}" for pair in relation.join_keys
    ) or "none"
    source_ref = f"{relation.left_entity}->{relation.right_entity}"
    text = _join_text(
        [
            f"relation: {relation.left_entity} -> {relation.right_entity}",
            f"join_keys: {join_keys}",
            f"cardinality: {relation.cardinality}",
            f"description: {relation.description}",
        ]
    )
    return RetrievalDocument(
        doc_id=doc_id,
        scope="task",
        doc_type="relation",
        text=text,
        metadata={
            "left_entity": relation.left_entity,
            "right_entity": relation.right_entity,
            "cardinality": relation.cardinality,
        },
        source_ref=source_ref,
        confidence=relation.confidence,
        evidence_refs=[],
    )


def _knowledge_documents(catalog: SemanticCatalog) -> list[RetrievalDocument]:
    documents: list[RetrievalDocument] = []
    knowledge = catalog.knowledge_contract
    for section_name, text in knowledge.sections.items():
        documents.append(
            RetrievalDocument(
                doc_id=f"knowledge::{section_name}",
                scope="task",
                doc_type="knowledge",
                text=_join_text([f"section: {section_name}", text]),
                metadata={"section": section_name},
                source_ref="knowledge.md",
                confidence="medium",
                evidence_refs=[],
            )
        )
    for evidence in catalog.evidence:
        documents.append(
            RetrievalDocument(
                doc_id=f"evidence::{evidence.id}",
                scope="task",
                doc_type="knowledge",
                text=_join_text(
                    [
                        f"claim: {evidence.claim}",
                        f"source_file: {evidence.source_file}",
                        f"location: {evidence.location_hint}",
                        f"snippet: {evidence.snippet}",
                    ]
                ),
                metadata={"source_type": evidence.source_type, "source_file": evidence.source_file},
                source_ref=evidence.source_file,
                confidence=evidence.confidence,
                evidence_refs=[evidence.id],
            )
        )
    return documents


def _source_item_document(doc_id: str, item) -> RetrievalDocument:
    return RetrievalDocument(
        doc_id=doc_id,
        scope="task",
        doc_type="source_item",
        text=item.retrieval_text,
        metadata={
            "entity": item.entity,
            "item_type": item.item_type,
            "source_type": item.source_type,
            "source_file": item.source_file,
            "source_path": item.source_path,
            "field_ref": item.field_ref,
            "semantic_role": item.semantic_role,
            "anchor_names": list(item.anchor_names),
            "aliases": list(item.aliases),
            **dict(item.metadata),
        },
        source_ref=item.source_file,
        confidence=item.confidence,
        evidence_refs=[],
    )


def _cross_source_anchor_document(doc_id: str, anchor) -> RetrievalDocument:
    text = _join_text(
        [
            f"anchor_name: {anchor.anchor_name}",
            f"source_files: {', '.join(anchor.source_files) if anchor.source_files else 'none'}",
            f"members: {', '.join(anchor.members) if anchor.members else 'none'}",
            f"description: {anchor.description}",
        ]
    )
    return RetrievalDocument(
        doc_id=doc_id,
        scope="task",
        doc_type="anchor",
        text=text,
        metadata=dict(anchor.metadata) | {"anchor_name": anchor.anchor_name, "source_files": list(anchor.source_files)},
        source_ref="cross_source_anchor",
        confidence=anchor.confidence,
        evidence_refs=[],
    )


def _routing_rule_document(doc_id: str, rule) -> RetrievalDocument:
    text = _join_text(
        [
            f"rule_type: {rule.rule_type}",
            f"description: {rule.description}",
            f"condition: {rule.condition}",
            f"target_sources: {', '.join(rule.target_sources) if rule.target_sources else 'none'}",
            f"anchor_names: {', '.join(rule.anchor_names) if rule.anchor_names else 'none'}",
        ]
    )
    return RetrievalDocument(
        doc_id=doc_id,
        scope="task",
        doc_type="routing_rule",
        text=text,
        metadata=dict(rule.metadata) | {
            "rule_id": rule.rule_id,
            "rule_type": rule.rule_type,
            "source_file": rule.source_file,
            "target_sources": list(rule.target_sources),
            "anchor_names": list(rule.anchor_names),
        },
        source_ref=rule.source_file,
        confidence=rule.confidence,
        evidence_refs=[],
    )


def build_retrieval_documents(catalog: SemanticCatalog) -> list[RetrievalDocument]:
    documents: list[RetrievalDocument] = []
    for index, entity in enumerate(catalog.entities, start=1):
        documents.append(_entity_document(f"entity::{index}::{entity.name}", entity))
    for index, dimension in enumerate(catalog.dimensions, start=1):
        documents.append(
            _field_document(
                f"dimension::{index}::{dimension.name}",
                doc_type="field",
                spec=dimension,
                catalog=catalog,
            )
        )
    for index, measure in enumerate(catalog.measures, start=1):
        documents.append(
            _field_document(
                f"measure::{index}::{measure.name}",
                doc_type="field",
                spec=measure,
                catalog=catalog,
            )
        )
    for index, relation in enumerate(catalog.relations, start=1):
        documents.append(_relation_document(f"relation::{index}", relation))
    for index, source_item in enumerate(catalog.source_items, start=1):
        documents.append(_source_item_document(f"source_item::{index}::{source_item.item_id}", source_item))
    for index, anchor in enumerate(catalog.cross_source_anchors, start=1):
        documents.append(_cross_source_anchor_document(f"anchor::{index}::{anchor.anchor_name}", anchor))
    for index, rule in enumerate(catalog.routing_rules, start=1):
        documents.append(_routing_rule_document(f"routing_rule::{index}::{rule.rule_id}", rule))
    documents.extend(_knowledge_documents(catalog))
    return documents


def build_task_retrieval_index(
    catalog: SemanticCatalog,
    provider: EmbeddingProvider,
) -> TaskRetrievalIndex:
    documents = build_retrieval_documents(catalog)
    embeddings = provider.embed_texts([document.text for document in documents])
    return TaskRetrievalIndex(
        documents=documents,
        embeddings=embeddings,
        by_id={document.doc_id: document for document in documents},
    )


def _cosine_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Retrieval embeddings must be a 2D matrix.")
    if query_vector.ndim != 1:
        raise ValueError("Query embedding must be a 1D vector.")
    if matrix.shape[1] != query_vector.shape[0]:
        raise ValueError(
            f"Embedding dimension mismatch: documents={matrix.shape[1]} query={query_vector.shape[0]}"
        )
    return matrix @ query_vector


def retrieve_dense(
    index: TaskRetrievalIndex,
    provider: EmbeddingProvider,
    query: str,
    top_k: int = 8,
) -> list[tuple[RetrievalDocument, float]]:
    if top_k < 1 or not index.documents:
        return []
    query_vector = provider.embed_query(query)
    scores = _cosine_scores(index.embeddings, query_vector)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [(index.documents[int(item)], float(scores[int(item)])) for item in ranked_indices]


def _lexical_score(document: RetrievalDocument, normalized_query: str, query_tokens: set[str]) -> float:
    normalized_text = _normalize_identifier(document.text)
    if not normalized_text:
        return 0.0
    if normalized_query and normalized_query in normalized_text:
        return 1.0
    doc_tokens = {token for token in normalized_text.split("_") if token}
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / max(len(query_tokens), 1)


def retrieve_lexical(
    index: TaskRetrievalIndex,
    query: str,
    top_k: int = 8,
) -> list[tuple[RetrievalDocument, float]]:
    if top_k < 1 or not index.documents:
        return []
    normalized_query = _normalize_identifier(query)
    query_tokens = {token for token in normalized_query.split("_") if token}
    scored: list[tuple[RetrievalDocument, float]] = []
    for document in index.documents:
        score = _lexical_score(document, normalized_query, query_tokens)
        if score > 0.0:
            scored.append((document, score))
    scored.sort(key=lambda item: (-item[1], item[0].doc_id))
    return scored[:top_k]
