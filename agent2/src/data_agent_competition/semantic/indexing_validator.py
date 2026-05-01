from __future__ import annotations

from dataclasses import dataclass

from data_agent_competition.artifacts.schema import KnowledgeFact, SourceDescriptor
from data_agent_competition.semantic.graph_types import GraphNodeType
from data_agent_competition.semantic.indexing_schemas import business_relation_type_enum
from data_agent_competition.semantic.normalization import normalize_identifier


@dataclass(frozen=True, slots=True)
class ValidatedBusinessRelation:
    relation_type: str
    fact_id: str
    source_fact_ids: tuple[str, ...]
    from_node_label: str
    from_node_type: str
    to_source_id: str | None
    to_field_name: str | None
    to_node_label: str | None
    to_node_type: str | None
    canonical_relation_text: str
    rationale: str
    confidence: float


def validate_business_relations(
    *,
    payload: dict[str, object] | None,
    knowledge_facts: list[KnowledgeFact],
    sources: list[SourceDescriptor],
) -> tuple[ValidatedBusinessRelation, ...]:
    if not payload:
        return ()
    fact_ids = {fact.fact_id for fact in knowledge_facts}
    source_map = {source.source_id: source for source in sources}
    field_pairs = {
        (source.source_id, field.field_name)
        for source in sources
        for field in source.fields
    }
    valid_relation_types = set(business_relation_type_enum())
    valid_node_types = {item.value for item in GraphNodeType}
    relations: list[ValidatedBusinessRelation] = []
    seen: set[tuple[str, str, str | None, str | None, str | None]] = set()
    for raw_item in payload.get("relations", []):
        if not isinstance(raw_item, dict):
            continue
        relation_type = _clean(raw_item.get("relation_type"))
        fact_id = _clean(raw_item.get("fact_id"))
        from_node_label = _clean(raw_item.get("from_node_label"))
        from_node_type = _clean(raw_item.get("from_node_type"))
        canonical_relation_text = _clean(raw_item.get("canonical_relation_text"))
        rationale = _clean(raw_item.get("rationale"))
        if not all((relation_type, fact_id, from_node_label, from_node_type, canonical_relation_text)):
            continue
        if relation_type not in valid_relation_types:
            continue
        if fact_id not in fact_ids:
            continue
        if from_node_type not in valid_node_types:
            continue
        source_fact_ids = tuple(
            item
            for item in (
                _clean(value)
                for value in raw_item.get("source_fact_ids", [])
                if isinstance(value, str)
            )
            if item in fact_ids
        )
        to_source_id = _clean(raw_item.get("to_source_id"))
        to_field_name = _clean(raw_item.get("to_field_name"))
        to_node_label = _clean(raw_item.get("to_node_label"))
        to_node_type = _clean(raw_item.get("to_node_type"))
        if to_source_id and to_source_id not in source_map:
            continue
        if to_source_id and to_field_name and (to_source_id, to_field_name) not in field_pairs:
            continue
        if to_node_type and to_node_type not in valid_node_types:
            continue
        confidence = _confidence(raw_item.get("confidence"))
        dedupe_key = (
            relation_type,
            fact_id,
            to_source_id,
            to_field_name,
            normalize_identifier(to_node_label or ""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        relations.append(
            ValidatedBusinessRelation(
                relation_type=relation_type,
                fact_id=fact_id,
                source_fact_ids=source_fact_ids or (fact_id,),
                from_node_label=from_node_label,
                from_node_type=from_node_type,
                to_source_id=to_source_id,
                to_field_name=to_field_name,
                to_node_label=to_node_label,
                to_node_type=to_node_type,
                canonical_relation_text=canonical_relation_text,
                rationale=rationale,
                confidence=confidence,
            )
        )
    return tuple(relations)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _confidence(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, 1.0))
