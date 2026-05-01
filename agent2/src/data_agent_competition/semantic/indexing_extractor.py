from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from data_agent_competition.artifacts.schema import KnowledgeFact, SourceDescriptor
from data_agent_competition.semantic.indexing_schemas import (
    business_relation_type_enum,
    knowledge_relation_extraction_schema,
)
from data_agent_competition.semantic.indexing_validator import (
    ValidatedBusinessRelation,
    validate_business_relations,
)
from data_agent_competition.semantic.llm_support import SemanticRuntime

_SCHEMA_VERSION = "business_relation_extraction.v1"
_MAX_SECTION_FACTS = 18
_MAX_SECTION_CHARS = 2600

_SECTION_RELATION_TYPES: dict[str, tuple[str, ...]] = {
    "core_entities": (
        "business_alias_of_field",
        "value_concept_binds_field",
    ),
    "metrics": (
        "metric_depends_on_field",
        "metric_scoped_to_source",
    ),
    "constraints": (
        "constraint_applies_to_field",
        "constraint_applies_to_source",
    ),
    "use_cases": (
        "use_case_targets_source",
        "use_case_requires_metric",
        "use_case_requires_constraint",
        "doc_enrichment_matches_field",
    ),
    "ambiguity": (
        "ambiguity_prefers_field",
        "business_alias_of_field",
    ),
}

_SECTION_INSTRUCTIONS: dict[str, str] = {
    "core_entities": (
        "Focus on business aliases, concept-to-field bindings, and explicit semantic field names. "
        "Do not emit source targeting unless the fact clearly names a source and field."
    ),
    "metrics": (
        "Focus on metric formulas, required source scope, and which concrete fields a metric depends on."
    ),
    "constraints": (
        "Focus on business rules, formatting conventions, temporal restrictions, and the fields or sources they govern."
    ),
    "use_cases": (
        "Treat SQL snippets and examples as evidence only. Extract reusable routing patterns, required sources, metrics, constraints, and post-SQL enrichment matches."
    ),
    "ambiguity": (
        "Focus on preferred interpretations, field disambiguation, and competing aliases."
    ),
}


@dataclass(frozen=True, slots=True)
class BusinessRelationExtractionResult:
    relations: tuple[ValidatedBusinessRelation, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionScope:
    section: str
    relation_types: tuple[str, ...]
    facts: tuple[KnowledgeFact, ...]


def extract_business_relations(
    *,
    knowledge_facts: list[KnowledgeFact],
    sources: list[SourceDescriptor],
    runtime: SemanticRuntime | None,
) -> BusinessRelationExtractionResult:
    if runtime is None or not runtime.llm_client.enabled or not knowledge_facts or not sources:
        return BusinessRelationExtractionResult(relations=(), notes=("llm_indexing_skipped",))
    relations: list[ValidatedBusinessRelation] = []
    incomplete = False
    scopes = _extraction_scopes(knowledge_facts)
    source_context = _source_context(sources)
    notes = [
        "llm_indexing_enabled",
        f"schema_version={_SCHEMA_VERSION}",
        f"section_scope_count={len(scopes)}",
    ]
    for scope in scopes:
        payload = runtime.llm_client.call_structured(
            system_prompt=(
                "You are a semantic indexing engine for a data competition agent. "
                "Extract only business relations that can be grounded to the listed sources and fields. "
                "Do not invent sources or fields. Prefer empty output over weak guesses."
            ),
            user_prompt=_extraction_prompt(scope, source_context),
            function_name="extract_business_relations",
            schema=knowledge_relation_extraction_schema(),
        )
        batch_relations = validate_business_relations(
            payload=payload,
            knowledge_facts=knowledge_facts,
            sources=sources,
        )
        relations.extend(batch_relations)
        if payload and payload.get("complete") is False:
            incomplete = True
            continue_payload = runtime.llm_client.call_structured(
                system_prompt=(
                    "Continue extracting missed business relations from the same facts. "
                    "Return only new relations not already returned."
                ),
                user_prompt=_continue_prompt(scope, source_context, batch_relations),
                function_name="extract_business_relations",
                schema=knowledge_relation_extraction_schema(),
            )
            relations.extend(
                validate_business_relations(
                    payload=continue_payload,
                    knowledge_facts=knowledge_facts,
                    sources=sources,
                )
            )
    deduped = _dedupe_relations(relations)
    if incomplete:
        notes.append("continue_extraction_used")
    return BusinessRelationExtractionResult(relations=deduped, notes=tuple(notes))


def _extraction_scopes(
    knowledge_facts: list[KnowledgeFact],
) -> tuple[ExtractionScope, ...]:
    grouped: dict[str, list[KnowledgeFact]] = defaultdict(list)
    for fact in knowledge_facts:
        section = fact.tags[0] if fact.tags else "introduction"
        if section not in _SECTION_RELATION_TYPES:
            continue
        grouped[section].append(fact)

    scopes: list[ExtractionScope] = []
    for section in ("core_entities", "metrics", "constraints", "use_cases", "ambiguity"):
        section_facts = grouped.get(section, [])
        if not section_facts:
            continue
        relation_types = _SECTION_RELATION_TYPES[section]
        for fact_group in _chunk_section_facts(section_facts):
            scopes.append(
                ExtractionScope(
                    section=section,
                    relation_types=relation_types,
                    facts=fact_group,
                )
            )
    return tuple(scopes)


def _chunk_section_facts(
    section_facts: list[KnowledgeFact],
    *,
    max_facts: int = _MAX_SECTION_FACTS,
    max_chars: int = _MAX_SECTION_CHARS,
) -> tuple[tuple[KnowledgeFact, ...], ...]:
    chunks: list[tuple[KnowledgeFact, ...]] = []
    current: list[KnowledgeFact] = []
    current_chars = 0
    for fact in section_facts:
        fact_len = len(fact.statement) + len(fact.fact_id) + 16
        would_overflow = current and (
            len(current) >= max_facts or current_chars + fact_len > max_chars
        )
        if would_overflow:
            chunks.append(tuple(current))
            current = []
            current_chars = 0
        current.append(fact)
        current_chars += fact_len
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _source_context(sources: list[SourceDescriptor]) -> str:
    lines: list[str] = []
    for source in sources:
        field_list = ", ".join(field.field_name for field in source.fields[:16])
        lines.append(f"- {source.source_id}: {source.object_name} [{source.source_kind}] fields={field_list}")
    return "\n".join(lines)


def _extraction_prompt(scope: ExtractionScope, source_context: str) -> str:
    fact_lines = "\n".join(
        f"- {fact.fact_id}: [{','.join(fact.tags)}] {fact.statement}"
        for fact in scope.facts
    )
    relation_types = "\n".join(f"- {relation_type}" for relation_type in scope.relation_types)
    section_instruction = _SECTION_INSTRUCTIONS.get(
        scope.section,
        "Extract only high-confidence business relations grounded in schema.",
    )
    return (
        f"Schema version: {_SCHEMA_VERSION}\n"
        f"Knowledge section: {scope.section}\n"
        "Allowed relation types:\n"
        f"{relation_types}\n"
        "Rules:\n"
        "- Use from_node_label as a short canonical business concept name, not the whole sentence.\n"
        "- Use from_node_type and to_node_type from: metric, constraint, value_concept, ambiguity, use_case, source, field.\n"
        "- Prefer to_source_id and to_field_name when a relation points to a real schema field.\n"
        "- Use to_node_label and to_node_type only for business-node targets such as use_case->metric.\n"
        "- Set complete=true only when the listed facts are fully covered for this section.\n"
        "Section guidance:\n"
        f"{section_instruction}\n"
        "Available sources and fields:\n"
        f"{source_context}\n"
        "Knowledge facts:\n"
        f"{fact_lines}\n"
        "Extract business relations only when they bind to the available sources and fields."
    )


def _continue_prompt(
    scope: ExtractionScope,
    source_context: str,
    relations: tuple[ValidatedBusinessRelation, ...],
) -> str:
    previous = "\n".join(
        (
            f"- {relation.relation_type}: {relation.from_node_label} -> "
            f"{relation.to_source_id}.{relation.to_field_name}"
        )
        for relation in relations
    ) or "- none"
    return (
        _extraction_prompt(scope, source_context)
        + "\nAlready extracted relations:\n"
        + previous
        + "\nReturn only additional relations that are missing."
    )


def _dedupe_relations(
    relations: list[ValidatedBusinessRelation],
) -> tuple[ValidatedBusinessRelation, ...]:
    best: dict[tuple[str, str, str | None, str | None, str], ValidatedBusinessRelation] = {}
    for relation in relations:
        key = (
            relation.relation_type,
            relation.fact_id,
            relation.to_source_id,
            relation.to_field_name,
            relation.from_node_label,
        )
        previous = best.get(key)
        if previous is None or relation.confidence > previous.confidence:
            best[key] = relation
    return tuple(best.values())
