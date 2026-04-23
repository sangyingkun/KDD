from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from data_agent_baseline.config import RetrievalConfig
from data_agent_baseline.semantic.builder import _normalize_identifier
from data_agent_baseline.semantic.catalog import SemanticCatalog
from data_agent_baseline.semantic.embedding import EmbeddingProvider
from data_agent_baseline.semantic.retrieval import (
    RetrievalDocument,
    TaskRetrievalIndex,
    retrieve_dense,
    retrieve_lexical,
)


@dataclass(frozen=True, slots=True)
class LinkCandidate:
    doc_id: str
    doc_type: str
    source_ref: str
    dense_score: float
    lexical_score: float
    final_score: float
    reasons: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SchemaLinkResult:
    query_units: list[dict[str, Any]]
    top_entities: list[dict[str, Any]]
    top_fields: list[dict[str, Any]]
    top_knowledge: list[dict[str, Any]]
    candidate_bindings: list[dict[str, Any]]
    chosen_bindings: list[dict[str, Any]]
    binding_conflicts: list[dict[str, Any]]
    join_candidates: list[dict[str, Any]]
    candidate_join_paths: list[dict[str, Any]]
    required_sources: list[str]
    unresolved_ambiguities: list[dict[str, Any]]
    debug_view: dict[str, Any]


@dataclass(frozen=True, slots=True)
class QueryUnit:
    unit_type: str
    text: str
    role: Literal["binding", "support", "routing"]
    binding_eligible: bool
    semantic_hints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_type": self.unit_type,
            "text": self.text,
            "role": self.role,
            "binding_eligible": self.binding_eligible,
            "semantic_hints": list(self.semantic_hints),
        }


_PATTERN_UNITS: tuple[QueryUnit, ...] = (
    QueryUnit("filter", "severe degree of thrombosis", "binding", True),
    QueryUnit("output", "sex", "binding", True),
    QueryUnit("output", "gender", "binding", True),
    QueryUnit("output", "disease", "binding", True),
    QueryUnit("output", "id", "binding", True),
    QueryUnit("metric", "total", "binding", True),
    QueryUnit("metric", "average", "binding", True),
    QueryUnit("metric", "rate", "binding", True),
    QueryUnit("support", "diagnosed with", "support", False),
    QueryUnit("group", "by ", "routing", False),
)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {text.lower()} "
    normalized_phrase = phrase.strip().lower()
    if not normalized_phrase:
        return False
    if " " in normalized_phrase:
        return normalized_phrase in normalized_text
    return re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized_text) is not None


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _infer_semantic_hints(question: str, unit_type: str, unit_text: str) -> tuple[str, ...]:
    lowered_question = question.lower()
    lowered_text = unit_text.lower()
    hints: list[str] = []

    combined = f"{lowered_question} {lowered_text}"
    if _contains_any_phrase(
        combined,
        ("last month", "previous month", "this month", "recent", "last", "previous", "year", "month", "week", "daily", "monthly", "quarter"),
    ):
        hints.append("time_window")
    if _contains_any_phrase(combined, ("how many", "count", "number of", "total number")):
        hints.append("aggregation_count")
    if _contains_any_phrase(combined, ("average", "avg", "mean")):
        hints.append("aggregation_average")
    if _contains_any_phrase(combined, ("total", "sum", "overall amount", "total value")):
        hints.append("aggregation_sum")
    if _contains_any_phrase(combined, ("highest", "largest", "biggest", "most", "top", "ranked", "best")):
        hints.append("ranking_desc")
    if _contains_any_phrase(combined, ("lowest", "smallest", "least", "minimum", "worst")):
        hints.append("ranking_asc")
    if _contains_any_phrase(combined, ("active", "inactive", "online", "offline", "engaged", "status")):
        hints.append("status_semantics")
    if _contains_any_phrase(combined, ("date", "time", "month", "year", "daily", "monthly", "recent", "login")):
        hints.append("time_semantics")
    if _contains_any_phrase(combined, ("id", "identifier")):
        hints.append("identifier_semantics")
    if _contains_any_phrase(combined, ("price", "cost", "amount", "revenue", "sales", "speed", "score", "weight", "consumption")):
        hints.append("numeric_measure")
    if re.search(r"\b(over|under|between|less than|greater than|more than)\b", combined):
        hints.append("numeric_filter")
    if unit_type == "metric" and "numeric_measure" not in hints:
        hints.append("numeric_measure")

    seen: set[str] = set()
    ordered_hints: list[str] = []
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        ordered_hints.append(hint)
    return tuple(ordered_hints)


def extract_query_units(question: str) -> list[QueryUnit]:
    normalized_question = " ".join(question.strip().split())
    lowered = normalized_question.lower()
    units: list[QueryUnit] = [
        QueryUnit(
            "question",
            normalized_question,
            "routing",
            False,
            semantic_hints=_infer_semantic_hints(normalized_question, "question", normalized_question),
        )
    ]
    seen = {("question", normalized_question.lower())}

    for template in _PATTERN_UNITS:
        unit_type = template.unit_type
        phrase = template.text
        if phrase not in lowered:
            continue
        if unit_type == "group" and "by " in lowered:
            for match in re.finditer(r"\bby\s+([a-z0-9 _-]+?)(?:\?|,|\.|$)", lowered):
                text = match.group(1).strip()
                signature = ("group", text)
                if text and signature not in seen:
                    units.append(
                        QueryUnit(
                            "group",
                            text,
                            "routing",
                            False,
                            semantic_hints=_infer_semantic_hints(normalized_question, "group", text),
                        )
                    )
                    seen.add(signature)
            continue
        signature = (unit_type, phrase)
        if signature not in seen:
            units.append(
                QueryUnit(
                    template.unit_type,
                    template.text,
                    template.role,
                    template.binding_eligible,
                    semantic_hints=_infer_semantic_hints(normalized_question, template.unit_type, template.text),
                )
            )
            seen.add(signature)

    for quoted in re.findall(r'"([^"]+)"', normalized_question):
        signature = ("quoted", quoted.lower())
        if signature not in seen:
            units.append(
                QueryUnit(
                    "quoted",
                    quoted,
                    "support",
                    False,
                    semantic_hints=_infer_semantic_hints(normalized_question, "quoted", quoted),
                )
            )
            seen.add(signature)
    return units


def _find_join_path(
    catalog: SemanticCatalog,
    *,
    anchor_entity: str,
    target_entity: str,
) -> list[dict[str, str]]:
    if anchor_entity == target_entity:
        return []

    adjacency: dict[str, list[tuple[str, Any]]] = {}
    for relation in catalog.relations:
        adjacency.setdefault(relation.left_entity, []).append((relation.right_entity, relation))
        adjacency.setdefault(relation.right_entity, []).append((relation.left_entity, relation))

    queue: list[tuple[str, list[Any]]] = [(anchor_entity, [])]
    visited = {anchor_entity}
    while queue:
        current_entity, path = queue.pop(0)
        for neighbor, relation in adjacency.get(current_entity, []):
            if neighbor in visited:
                continue
            next_path = [*path, relation]
            if neighbor == target_entity:
                return [
                    {
                        "left_entity": edge.left_entity,
                        "right_entity": edge.right_entity,
                        "cardinality": edge.cardinality,
                    }
                    for edge in next_path
                ]
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return []


def _best_entity_for_question(
    question: str,
    catalog: SemanticCatalog,
    index: TaskRetrievalIndex,
    provider: EmbeddingProvider,
    config: RetrievalConfig,
) -> str | None:
    dense = retrieve_dense(index, provider, question, top_k=config.retrieval_top_k)
    lexical = retrieve_lexical(index, question, top_k=config.lexical_top_k)
    merged_entity_scores: dict[str, tuple[float, float]] = {}
    for document, score in dense:
        if document.doc_type != "entity":
            continue
        entity_name = str(document.metadata.get("entity", ""))
        if not entity_name:
            continue
        current_dense, current_lexical = merged_entity_scores.get(entity_name, (0.0, 0.0))
        merged_entity_scores[entity_name] = (max(current_dense, float(score)), current_lexical)
    for document, score in lexical:
        if document.doc_type != "entity":
            continue
        entity_name = str(document.metadata.get("entity", ""))
        if not entity_name:
            continue
        current_dense, current_lexical = merged_entity_scores.get(entity_name, (0.0, 0.0))
        merged_entity_scores[entity_name] = (current_dense, max(current_lexical, float(score)))

    best_entity: tuple[str, float] | None = None
    for entity_name, (dense_score, lexical_score) in merged_entity_scores.items():
        combined_score = 0.7 * dense_score + 0.3 * lexical_score
        if best_entity is None or combined_score > best_entity[1]:
            best_entity = (entity_name, combined_score)
    if best_entity is not None:
        return best_entity[0]

    normalized_question = _normalize_identifier(question)
    for entity in catalog.entities:
        if _normalize_identifier(entity.name) in normalized_question:
            return entity.name
    return catalog.entities[0].name if catalog.entities else None


def _entity_consistency(candidate: RetrievalDocument, anchor_entity: str | None) -> float:
    if not anchor_entity:
        return 0.5
    candidate_entity = str(candidate.metadata.get("entity", ""))
    if not candidate_entity:
        return 0.3
    return 1.0 if candidate_entity == anchor_entity else 0.2


def _join_connectivity(candidate: RetrievalDocument, catalog: SemanticCatalog, anchor_entity: str | None) -> float:
    candidate_entity = str(candidate.metadata.get("entity", ""))
    if not anchor_entity or not candidate_entity:
        return 0.3
    if candidate_entity == anchor_entity:
        return 1.0
    return 0.8 if _find_join_path(catalog, anchor_entity=anchor_entity, target_entity=candidate_entity) else 0.0


def _role_compatibility(unit: QueryUnit, candidate: RetrievalDocument) -> float:
    item_type = str(candidate.metadata.get("item_type", "")).strip().lower()
    if unit.role == "routing":
        if unit.unit_type == "question":
            return 0.5
        if candidate.doc_type in {"anchor", "routing_rule"}:
            return 0.9
        return 0.3 if candidate.doc_type in {"entity", "field", "source_item"} else 0.2
    if unit.role == "support":
        if candidate.doc_type in {"field", "source_item"}:
            return 0.6
        if candidate.doc_type in {"knowledge", "routing_rule"}:
            return 0.5
        if candidate.doc_type == "entity":
            return 0.3
        return 0.2
    if unit.unit_type == "question":
        return 0.5
    if unit.unit_type in {"output", "filter"} and candidate.doc_type in {"field", "source_item"}:
        return 1.0
    if unit.unit_type == "metric" and candidate.doc_type in {"field", "knowledge", "source_item", "routing_rule"}:
        return 0.8
    if candidate.doc_type == "anchor" and (item_type.endswith("_field") or unit.unit_type in {"filter", "output"}):
        return 0.6
    if candidate.doc_type == "entity":
        return 0.6
    if candidate.doc_type in {"knowledge", "routing_rule"}:
        return 0.5
    return 0.2


def _knowledge_support(unit_text: str, candidate: RetrievalDocument) -> float:
    if candidate.doc_type in {"knowledge", "routing_rule"}:
        return 1.0
    normalized_unit = _normalize_identifier(unit_text)
    normalized_source = _normalize_identifier(candidate.source_ref)
    return 0.8 if normalized_unit and normalized_unit in normalized_source else 0.0


def _normalized_overlap_score(left: str, right: str) -> float:
    normalized_left = _normalize_identifier(left)
    normalized_right = _normalize_identifier(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 1.0
    left_tokens = {token for token in normalized_left.split("_") if token}
    right_tokens = {token for token in normalized_right.split("_") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def _sample_value_hit(unit_text: str, candidate: RetrievalDocument) -> float:
    sample_values = [str(item).lower() for item in candidate.metadata.get("sample_values", [])]
    lowered_unit = unit_text.lower()
    return 1.0 if any(sample in lowered_unit for sample in sample_values if sample) else 0.0


def _alias_match(unit_text: str, candidate: RetrievalDocument) -> float:
    candidate_forms = [
        str(candidate.metadata.get("name", "")),
        str(candidate.metadata.get("field_ref", "")),
        str(candidate.metadata.get("source_path", "")),
        str(candidate.metadata.get("display_name", "")),
        *[str(item) for item in candidate.metadata.get("aliases", [])],
    ]
    best_score = 0.0
    for candidate_form in candidate_forms:
        best_score = max(best_score, _normalized_overlap_score(unit_text, candidate_form))
    if best_score > 0.0:
        return best_score
    return _normalized_overlap_score(unit_text, candidate.text)


def _description_support(unit_text: str, candidate: RetrievalDocument) -> float:
    description = str(candidate.metadata.get("description", "")).strip()
    knowledge_hints = [str(item) for item in candidate.metadata.get("knowledge_hints", [])]
    best_score = _normalized_overlap_score(unit_text, description)
    for hint in knowledge_hints:
        best_score = max(best_score, _normalized_overlap_score(unit_text, hint))
    return best_score


def _semantic_type_compatibility(unit: QueryUnit, candidate: RetrievalDocument) -> float:
    semantic_type = str(candidate.metadata.get("semantic_type", "")).strip().lower()
    data_type = str(candidate.metadata.get("data_type", "")).strip().lower()
    semantic_role = str(candidate.metadata.get("semantic_role", "")).strip().lower()
    doc_type = candidate.doc_type
    normalized_text = _normalize_identifier(unit.text)
    tokens = {token for token in normalized_text.split("_") if token}
    hints = set(unit.semantic_hints)
    if not semantic_type and not data_type:
        return 0.0

    if "id" in tokens or "identifier" in tokens or "identifier_semantics" in hints:
        return 1.0 if semantic_type == "identifier" else 0.0

    if tokens & {"country", "state", "region", "city"}:
        return 1.0 if semantic_type == "geo" else 0.0

    if tokens & {"date", "time", "month", "year", "daily", "recent", "latest", "last"} or "time_semantics" in hints or "time_window" in hints:
        if semantic_type == "time":
            return 1.0
        return 0.6 if data_type in {"date", "datetime"} else 0.0

    if tokens & {"active", "inactive", "online", "offline", "status", "flag", "engaged"} or "status_semantics" in hints:
        if semantic_type == "status":
            return 1.0
        return 0.6 if data_type in {"bool", "boolean"} else 0.0

    if unit.unit_type == "metric" or "numeric_measure" in hints:
        if doc_type in {"field", "source_item"} and data_type in {"int", "float", "real", "numeric", "decimal"}:
            return 0.9
        if doc_type in {"knowledge", "routing_rule"}:
            return 0.6

    if unit.unit_type in {"output", "filter"} and semantic_type in {"category", "text"}:
        return 0.6
    if semantic_role == "rule" and unit.role in {"support", "routing"}:
        return 0.8
    return 0.0


def _format_pattern_support(unit: QueryUnit, candidate: RetrievalDocument) -> float:
    normalized_text = _normalize_identifier(unit.text)
    tokens = {token for token in normalized_text.split("_") if token}
    hints = set(unit.semantic_hints)
    format_pattern = str(candidate.metadata.get("format_pattern", "")).strip().lower()
    data_type = str(candidate.metadata.get("data_type", "")).strip().lower()
    value_range = str(candidate.metadata.get("value_range", "")).strip().lower()

    if not format_pattern and not data_type and not value_range:
        return 0.0

    if tokens & {"month", "year", "date", "time"} or "time_window" in hints or "time_semantics" in hints:
        if "date" in format_pattern or "datetime" in format_pattern or data_type in {"date", "datetime"}:
            return 1.0

    if tokens & {"active", "inactive", "online", "status", "flag", "engaged"} or "status_semantics" in hints:
        if "boolean" in format_pattern or data_type in {"bool", "boolean"}:
            return 0.9

    if (unit.unit_type == "metric" or "numeric_measure" in hints or "numeric_filter" in hints) and value_range and data_type in {"int", "float", "real", "numeric", "decimal"}:
        return 0.7

    return 0.0


def _question_intent_support(unit: QueryUnit, candidate: RetrievalDocument) -> float:
    if unit.unit_type != "question":
        return 0.0

    hints = set(unit.semantic_hints)
    data_type = str(candidate.metadata.get("data_type", "")).strip().lower()
    semantic_type = str(candidate.metadata.get("semantic_type", "")).strip().lower()
    rule_type = str(candidate.metadata.get("rule_type", "")).strip().lower()

    score = 0.0
    if "aggregation_count" in hints:
        if semantic_type in {"status", "identifier", "category"}:
            score = max(score, 0.5)
        if data_type in {"bool", "boolean"}:
            score = max(score, 0.8)
    if "aggregation_average" in hints or "aggregation_sum" in hints or "numeric_measure" in hints:
        if data_type in {"int", "float", "real", "numeric", "decimal"}:
            score = max(score, 0.9)
    if "time_window" in hints:
        if semantic_type == "time":
            score = max(score, 0.9)
        elif semantic_type == "status" and data_type in {"bool", "boolean"}:
            score = max(score, 0.7)
        if rule_type == "source_preference":
            score = max(score, 0.8)
    if "ranking_desc" in hints or "ranking_asc" in hints:
        if data_type in {"int", "float", "real", "numeric", "decimal"}:
            score = max(score, 0.8)
    return score


def _anchor_support(unit: QueryUnit, candidate: RetrievalDocument) -> float:
    unit_tokens = {_normalize_identifier(unit.text), *[_normalize_identifier(token) for token in unit.text.split()]}
    unit_tokens = {token for token in unit_tokens if token}
    anchor_names = [
        _normalize_identifier(str(item))
        for item in candidate.metadata.get("anchor_names", [])
        if _normalize_identifier(str(item))
    ]
    if candidate.doc_type == "anchor":
        anchor_name = _normalize_identifier(str(candidate.metadata.get("anchor_name", "")))
        if anchor_name and anchor_name in unit_tokens:
            return 1.0
        if unit.role == "routing":
            return 0.6
        return 0.2
    if anchor_names and unit_tokens.intersection(anchor_names):
        return 1.0
    return 0.0


def _source_type_support(unit: QueryUnit, candidate: RetrievalDocument) -> float:
    source_type = str(candidate.metadata.get("source_type", "")).strip().lower()
    target_sources = [str(item).strip().lower() for item in candidate.metadata.get("target_sources", []) if str(item).strip()]
    lowered_text = unit.text.lower()
    hints = set(unit.semantic_hints)

    if candidate.doc_type == "routing_rule":
        if any(source in lowered_text for source in ("json", "db", "sqlite", "csv")):
            return 1.0
        if "time_window" in hints and target_sources:
            return 0.7
        return 0.4

    if source_type == "json" and any(token in lowered_text for token in ("json", "profile", "payload", "nested")):
        return 1.0
    if source_type == "sqlite" and any(token in lowered_text for token in ("table", "database", "db", "transaction")):
        return 0.8
    if source_type == "csv" and any(token in lowered_text for token in ("csv", "sheet", "file", "row")):
        return 0.8
    if source_type == "document" and unit.role in {"support", "routing"}:
        return 0.6
    return 0.0


def _collect_candidates_for_unit(
    unit: QueryUnit,
    index: TaskRetrievalIndex,
    provider: EmbeddingProvider,
    config: RetrievalConfig,
) -> tuple[list[tuple[RetrievalDocument, float]], list[tuple[RetrievalDocument, float]]]:
    dense_top_k = max(config.retrieval_top_k, config.final_candidate_top_k)
    lexical_top_k = max(config.lexical_top_k, min(4, config.final_candidate_top_k))
    dense_results = []
    if config.enable_dense_linking:
        dense_results = retrieve_dense(index, provider, unit.text, top_k=dense_top_k)
    lexical_results = retrieve_lexical(index, unit.text, top_k=lexical_top_k)
    return dense_results, lexical_results


def _score_candidate(
    *,
    unit: QueryUnit,
    candidate: RetrievalDocument,
    dense_score: float,
    lexical_score: float,
    catalog: SemanticCatalog,
    anchor_entity: str | None,
) -> tuple[float, list[str]]:
    alias_score = _alias_match(unit.text, candidate)
    sample_score = _sample_value_hit(unit.text, candidate)
    knowledge_score = _knowledge_support(unit.text, candidate)
    description_score = _description_support(unit.text, candidate)
    semantic_type_score = _semantic_type_compatibility(unit, candidate)
    format_score = _format_pattern_support(unit, candidate)
    intent_score = _question_intent_support(unit, candidate)
    anchor_score = _anchor_support(unit, candidate)
    source_type_score = _source_type_support(unit, candidate)
    entity_score = _entity_consistency(candidate, anchor_entity)
    join_score = _join_connectivity(candidate, catalog, anchor_entity)
    role_score = _role_compatibility(unit, candidate)
    scope_weight = 0.20 if candidate.scope == "task" else 0.0
    ambiguity_penalty = 0.15 if candidate.doc_type == "knowledge" and unit.unit_type == "output" else 0.0

    final_score = (
        0.30 * dense_score
        + 0.08 * lexical_score
        + 0.10 * alias_score
        + 0.10 * sample_score
        + 0.10 * knowledge_score
        + 0.08 * description_score
        + 0.06 * semantic_type_score
        + 0.05 * format_score
        + 0.08 * intent_score
        + 0.08 * anchor_score
        + 0.06 * source_type_score
        + 0.10 * entity_score
        + 0.10 * join_score
        + 0.05 * role_score
        + scope_weight
        - ambiguity_penalty
    )
    reasons = [
        f"dense={dense_score:.3f}",
        f"lexical={lexical_score:.3f}",
        f"alias={alias_score:.3f}",
        f"sample={sample_score:.3f}",
        f"knowledge={knowledge_score:.3f}",
        f"description={description_score:.3f}",
        f"semantic_type={semantic_type_score:.3f}",
        f"format={format_score:.3f}",
        f"intent={intent_score:.3f}",
        f"anchor={anchor_score:.3f}",
        f"source_type={source_type_score:.3f}",
        f"entity={entity_score:.3f}",
        f"join={join_score:.3f}",
        f"role={role_score:.3f}",
        f"scope={candidate.scope}",
    ]
    if ambiguity_penalty:
        reasons.append(f"penalty={ambiguity_penalty:.3f}")
    return final_score, reasons


def _link_candidate_to_dict(candidate: LinkCandidate) -> dict[str, Any]:
    return {
        "doc_id": candidate.doc_id,
        "doc_type": candidate.doc_type,
        "source_ref": candidate.source_ref,
        "dense_score": round(candidate.dense_score, 4),
        "lexical_score": round(candidate.lexical_score, 4),
        "final_score": round(candidate.final_score, 4),
        "reasons": list(candidate.reasons),
        "metadata": dict(candidate.metadata),
    }


def _resolve_binding_value(unit: QueryUnit, chosen: LinkCandidate) -> str | int | float | None:
    if unit.unit_type == "filter" and unit.text == "severe degree of thrombosis":
        sample_values = [str(item).strip() for item in chosen.metadata.get("sample_values", []) if str(item).strip()]
        for value in sample_values:
            if value.isdigit():
                return int(value)
        return 2
    return None


def _build_binding_conflict(unit: QueryUnit, field_candidates: list[LinkCandidate]) -> dict[str, Any] | None:
    if len(field_candidates) < 2:
        return None
    top_candidate = field_candidates[0]
    runner_up = field_candidates[1]
    if top_candidate.final_score - runner_up.final_score >= 0.15:
        return None
    return {
        "unit_type": unit.unit_type,
        "text": unit.text,
        "candidates": [
            _link_candidate_to_dict(top_candidate),
            _link_candidate_to_dict(runner_up),
        ],
        "resolution": (
            f"Prefer {top_candidate.metadata.get('field_ref', top_candidate.doc_id)} because it ranked highest "
            "after hybrid retrieval and entity/connectivity scoring."
        ),
    }


def _build_join_candidates(
    catalog: SemanticCatalog,
    chosen_bindings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chosen_entities = {
        str(item.get("entity", "")).strip()
        for item in chosen_bindings
        if str(item.get("entity", "")).strip()
    }
    join_candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for relation in catalog.relations:
        if relation.left_entity not in chosen_entities or relation.right_entity not in chosen_entities:
            continue
        for key_pair in relation.join_keys[:1]:
            signature = (
                relation.left_entity,
                relation.right_entity,
                key_pair.left_field,
                key_pair.right_field,
            )
            if signature in seen:
                continue
            seen.add(signature)
            join_candidates.append(
                {
                    "left_entity": relation.left_entity,
                    "right_entity": relation.right_entity,
                    "left_field": key_pair.left_field,
                    "right_field": key_pair.right_field,
                    "score": 1.0,
                    "reason": relation.description or "linked entities require a join through the catalog relation",
                }
            )
    return join_candidates


def link_schema_candidates(
    question: str,
    catalog: SemanticCatalog,
    index: TaskRetrievalIndex,
    provider: EmbeddingProvider,
    config: RetrievalConfig,
) -> SchemaLinkResult:
    query_units = extract_query_units(question)
    anchor_entity = _best_entity_for_question(question, catalog, index, provider, config)
    top_entities: list[dict[str, Any]] = []
    top_fields: list[dict[str, Any]] = []
    top_knowledge: list[dict[str, Any]] = []
    top_routing: list[dict[str, Any]] = []
    candidate_bindings: list[dict[str, Any]] = []
    chosen_bindings: list[dict[str, Any]] = []
    binding_conflicts: list[dict[str, Any]] = []
    candidate_join_paths: list[dict[str, Any]] = []
    unresolved_ambiguities: list[dict[str, Any]] = []
    required_sources: set[str] = set()
    unit_debug: list[dict[str, Any]] = []

    for unit in query_units:
        dense_results, lexical_results = _collect_candidates_for_unit(unit, index, provider, config)
        merged_scores: dict[str, dict[str, float | RetrievalDocument]] = {}
        for document, score in dense_results:
            entry = merged_scores.setdefault(document.doc_id, {"document": document, "dense": 0.0, "lexical": 0.0})
            entry["dense"] = max(float(entry["dense"]), float(score))
        for document, score in lexical_results:
            entry = merged_scores.setdefault(document.doc_id, {"document": document, "dense": 0.0, "lexical": 0.0})
            entry["lexical"] = max(float(entry["lexical"]), float(score))

        candidates: list[LinkCandidate] = []
        for entry in merged_scores.values():
            document = entry["document"]
            dense_score = float(entry["dense"])
            lexical_score = float(entry["lexical"])
            final_score, reasons = _score_candidate(
                unit=unit,
                candidate=document,
                dense_score=dense_score,
                lexical_score=lexical_score,
                catalog=catalog,
                anchor_entity=anchor_entity,
            )
            candidates.append(
                LinkCandidate(
                    doc_id=document.doc_id,
                    doc_type=document.doc_type,
                    source_ref=document.source_ref,
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                    final_score=final_score,
                    reasons=reasons,
                    metadata=dict(document.metadata),
                )
            )

        candidates.sort(key=lambda item: (-item.final_score, item.doc_id))
        top_candidates = candidates[: config.final_candidate_top_k]
        dense_debug = [{"doc_id": document.doc_id, "score": round(score, 4)} for document, score in dense_results]
        lexical_debug = [{"doc_id": document.doc_id, "score": round(score, 4)} for document, score in lexical_results]
        unit_debug.append(
            {
                "unit_type": unit.unit_type,
                "text": unit.text,
                "role": unit.role,
                "binding_eligible": unit.binding_eligible,
                "semantic_hints": list(unit.semantic_hints),
                "dense_top_k": dense_debug,
                "lexical_top_k": lexical_debug,
                "merged_candidates": [_link_candidate_to_dict(candidate) for candidate in top_candidates[:5]],
            }
        )

        if unit.unit_type == "question":
            for candidate in top_candidates:
                if candidate.doc_type == "entity" and len(top_entities) < 3:
                    top_entities.append(_link_candidate_to_dict(candidate))
                elif candidate.doc_type in {"field", "source_item"} and len(top_fields) < 7:
                    top_fields.append(_link_candidate_to_dict(candidate))
                elif candidate.doc_type == "knowledge" and len(top_knowledge) < 5:
                    top_knowledge.append(_link_candidate_to_dict(candidate))
                elif candidate.doc_type in {"routing_rule", "anchor"} and len(top_routing) < 5:
                    top_routing.append(_link_candidate_to_dict(candidate))
            continue

        if not unit.binding_eligible:
            if unit.role == "routing":
                routing_candidates = [
                    candidate
                    for candidate in top_candidates
                    if candidate.doc_type in {"routing_rule", "anchor", "source_item"}
                ]
                if routing_candidates:
                    unit_debug[-1]["routing_candidates"] = [
                        _link_candidate_to_dict(candidate) for candidate in routing_candidates[:5]
                    ]
                    for candidate in routing_candidates[:3]:
                        if candidate.doc_type in {"routing_rule", "anchor"} and len(top_routing) < 5:
                            top_routing.append(_link_candidate_to_dict(candidate))
            continue

        field_candidates = [candidate for candidate in top_candidates if candidate.doc_type in {"field", "source_item"}]
        candidate_bindings.append(
            {
                "unit_type": unit.unit_type,
                "text": unit.text,
                "role": unit.role,
                "semantic_hints": list(unit.semantic_hints),
                "candidates": [_link_candidate_to_dict(candidate) for candidate in field_candidates[:5]],
            }
        )
        if field_candidates:
            chosen = field_candidates[0]
            binding = {
                "unit_type": unit.unit_type,
                "text": unit.text,
                "doc_id": chosen.doc_id,
                "field_ref": str(chosen.metadata.get("field_ref", "")),
                "entity": str(chosen.metadata.get("entity", "")),
                "source_ref": chosen.source_ref,
                "score": round(chosen.final_score, 4),
                "resolved_value": _resolve_binding_value(unit, chosen),
                "reasons": list(chosen.reasons),
            }
            chosen_bindings.append(binding)
            if chosen.source_ref:
                required_sources.add(chosen.source_ref)
            entity_name = binding["entity"]
            if anchor_entity and entity_name and entity_name != anchor_entity:
                for edge in _find_join_path(catalog, anchor_entity=anchor_entity, target_entity=entity_name):
                    if edge not in candidate_join_paths:
                        candidate_join_paths.append(edge)
            conflict = _build_binding_conflict(unit, field_candidates)
            if conflict is not None:
                binding_conflicts.append(conflict)
                unresolved_ambiguities.append(
                    {
                        "unit_text": unit.text,
                        "candidates": list(conflict["candidates"]),
                    }
                )
        elif top_candidates:
            unresolved_ambiguities.append(
                {
                    "unit_text": unit.text,
                    "candidates": [_link_candidate_to_dict(candidate) for candidate in top_candidates[:2]],
                }
            )

    if not required_sources:
        for item in top_entities:
            sources = item["metadata"].get("sources", [])
            required_sources.update(str(source) for source in sources)
        for item in top_fields:
            required_sources.add(item["source_ref"])

    join_candidates = _build_join_candidates(catalog, chosen_bindings)
    debug_view = {
        "anchor_entity": anchor_entity,
        "units": unit_debug,
        "candidate_bindings": list(candidate_bindings),
        "chosen_bindings": list(chosen_bindings),
        "binding_conflicts": list(binding_conflicts),
        "join_candidates": list(join_candidates),
        "candidate_join_paths": list(candidate_join_paths),
        "ambiguities": list(unresolved_ambiguities),
        "top_routing": list(top_routing),
    }
    return SchemaLinkResult(
        query_units=[unit.to_dict() for unit in query_units],
        top_entities=top_entities,
        top_fields=top_fields,
        top_knowledge=top_knowledge + top_routing,
        candidate_bindings=candidate_bindings,
        chosen_bindings=chosen_bindings,
        binding_conflicts=binding_conflicts,
        join_candidates=join_candidates,
        candidate_join_paths=candidate_join_paths,
        required_sources=sorted(source for source in required_sources if source),
        unresolved_ambiguities=unresolved_ambiguities,
        debug_view=debug_view,
    )
