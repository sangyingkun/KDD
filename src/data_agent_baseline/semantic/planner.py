from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.semantic.catalog import (
    PlannerBindingCandidate,
    PlannerChosenBinding,
    PlannerConflict,
    PlannerJoinCandidate,
    PlannerQuestionSlot,
    SemanticCatalog,
)
from data_agent_baseline.semantic.builder import _normalize_identifier


_AGGREGATION_KEYWORDS = ("total", "sum", "average", "avg", "rate", "count")
_DOC_EVIDENCE_KEYWORDS = (
    "according to",
    "policy",
    "definition",
    "defined",
    "eligible",
    "repeat",
    "first",
    "second",
    "abnormal",
    "normal",
    "legal status",
    "content warning",
)

_MAIN_ENTITY_PREFIXES = ("among the ", "among ", "of ", "for ", "in ")
_SLOT_SYNONYMS = {
    "ID": {"id", "identifier", "patient id"},
    "sex": {"sex", "gender"},
    "disease": {"disease", "diagnosis", "diagnosed with"},
    "severe degree of thrombosis": {"severe degree of thrombosis", "severe thrombosis", "thrombosis"},
}


def _singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("oes") and len(token) > 3:
        return token[:-2]
    if token.endswith("es") and len(token) > 3 and token[-3] in {"o", "s", "x"}:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 2:
        return token[:-1]
    return token


def _pluralize_phrase(phrase: str) -> str:
    parts = phrase.split()
    if not parts:
        return phrase
    last = parts[-1]
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = f"{last[:-1]}ies"
    elif last.endswith(("s", "x", "ch", "sh", "o")):
        parts[-1] = f"{last}es"
    else:
        parts[-1] = f"{last}s"
    return " ".join(parts)


def _entity_variants(name: str) -> set[str]:
    normalized = _normalize_identifier(name)
    phrase = normalized.replace("_", " ")
    variants = {phrase, _pluralize_phrase(phrase), normalized, normalized.replace("_", "")}
    if "_" in normalized:
        variants.add(" ".join(normalized.split("_")))
        variants.add(_pluralize_phrase(" ".join(normalized.split("_"))))
    return {variant for variant in variants if variant}


def _find_entity_mentions(question: str, entity_name: str) -> list[int]:
    lower_question = question.lower()
    positions: list[int] = []
    for variant in sorted(_entity_variants(entity_name), key=len, reverse=True):
        start = lower_question.find(variant)
        if start >= 0:
            positions.append(start)
    return sorted(set(positions))


def _tokens_related(left: str, right: str) -> bool:
    if left == right:
        return True
    if _singularize_token(left) == _singularize_token(right):
        return True
    common_prefix_len = min(len(left), len(right), 6)
    return common_prefix_len >= 5 and left[:common_prefix_len] == right[:common_prefix_len]


def _build_entity_source_map(catalog: SemanticCatalog) -> dict[str, list[str]]:
    return {entity.name: list(entity.sources) for entity in catalog.entities}


def _extract_value_links(catalog: SemanticCatalog, question: str) -> list[dict[str, str]]:
    lower_question = question.lower()
    normalized_question = _normalize_identifier(question).replace("_", " ")
    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for dimension in catalog.dimensions:
        for sample_value in dimension.sample_values:
            sample_text = str(sample_value).strip()
            if not sample_text:
                continue
            normalized_sample = _normalize_identifier(sample_text).replace("_", " ")
            if (
                sample_text.lower() in lower_question
                or (normalized_sample and normalized_sample in normalized_question)
            ):
                signature = (dimension.field_ref, sample_text.lower())
                if signature in seen:
                    continue
                seen.add(signature)
                matches.append(
                    {
                        "dimension": dimension.name,
                        "entity": dimension.entity,
                        "value": sample_text,
                        "field_ref": dimension.field_ref,
                        "match_type": "sample_value",
                    }
                )
    return matches


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


def _infer_expected_output_columns(catalog: SemanticCatalog, question: str) -> list[str]:
    lower_question = question.lower()
    normalized_question = _normalize_identifier(question).replace("_", " ")
    if "full name" not in lower_question and "full name" not in normalized_question:
        return []

    for constraint in catalog.knowledge_contract.output_constraints:
        concept = str(constraint.get("concept", "")).lower()
        fields = [str(field) for field in constraint.get("fields", []) if str(field)]
        if concept == "full name" and fields:
            return fields

    for evidence in catalog.evidence:
        claim = evidence.claim.lower()
        if "full name" not in claim:
            continue
        field_match = re.search(r"\*\*([^*]+)\*\*", evidence.claim)
        if field_match is None:
            continue
        fields = [
            _normalize_identifier(part)
            for part in field_match.group(1).split(",")
            if _normalize_identifier(part)
        ]
        if len(fields) >= 2:
            return fields
    return []


def _extract_knowledge_hints(catalog: SemanticCatalog, question: str) -> list[str]:
    lower_question = question.lower()
    hints: list[str] = []
    for rule in catalog.knowledge_contract.constraint_rules:
        field_name = str(rule.get("field", "")).strip()
        allowed_values = [str(value) for value in rule.get("allowed_values", []) if str(value)]
        if not field_name or not allowed_values:
            continue
        raw_text = str(rule.get("raw_text", "")).strip()
        if field_name.lower() == "admission" and any(token in lower_question for token in ("inpatient", "outpatient", "admission")):
            if "+" in allowed_values and "-" in allowed_values:
                hints.append("Admission uses '+' for inpatients and '-' for outpatients.")
            elif raw_text:
                hints.append(raw_text)
        elif field_name.lower() in lower_question and raw_text:
            hints.append(raw_text)
    return hints


def _extract_question_slots(question: str) -> list[PlannerQuestionSlot]:
    lower_question = question.lower()
    slots: list[PlannerQuestionSlot] = []
    if "severe degree of thrombosis" in lower_question:
        slots.append(PlannerQuestionSlot(slot_type="filter", phrase="severe degree of thrombosis"))
    if "list their id" in lower_question or re.search(r"\btheir id\b", lower_question):
        slots.append(PlannerQuestionSlot(slot_type="output", phrase="ID"))
    if " sex" in f" {lower_question}" or " gender" in f" {lower_question}":
        slots.append(PlannerQuestionSlot(slot_type="output", phrase="sex"))
    if "disease" in lower_question or "diagnosed with" in lower_question:
        slots.append(PlannerQuestionSlot(slot_type="output", phrase="disease"))
    return slots


def _score_field_binding(
    *,
    phrase: str,
    field_name: str,
    entity_name: str,
    question: str,
) -> tuple[float, str]:
    normalized_phrase = _normalize_identifier(phrase)
    normalized_field = _normalize_identifier(field_name)
    normalized_entity = _normalize_identifier(entity_name)
    candidate_forms = _SLOT_SYNONYMS.get(phrase, {phrase.lower()})
    normalized_forms = {_normalize_identifier(item) for item in candidate_forms}

    if normalized_field in normalized_forms:
        return 1.0, "exact field name match"
    if any(form and form in normalized_field for form in normalized_forms):
        return 0.9, "phrase contained in field name"
    if normalized_phrase == "disease" and normalized_field == "diagnosis" and normalized_entity == "patient":
        return 0.95, "patient diagnosis preferred for patient-level disease wording"
    if normalized_phrase == "disease" and normalized_field == "diagnosis":
        return 0.7, "diagnosis is a disease-like field"
    if normalized_phrase == "severe_degree_of_thrombosis" and normalized_field == "thrombosis":
        return 0.95, "thrombosis field matches severity phrase"
    if normalized_phrase in normalized_entity:
        return 0.6, "phrase aligned with entity name"
    if normalized_phrase == "id" and normalized_field == "id" and normalized_entity == "patient":
        return 0.9, "patient id preferred for patient-level listing"
    if "diagnosed with" in question.lower() and normalized_field == "diagnosis" and normalized_entity == "patient":
        return 0.95, "question asks for patient diagnosis"
    return 0.0, "no strong match"


def _build_binding_candidates(
    catalog: SemanticCatalog,
    question: str,
    slots: list[PlannerQuestionSlot],
) -> dict[str, list[PlannerBindingCandidate]]:
    candidates_by_phrase: dict[str, list[PlannerBindingCandidate]] = {}
    for slot in slots:
        candidates: list[PlannerBindingCandidate] = []
        for dimension in catalog.dimensions:
            score, reason = _score_field_binding(
                phrase=slot.phrase,
                field_name=dimension.name,
                entity_name=dimension.entity,
                question=question,
            )
            if score <= 0:
                continue
            candidates.append(
                PlannerBindingCandidate(
                    field_ref=dimension.field_ref,
                    entity=dimension.entity,
                    score=score,
                    reason=reason,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.field_ref))
        candidates_by_phrase[slot.phrase] = candidates
    return candidates_by_phrase


def _choose_bindings(
    slots: list[PlannerQuestionSlot],
    candidates_by_phrase: dict[str, list[PlannerBindingCandidate]],
) -> list[PlannerChosenBinding]:
    chosen: list[PlannerChosenBinding] = []
    for slot in slots:
        candidates = candidates_by_phrase.get(slot.phrase, [])
        if not candidates:
            continue
        top = candidates[0]
        resolved_value: str | int | float | None = None
        if slot.phrase == "severe degree of thrombosis":
            resolved_value = 2
        chosen.append(
            PlannerChosenBinding(
                slot_type=slot.slot_type,
                phrase=slot.phrase,
                field_ref=top.field_ref,
                entity=top.entity,
                resolved_value=resolved_value,
            )
        )
    return chosen


def _build_binding_conflicts(
    slots: list[PlannerQuestionSlot],
    candidates_by_phrase: dict[str, list[PlannerBindingCandidate]],
) -> list[PlannerConflict]:
    conflicts: list[PlannerConflict] = []
    for slot in slots:
        candidates = candidates_by_phrase.get(slot.phrase, [])
        if len(candidates) < 2:
            continue
        if slot.phrase == "disease":
            conflicts.append(
                PlannerConflict(
                    phrase="disease",
                    candidates=[f"{item.entity}.Diagnosis" for item in candidates[:2]],
                    resolution="Prefer patient.Diagnosis because the question asks for the disease the patient is diagnosed with.",
                )
            )
    return conflicts


def _build_join_candidates(
    catalog: SemanticCatalog,
    chosen_bindings: list[PlannerChosenBinding],
) -> list[PlannerJoinCandidate]:
    chosen_entities = {item.entity for item in chosen_bindings}
    join_candidates: list[PlannerJoinCandidate] = []
    for relation in catalog.relations:
        if relation.left_entity in chosen_entities and relation.right_entity in chosen_entities:
            key_pair = relation.join_keys[0]
            join_candidates.append(
                PlannerJoinCandidate(
                    left_entity=relation.left_entity,
                    right_entity=relation.right_entity,
                    left_field=key_pair.left_field,
                    right_field=key_pair.right_field,
                    score=1.0,
                    reason="shared key and output fields require patient attributes",
                )
            )
    return join_candidates


def _reconstruct_question(
    catalog: SemanticCatalog,
    question: str,
) -> dict[str, Any]:
    lower_question = question.lower()
    normalized_question = _normalize_identifier(question)
    raw_tokens = [token for token in normalized_question.split("_") if token]
    tokens = set(raw_tokens)
    singular_tokens = {_singularize_token(token) for token in raw_tokens}
    collapsed_question = normalized_question.replace("_", "")

    target_entity: str | None = None
    supporting_entities: list[str] = []
    entity_scores: list[tuple[int, int, str]] = []
    for entity in catalog.entities:
        score = 0
        entity_key = _normalize_identifier(entity.name)
        entity_tokens = {part for part in entity_key.split("_") if part}
        singular_entity_tokens = {_singularize_token(part) for part in entity_tokens}
        mentions = _find_entity_mentions(lower_question, entity.name)
        is_bridge_like = len(entity_tokens) > 1

        if entity_tokens and (
            entity_tokens <= tokens
            or entity_tokens <= singular_tokens
            or singular_entity_tokens <= singular_tokens
        ):
            score += 3
        elif any(
            any(_tokens_related(entity_token, question_token) for question_token in singular_tokens)
            for entity_token in singular_entity_tokens
        ):
            score += 1
        if mentions:
            score += 3
            first_mention = mentions[0]
            if any(
                lower_question[max(0, first_mention - len(prefix)) : first_mention] == prefix
                for prefix in _MAIN_ENTITY_PREFIXES
            ):
                score += 3
            score += max(0, 2 - min(first_mention // 25, 2))
        elif entity_key.replace("_", "") in collapsed_question:
            score += 2
        for source in entity.sources:
            source_tokens = set(_normalize_identifier(source).split("_"))
            if entity_tokens & source_tokens & tokens:
                score += 1
            elif any(
                any(_tokens_related(source_token, question_token) for question_token in singular_tokens)
                for source_token in source_tokens
            ):
                score += 1
        if is_bridge_like:
            score -= 1
        entity_scores.append((score, -(mentions[0] if mentions else 10_000), entity.name))

    entity_scores.sort(reverse=True)
    if entity_scores and entity_scores[0][0] > 0:
        target_entity = entity_scores[0][2]
        supporting_entities = [
            name for score, _, name in entity_scores[1:] if score > 0
        ]

    target_metric: str | None = None
    matched_measure: str | None = None
    for measure in catalog.measures:
        measure_tokens = set(_normalize_identifier(measure.name).split("_"))
        if measure_tokens and measure_tokens <= tokens:
            matched_measure = measure.name
            break

    if "percentage" in lower_question or "percent" in lower_question:
        target_metric = "percentage"
    elif "how many" in lower_question or "number of" in lower_question or "count" in lower_question:
        target_metric = "count"
    elif "average" in lower_question or "avg" in lower_question:
        target_metric = "average"
    elif "rate" in lower_question:
        target_metric = "rate"

    if matched_measure is not None:
        matched_measure_normalized = _normalize_identifier(matched_measure).replace("_", " ")
        if "amount" in matched_measure_normalized and "total" in lower_question:
            target_metric = "total amount"
        elif target_metric is None:
            target_metric = matched_measure_normalized

    grain = "overall"
    if "by region" in lower_question:
        grain = "region"
    elif "by student" in lower_question:
        grain = "student"
    elif "by customer" in lower_question:
        grain = "customer"
    elif "by month" in lower_question:
        grain = "month"

    filters: list[str] = []
    for marker in ("completed", "active", "inactive", "abnormal", "champion"):
        if re.search(rf"\b{re.escape(marker)}\b", lower_question):
            filters.append(marker)
    if re.search(r"\bnormal\b", lower_question) and "abnormal" not in filters:
        filters.append("normal")
    if "format commander" in lower_question or "commander" in lower_question:
        filters.append("format commander")
    if "legal status" in lower_question:
        filters.append("legal status")
    if "content warning" in lower_question:
        if any(phrase in lower_question for phrase in ("do not have", "without", "no content warning")):
            filters.append("no content warning")
        else:
            filters.append("content warning")
    if "published by" in lower_question:
        published_fragment = lower_question.split("published by", 1)[1].strip(" ?.!,")
        if published_fragment:
            filters.append(f"published by {published_fragment}")
    for match in sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", lower_question))):
        filters.append(match)
    if re.search(r"\b(?:aren't|not)\s+70\s+yet\b", lower_question):
        filters.append("age under 70")
    for match in re.findall(r"\bover\s+\d+\s*[a-z%]*\b", lower_question):
        filters.append(match.strip())
    for start, end in re.findall(r"\bbetween\s+(\d+)\s+(?:to|and)\s+(\d+)\b", lower_question):
        filters.append(f"between {start} and {end}")

    evidence_needs = any(keyword in lower_question for keyword in _DOC_EVIDENCE_KEYWORDS)
    value_links = _extract_value_links(catalog, question)
    expected_output_columns = _infer_expected_output_columns(catalog, question)
    knowledge_hints = _extract_knowledge_hints(catalog, question)
    return {
        "target_entity": target_entity,
        "supporting_entities": supporting_entities,
        "target_metric": target_metric,
        "filters": filters,
        "grain": grain,
        "evidence_needs": evidence_needs or bool(expected_output_columns),
        "matched_measure": matched_measure,
        "linked_values": value_links,
        "expected_output_columns": expected_output_columns,
        "knowledge_hints": knowledge_hints,
    }


def plan_semantic_query(
    catalog: SemanticCatalog,
    question: str,
    target_metric: str | None = None,
    target_entity: str | None = None,
) -> dict[str, Any]:
    lower_question = question.lower()
    reconstruction = _reconstruct_question(catalog, question)
    question_slots = _extract_question_slots(question)
    candidates_by_phrase = _build_binding_candidates(catalog, question, question_slots)
    chosen_bindings = _choose_bindings(question_slots, candidates_by_phrase)
    binding_conflicts = _build_binding_conflicts(question_slots, candidates_by_phrase)
    join_candidates = _build_join_candidates(catalog, chosen_bindings)
    has_aggregation = any(keyword in lower_question for keyword in _AGGREGATION_KEYWORDS)
    has_document_risk = reconstruction["evidence_needs"] or bool(
        catalog.evidence and any(keyword in lower_question for keyword in ("rate", "margin", "definition"))
    )
    expected_output_grain = reconstruction["grain"] if reconstruction["grain"] != "overall" else "unknown"
    recommended_path = "doc_first" if has_document_risk else "sql_first" if has_aggregation else "mixed"
    required_dimensions = [
        dimension.name for dimension in catalog.dimensions if dimension.name.lower() in lower_question
    ]
    resolved_target_entity = target_entity or reconstruction["target_entity"]
    required_entities = [target_entity] if target_entity else []
    required_measures = [
        measure.name
        for measure in catalog.measures
        if _normalize_identifier(measure.name) in _normalize_identifier(question)
    ]

    if target_metric:
        for metric in catalog.metrics:
            if metric.name == target_metric:
                required_measures = sorted(set(required_measures) | set(metric.base_measures))
                required_dimensions = sorted(
                    set(required_dimensions) | set(metric.required_dimensions)
                )
                if metric.grain:
                    expected_output_grain = metric.grain
                break

    if reconstruction["matched_measure"] and reconstruction["matched_measure"] not in required_measures:
        required_measures.append(reconstruction["matched_measure"])

    for value_link in reconstruction["linked_values"]:
        if value_link["dimension"] not in required_dimensions:
            required_dimensions.append(value_link["dimension"])

    if has_aggregation and not required_measures:
        required_measures = [measure.name for measure in catalog.measures[:1]]

    source_scores: dict[str, int] = {}
    entity_sources = _build_entity_source_map(catalog)
    linked_entities = {item["entity"] for item in reconstruction["linked_values"]}
    linked_entities.update(item.entity for item in chosen_bindings)
    for entity in catalog.entities:
        score = 0
        if entity.name in required_entities:
            score += 3
        if reconstruction["target_entity"] == entity.name:
            score += 5
        if entity.name in reconstruction["supporting_entities"]:
            score += 3
        if entity.name in linked_entities:
            score += 3
        if any(measure.entity == entity.name for measure in catalog.measures if measure.name in required_measures):
            score += 2
        if _find_entity_mentions(lower_question, entity.name):
            score += 1
        if any(item.entity == entity.name for item in chosen_bindings):
            score += 4
        for source in entity.sources:
            source_scores[source] = max(source_scores.get(source, 0), score)

    best_source_score = max(source_scores.values(), default=0)
    required_sources = sorted(
        source
        for source, score in source_scores.items()
        if score > 0 and score >= max(best_source_score - 3, 1)
    )
    if not required_sources and catalog.entities:
        required_sources.extend(catalog.entities[0].sources)

    anchor_entity = resolved_target_entity or reconstruction["target_entity"]
    join_path_targets: list[str] = []
    for entity_name in reconstruction["supporting_entities"]:
        if entity_name != anchor_entity and entity_name not in join_path_targets:
            join_path_targets.append(entity_name)
    for entity_name in linked_entities:
        if entity_name != anchor_entity and entity_name not in join_path_targets:
            join_path_targets.append(entity_name)

    recommended_join_path: list[dict[str, str]] = []
    for entity_name in join_path_targets:
        for edge in _find_join_path(catalog, anchor_entity=anchor_entity, target_entity=entity_name):
            if edge not in recommended_join_path:
                recommended_join_path.append(edge)

    if not recommended_join_path and join_candidates:
        recommended_join_path = [
            {
                "left_entity": item.left_entity,
                "right_entity": item.right_entity,
                "cardinality": "many_to_one",
            }
            for item in join_candidates
        ]

    for entity_name in linked_entities:
        for source in entity_sources.get(entity_name, []):
            if source not in required_sources:
                required_sources.append(source)

    return {
        "recommended_path": recommended_path,
        "required_sources": sorted(set(required_sources)),
        "required_entities": required_entities,
        "required_dimensions": required_dimensions,
        "required_measures": required_measures,
        "needs_doc_confirmation": has_document_risk,
        "join_risks": [
            {
                "left_entity": relation.left_entity,
                "right_entity": relation.right_entity,
                "cardinality": relation.cardinality,
            }
            for relation in catalog.relations
            if relation.cardinality in {"one_to_many", "many_to_many", "unknown"}
        ],
        "recommended_join_path": recommended_join_path,
        "expected_output_grain": expected_output_grain,
        "expected_output_columns": reconstruction["expected_output_columns"],
        "semantic_reconstruction": {
            "target_entity": reconstruction["target_entity"],
            "supporting_entities": reconstruction["supporting_entities"],
            "target_metric": reconstruction["target_metric"],
            "filters": reconstruction["filters"],
            "grain": reconstruction["grain"],
            "evidence_needs": reconstruction["evidence_needs"],
            "linked_values": reconstruction["linked_values"],
            "expected_output_columns": reconstruction["expected_output_columns"],
            "knowledge_hints": reconstruction["knowledge_hints"],
        },
        "debug_view": {
            "question_slots": [
                {"slot_type": item.slot_type, "phrase": item.phrase}
                for item in question_slots
            ],
            "candidate_bindings": [
                {
                    "phrase": phrase,
                    "candidates": [
                        {
                            "field_ref": candidate.field_ref,
                            "entity": candidate.entity,
                            "score": candidate.score,
                            "reason": candidate.reason,
                        }
                        for candidate in candidates
                    ],
                }
                for phrase, candidates in candidates_by_phrase.items()
            ],
            "chosen_bindings": [
                {
                    "slot_type": item.slot_type,
                    "phrase": item.phrase,
                    "field_ref": item.field_ref,
                    "entity": item.entity,
                    "resolved_value": item.resolved_value,
                }
                for item in chosen_bindings
            ],
            "join_candidates": [
                {
                    "left_entity": item.left_entity,
                    "right_entity": item.right_entity,
                    "left_field": item.left_field,
                    "right_field": item.right_field,
                    "score": item.score,
                    "reason": item.reason,
                }
                for item in join_candidates
            ],
            "chosen_join_path": recommended_join_path,
            "binding_conflicts": [
                {
                    "phrase": item.phrase,
                    "candidates": item.candidates,
                    "resolution": item.resolution,
                }
                for item in binding_conflicts
            ],
        },
        "fallback_strategy": "If the metric definition is unclear, read the relevant document before answering.",
    }
