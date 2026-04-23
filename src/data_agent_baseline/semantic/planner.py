from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.semantic.catalog import SemanticCatalog
from data_agent_baseline.semantic.builder import _normalize_identifier
from data_agent_baseline.semantic.linker import SchemaLinkResult


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


def _source_item_lookup(catalog: SemanticCatalog) -> dict[str, dict[str, Any]]:
    return {
        item.item_id: {
            "item_id": item.item_id,
            "item_type": item.item_type,
            "entity": item.entity,
            "source_type": item.source_type,
            "source_file": item.source_file,
            "source_path": item.source_path,
            "field_ref": item.field_ref,
            "display_name": item.display_name,
            "semantic_role": item.semantic_role,
            "anchor_names": list(item.anchor_names),
            "description": item.description,
            "metadata": dict(item.metadata),
        }
        for item in catalog.source_items
    }


def _anchor_lookup(catalog: SemanticCatalog) -> dict[str, dict[str, Any]]:
    return {
        anchor.anchor_name: {
            "anchor_name": anchor.anchor_name,
            "members": list(anchor.members),
            "source_files": list(anchor.source_files),
            "description": anchor.description,
            "metadata": dict(anchor.metadata),
        }
        for anchor in catalog.cross_source_anchors
    }


def _collect_routing_candidates(link_result: SchemaLinkResult | None) -> list[dict[str, Any]]:
    if link_result is None:
        return []
    candidates = []
    for item in link_result.top_knowledge:
        if str(item.get("doc_type", "")) in {"routing_rule", "anchor"}:
            candidates.append(dict(item))
    linker_debug = link_result.debug_view if isinstance(link_result.debug_view, dict) else {}
    for item in linker_debug.get("top_routing", []):
        if isinstance(item, dict):
            candidates.append(dict(item))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(item)
    return deduped


def _build_routing_plan(
    *,
    catalog: SemanticCatalog,
    chosen_bindings: list[dict[str, Any]],
    required_sources: list[str],
    routing_candidates: list[dict[str, Any]],
    recommended_join_path: list[dict[str, str]],
    needs_doc_confirmation: bool,
) -> list[dict[str, Any]]:
    source_items = _source_item_lookup(catalog)
    anchors = _anchor_lookup(catalog)
    routing_plan: list[dict[str, Any]] = []

    selected_source_items: list[dict[str, Any]] = []
    selected_anchor_names: set[str] = set()
    for binding in chosen_bindings:
        doc_id = str(binding.get("doc_id", "")).strip()
        if "source_item::" in doc_id:
            item_id = doc_id.split("::", 2)[-1]
            source_item = source_items.get(item_id)
            if source_item is not None:
                selected_source_items.append(source_item)
                selected_anchor_names.update(source_item.get("anchor_names", []))

    preferred_source_hints: list[str] = []
    rule_constraints: list[dict[str, Any]] = []
    for candidate in routing_candidates:
        doc_type = str(candidate.get("doc_type", "")).strip()
        metadata = dict(candidate.get("metadata", {}))
        if doc_type == "routing_rule":
            preferred_source_hints.extend(str(item) for item in metadata.get("target_sources", []) if str(item).strip())
            rule_constraints.append(
                {
                    "doc_id": candidate.get("doc_id"),
                    "rule_type": metadata.get("rule_type"),
                    "description": next(
                        (reason for reason in candidate.get("reasons", []) if str(reason).startswith("knowledge=")),
                        str(candidate.get("source_ref", "")),
                    ),
                    "target_sources": list(metadata.get("target_sources", [])),
                    "anchor_names": list(metadata.get("anchor_names", [])),
                }
            )
            selected_anchor_names.update(
                str(item) for item in metadata.get("anchor_names", []) if str(item).strip()
            )
        elif doc_type == "anchor":
            anchor_name = str(metadata.get("anchor_name", "")).strip()
            if anchor_name:
                selected_anchor_names.add(anchor_name)

    def _source_matches_hint(source_ref: str, hint: str) -> bool:
        lowered_source = source_ref.lower()
        lowered_hint = hint.lower()
        if lowered_hint in {"db", "sqlite"}:
            return lowered_source.endswith((".db", ".sqlite"))
        if lowered_hint == "json":
            return lowered_source.endswith(".json")
        if lowered_hint == "csv":
            return lowered_source.endswith(".csv")
        if lowered_hint in {"document", "doc"}:
            return lowered_source.endswith((".md", ".txt"))
        return lowered_hint == lowered_source

    ordered_sources: list[str] = []
    for hint in preferred_source_hints:
        for source in required_sources:
            if _source_matches_hint(source, hint) and source not in ordered_sources:
                ordered_sources.append(source)
    for source in required_sources:
        normalized_source = str(source).strip()
        if normalized_source and normalized_source not in ordered_sources:
            ordered_sources.append(normalized_source)

    if not ordered_sources:
        ordered_sources = list(required_sources)

    if needs_doc_confirmation:
        routing_plan.append(
            {
                "step_type": "knowledge_check",
                "source_type": "document",
                "source_ref": "knowledge.md",
                "target_items": [],
                "operation_hint": "Confirm business rules and metric definitions before executing data access.",
                "join_anchor": None,
                "depends_on": [],
                "knowledge_constraints": rule_constraints,
            }
        )

    for index, source_ref in enumerate(ordered_sources, start=1):
        source_kind = "document"
        lowered_source = source_ref.lower()
        if lowered_source.endswith((".db", ".sqlite")):
            source_kind = "sqlite"
        elif lowered_source.endswith(".csv"):
            source_kind = "csv"
        elif lowered_source.endswith(".json"):
            source_kind = "json"

        target_items = [
            {
                "field_ref": item.get("field_ref"),
                "display_name": item.get("display_name"),
                "source_path": item.get("source_path"),
                "semantic_role": item.get("semantic_role"),
            }
            for item in selected_source_items
            if item.get("source_file") == source_ref
        ]
        anchor_name = next(
            (
                anchor_name
                for anchor_name in selected_anchor_names
                if source_ref in anchors.get(anchor_name, {}).get("source_files", [])
            ),
            None,
        )
        depends_on = []
        if index > 1 and ordered_sources:
            depends_on.append(ordered_sources[index - 2])
        routing_plan.append(
            {
                "step_type": "source_access",
                "source_type": source_kind,
                "source_ref": source_ref,
                "target_items": target_items,
                "operation_hint": (
                    "Use source-specific exploration or execution to retrieve the required items."
                ),
                "join_anchor": anchor_name,
                "depends_on": depends_on,
                "knowledge_constraints": rule_constraints,
            }
        )

    if recommended_join_path:
        routing_plan.append(
            {
                "step_type": "join_or_align",
                "source_type": "cross_source",
                "source_ref": "semantic_catalog",
                "target_items": recommended_join_path,
                "operation_hint": "Align intermediate results using the recommended join path and available cross-source anchors.",
                "join_anchor": sorted(selected_anchor_names)[0] if selected_anchor_names else None,
                "depends_on": list(ordered_sources),
                "knowledge_constraints": rule_constraints,
            }
        )

    return routing_plan


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
    link_result: SchemaLinkResult | None = None,
    feedback: str | None = None,
) -> dict[str, Any]:
    lower_question = question.lower()
    reconstruction = _reconstruct_question(catalog, question)
    linker_query_units = list(link_result.query_units) if link_result is not None else []
    chosen_bindings = list(link_result.chosen_bindings) if link_result is not None else []
    candidate_bindings = list(link_result.candidate_bindings) if link_result is not None else []
    binding_conflicts = list(link_result.binding_conflicts) if link_result is not None else []
    join_candidates = list(link_result.join_candidates) if link_result is not None else []
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
    linked_entities.update(
        str(item.get("entity", "")).strip()
        for item in chosen_bindings
        if str(item.get("entity", "")).strip()
    )
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
        if any(str(item.get("entity", "")).strip() == entity.name for item in chosen_bindings):
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
    if link_result is not None and link_result.required_sources:
        required_sources = sorted(set(required_sources) | set(link_result.required_sources))

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
                "left_entity": str(item.get("left_entity", "")),
                "right_entity": str(item.get("right_entity", "")),
                "cardinality": "many_to_one",
            }
            for item in join_candidates
        ]
    if link_result is not None and link_result.candidate_join_paths:
        for edge in link_result.candidate_join_paths:
            if edge not in recommended_join_path:
                recommended_join_path.append(edge)

    for entity_name in linked_entities:
        for source in entity_sources.get(entity_name, []):
            if source not in required_sources:
                required_sources.append(source)

    if link_result is not None and not required_entities:
        linked_required_entities = [
            item["entity"]
            for item in link_result.chosen_bindings
            if str(item.get("entity", "")).strip()
        ]
        required_entities = sorted(set(linked_required_entities))

    replanning_guidance: list[str] = []
    if feedback:
        lowered_feedback = feedback.lower()
        if "empty result" in lowered_feedback or "zero rows" in lowered_feedback:
            replanning_guidance.extend(
                [
                    "Re-check filter values, casing, and time-format normalization before retrying.",
                    "If the query used a join, verify the join path and whether the driving table is correct.",
                ]
            )
        if "null-like" in lowered_feedback or "all null" in lowered_feedback:
            replanning_guidance.extend(
                [
                    "Re-check whether the projected field is coming from the correct table.",
                    "Inspect whether the join direction or join type is creating null-only projections.",
                ]
            )
        if "execution failures" in lowered_feedback or "field mappings" in lowered_feedback:
            replanning_guidance.extend(
                [
                    "Inspect schema names again before issuing another SQL or Python query.",
                    "Prefer a simpler validation query or schema preview before retrying a full join or aggregation.",
                ]
            )
        if "shape mismatch" in lowered_feedback or "scalar-like result" in lowered_feedback:
            replanning_guidance.extend(
                [
                    "Re-check whether the plan expects a scalar, a single-row summary, or a projected list before retrying.",
                    "Inspect whether aggregation, ranking, DISTINCT, or final projection cleanup is still missing.",
                ]
            )

    routing_candidates = _collect_routing_candidates(link_result)
    routing_plan = _build_routing_plan(
        catalog=catalog,
        chosen_bindings=chosen_bindings,
        required_sources=sorted(set(required_sources)),
        routing_candidates=routing_candidates,
        recommended_join_path=recommended_join_path,
        needs_doc_confirmation=has_document_risk,
    )

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
        "routing_plan": routing_plan,
        "expected_output_grain": expected_output_grain,
        "expected_output_columns": reconstruction["expected_output_columns"],
        "replanning_guidance": replanning_guidance,
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
                {
                    "slot_type": item.get("unit_type"),
                    "phrase": item.get("text"),
                    "role": item.get("role"),
                    "binding_eligible": item.get("binding_eligible"),
                }
                for item in linker_query_units
            ],
            "candidate_bindings": candidate_bindings,
            "chosen_bindings": chosen_bindings,
            "join_candidates": join_candidates,
            "chosen_join_path": recommended_join_path,
            "routing_candidates": routing_candidates,
            "routing_plan": routing_plan,
            "binding_conflicts": binding_conflicts,
            "linker_debug": link_result.debug_view if link_result is not None else {},
            "feedback": feedback or "",
            "replanning_guidance": replanning_guidance,
        },
        "fallback_strategy": "If the metric definition is unclear, read the relevant document before answering.",
    }
