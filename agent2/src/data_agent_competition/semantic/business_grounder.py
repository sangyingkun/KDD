from __future__ import annotations

import re

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.semantic.graph_types import GraphCandidateSet
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.semantic.normalization import normalize_identifier, question_ngrams
from data_agent_competition.semantic.schemas import grounded_terms_schema
from data_agent_competition.semantic.types import GroundedTerm, GroundingResult, QuestionType, TaskBundle

QUOTED_LITERAL_PATTERN = re.compile(r"'([^']+)'|\"([^\"]+)\"")
VALUE_LITERAL_PATTERN = re.compile(
    r"'([^']+)'\s+(?:indicating|means?|for|being)\s+([^.;]+?)(?=(?:\s+and\s+'|[.;]|$))",
    re.IGNORECASE,
)


def ground_business_context(
    task: TaskBundle,
    artifact: SemanticArtifact,
    question_type: QuestionType,
    runtime: SemanticRuntime,
    graph_candidates: GraphCandidateSet,
    retrieval_context: tuple[str, ...] = (),
) -> GroundingResult:
    grounded_terms: list[GroundedTerm] = []
    unresolved_terms: set[str] = set()
    supporting_facts = tuple(candidate.canonical_text for candidate in graph_candidates.value_candidates[:8])

    for phrase in question_ngrams(task.question):
        field_grounding = _field_grounding(phrase, graph_candidates)
        if field_grounding is not None:
            grounded_terms.append(field_grounding)
            continue

        value_grounding = _value_grounding(phrase, graph_candidates)
        if value_grounding is not None:
            grounded_terms.append(value_grounding)
            continue

        if " " not in phrase and len(phrase) > 2:
            unresolved_terms.add(phrase)

    grounded_terms.extend(_quoted_literal_groundings(task.question, artifact, graph_candidates))
    grounded_terms.extend(_llm_groundings(task, graph_candidates, runtime, retrieval_context))
    grounded_terms = _deduplicate_groundings(grounded_terms)
    return GroundingResult(
        question_type=question_type,
        grounded_terms=tuple(grounded_terms),
        unresolved_terms=tuple(sorted(unresolved_terms)),
        supporting_facts=supporting_facts,
        notes=("graph_backed_grounding",),
    )


def _field_grounding(phrase: str, graph_candidates: GraphCandidateSet) -> GroundedTerm | None:
    normalized_phrase = normalize_identifier(phrase)
    for candidate in graph_candidates.field_candidates:
        if candidate.field_name is None:
            continue
        aliases = {
            normalize_identifier(candidate.field_name),
            normalize_identifier(candidate.label),
            normalize_identifier(candidate.canonical_text),
        }
        if normalized_phrase not in aliases:
            continue
        return GroundedTerm(
            term=phrase,
            normalized_term=normalized_phrase,
            grounding_type="graph_field",
            source_id=candidate.source_id,
            field_name=candidate.field_name,
            confidence=min(candidate.score, 0.98),
            evidence=candidate.evidence_node_ids,
            graph_node_id=candidate.node_id,
        )
    return None


def _value_grounding(phrase: str, graph_candidates: GraphCandidateSet) -> GroundedTerm | None:
    normalized_phrase = normalize_identifier(phrase)
    for candidate in graph_candidates.value_candidates:
        source_id, field_name = _infer_field_binding(candidate, graph_candidates)
        concept_value = _best_matching_mapping(normalized_phrase, candidate.canonical_text)
        if concept_value is None or source_id is None or field_name is None:
            continue
        _, resolved_value = concept_value
        return GroundedTerm(
            term=phrase,
            normalized_term=normalized_phrase,
            grounding_type="graph_value_mapping",
            resolved_value=resolved_value,
            source_id=source_id,
            field_name=field_name,
            confidence=min(candidate.score, 0.99),
            evidence=candidate.evidence_node_ids,
            graph_node_id=candidate.node_id,
        )
    return None


def _quoted_literal_groundings(
    question: str,
    artifact: SemanticArtifact,
    graph_candidates: GraphCandidateSet,
) -> list[GroundedTerm]:
    literal_values = [match.group(1) or match.group(2) for match in QUOTED_LITERAL_PATTERN.finditer(question)]
    grounded_terms: list[GroundedTerm] = []
    field_candidates = {(candidate.source_id, candidate.field_name): candidate for candidate in graph_candidates.field_candidates}
    for literal in literal_values:
        literal_norm = normalize_identifier(literal)
        for source in artifact.sources:
            for field in source.fields:
                samples = {normalize_identifier(value) for value in field.sample_values if value}
                if literal_norm not in samples:
                    continue
                field_candidate = field_candidates.get((source.source_id, field.field_name))
                grounded_terms.append(
                    GroundedTerm(
                        term=literal,
                        normalized_term=literal_norm,
                        grounding_type="quoted_literal",
                        resolved_value=literal,
                        source_id=source.source_id,
                        field_name=field.field_name,
                        confidence=0.95,
                        evidence=field_candidate.evidence_node_ids if field_candidate else (literal,),
                        graph_node_id=field_candidate.node_id if field_candidate else None,
                    )
                )
        if any(item.term == literal for item in grounded_terms):
            continue
        heuristic = _heuristic_literal_grounding(question, literal, graph_candidates)
        if heuristic is not None:
            grounded_terms.append(heuristic)
    return grounded_terms


def _llm_groundings(
    task: TaskBundle,
    graph_candidates: GraphCandidateSet,
    runtime: SemanticRuntime,
    retrieval_context: tuple[str, ...],
) -> list[GroundedTerm]:
    payload = runtime.llm_client.call_structured(
        system_prompt=(
            "You are a constrained semantic grounding engine. "
            "Return only grounded terms that map to listed source ids and field names. "
            "If evidence is weak, return an empty list."
        ),
        user_prompt=_grounder_prompt(task, graph_candidates, retrieval_context),
        function_name="ground_business_terms",
        schema=grounded_terms_schema(),
    )
    if not payload:
        return []
    valid_fields = {
        (candidate.source_id, candidate.field_name): candidate
        for candidate in graph_candidates.field_candidates
        if candidate.source_id and candidate.field_name
    }
    grounded: list[GroundedTerm] = []
    for item in payload.get("grounded_terms", []):
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        source_id = _nullable(item.get("source_id"))
        field_name = _nullable(item.get("field_name"))
        if source_id is not None and field_name is not None and (source_id, field_name) not in valid_fields:
            continue
        grounded.append(
            GroundedTerm(
                term=term,
                normalized_term=normalize_identifier(term),
                grounding_type=str(item.get("grounding_type", "llm")),
                resolved_value=_nullable(item.get("resolved_value")),
                source_id=source_id,
                field_name=field_name,
                confidence=float(item.get("confidence", 0.0)),
                evidence=tuple(str(value) for value in item.get("evidence", []) if str(value).strip()),
                graph_node_id=_nullable(item.get("graph_node_id")),
            )
        )
    return grounded


def _grounder_prompt(
    task: TaskBundle,
    graph_candidates: GraphCandidateSet,
    retrieval_context: tuple[str, ...],
) -> str:
    fields = "\n".join(
        f"- {candidate.source_id}.{candidate.field_name}: {candidate.canonical_text}"
        for candidate in graph_candidates.field_candidates[:16]
        if candidate.source_id and candidate.field_name
    )
    values = "\n".join(
        f"- {candidate.label}: {candidate.canonical_text}"
        for candidate in graph_candidates.value_candidates[:12]
    )
    return (
        f"Question: {task.question}\n"
        "Retrieved graph context:\n"
        f"{chr(10).join(retrieval_context) if retrieval_context else '- none'}\n"
        "Candidate fields:\n"
        f"{fields if fields else '- none'}\n"
        "Candidate value concepts:\n"
        f"{values if values else '- none'}\n"
        "Identify business literals and implied filter values using only these candidates."
    )


def _infer_field_binding(
    candidate,
    graph_candidates: GraphCandidateSet,
) -> tuple[str | None, str | None]:
    bound_refs = [
        str(item).strip()
        for item in candidate.metadata.get("bound_field_refs", [])
        if str(item).strip()
    ]
    if bound_refs:
        source_id, field_name = bound_refs[0].rsplit(".", maxsplit=1)
        return source_id, field_name
    source_id = _nullable(candidate.metadata.get("source_id"))
    field_name = _nullable(candidate.metadata.get("field_name"))
    if source_id and field_name:
        return source_id, field_name
    normalized_text = normalize_identifier(candidate.canonical_text)
    for field_candidate in graph_candidates.field_candidates:
        if not field_candidate.source_id or not field_candidate.field_name:
            continue
        field_norm = normalize_identifier(field_candidate.field_name)
        if field_norm and field_norm in normalized_text:
            return field_candidate.source_id, field_candidate.field_name
    return None, None

def _best_matching_mapping(normalized_phrase: str, text: str) -> tuple[str, str] | None:
    best: tuple[float, tuple[str, str]] | None = None
    for match in VALUE_LITERAL_PATTERN.finditer(text):
        value = match.group(1).strip()
        concept = normalize_identifier(match.group(2))
        score = _mapping_match_score(normalized_phrase, concept)
        if score <= 0:
            continue
        candidate = (concept, value)
        if best is None or score > best[0]:
            best = (score, candidate)
    return None if best is None else best[1]


def _mapping_match_score(normalized_phrase: str, concept: str) -> float:
    if not normalized_phrase or not concept:
        return 0.0
    if normalized_phrase == concept:
        return 1.0
    concept_tokens = [token for token in concept.split("_") if token]
    if normalized_phrase in concept_tokens:
        return 0.9
    if concept.startswith(f"{normalized_phrase}_"):
        return 0.85
    return 0.0


def _heuristic_literal_grounding(
    question: str,
    literal: str,
    graph_candidates: GraphCandidateSet,
) -> GroundedTerm | None:
    literal_norm = normalize_identifier(literal)
    best_candidate = None
    best_score = 0.0
    for candidate in graph_candidates.field_candidates:
        if not candidate.source_id or not candidate.field_name:
            continue
        field_name = candidate.field_name.lower()
        score = 0.0
        if "event" in question.lower() and "name" in field_name:
            score += 0.6
        if "name" in field_name:
            score += 0.2
        if "title" in field_name or "description" in field_name:
            score += 0.15
        if "category" in field_name:
            score += 0.1
        score += min(candidate.score, 1.0) * 0.2
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_candidate is None or best_score < 0.45:
        return None
    return GroundedTerm(
        term=literal,
        normalized_term=literal_norm,
        grounding_type="quoted_literal_heuristic",
        resolved_value=literal,
        source_id=best_candidate.source_id,
        field_name=best_candidate.field_name,
        confidence=min(best_score, 0.85),
        evidence=best_candidate.evidence_node_ids,
        graph_node_id=best_candidate.node_id,
    )


def _deduplicate_groundings(grounded_terms: list[GroundedTerm]) -> list[GroundedTerm]:
    best_by_key: dict[tuple[str, str | None, str | None, str | None], GroundedTerm] = {}
    for item in grounded_terms:
        key = (item.normalized_term, item.source_id, item.field_name, item.resolved_value)
        previous = best_by_key.get(key)
        if previous is None or item.confidence > previous.confidence:
            best_by_key[key] = item
    return list(best_by_key.values())


def _nullable(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None
