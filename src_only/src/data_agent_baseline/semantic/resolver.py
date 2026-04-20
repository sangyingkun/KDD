from __future__ import annotations

from typing import Any

from data_agent_baseline.semantic.catalog import SemanticCatalog
from data_agent_baseline.semantic.builder import _normalize_identifier


def _singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 2:
        return token[:-1]
    return token


def _normalized_forms(value: str) -> set[str]:
    normalized = _normalize_identifier(value)
    singular = "_".join(_singularize_token(part) for part in normalized.split("_") if part)
    forms = {value.strip().lower(), normalized, singular, normalized.replace("_", " "), singular.replace("_", " ")}
    return {form for form in forms if form}


def _score_match(term: str, candidate: str, aliases: list[str]) -> float:
    normalized_term_forms = _normalized_forms(term)
    normalized_name_forms = _normalized_forms(candidate)
    normalized_alias_forms = {
        form
        for alias in aliases
        for form in _normalized_forms(alias)
    }
    raw_term = term.strip().lower()
    raw_name = candidate.lower()
    raw_aliases = [alias.lower() for alias in aliases]

    if raw_term == raw_name:
        return 1.0
    if raw_term in raw_aliases:
        return 0.9
    if normalized_term_forms & normalized_name_forms:
        return 0.8
    if normalized_term_forms & normalized_alias_forms:
        return 0.75
    return 0.0


def resolve_business_term(
    catalog: SemanticCatalog,
    term: str,
    expected_types: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    normalized = term.strip().lower()
    matches: list[dict[str, Any]] = []

    def add_match(
        object_type: str,
        name: str,
        aliases: list[str],
        confidence: str,
        evidence_refs: list[str] | None = None,
    ) -> None:
        score = _score_match(term, name, aliases)
        if score > 0:
            matches.append(
                {
                    "object_type": object_type,
                    "name": name,
                    "score": score,
                    "confidence": confidence,
                    "reason": f"Matched `{term}` to {object_type} `{name}`.",
                    "evidence_refs": list(evidence_refs or []),
                }
            )

    if not expected_types or "entity" in expected_types:
        for entity in catalog.entities:
            add_match("entity", entity.name, entity.aliases, entity.confidence)

    if not expected_types or "dimension" in expected_types:
        for dimension in catalog.dimensions:
            add_match("dimension", dimension.name, dimension.aliases, dimension.confidence)

    if not expected_types or "measure" in expected_types:
        for measure in catalog.measures:
            add_match("measure", measure.name, [], measure.confidence)

    if not expected_types or "metric" in expected_types:
        for metric in catalog.metrics:
            add_match("metric", metric.name, [], metric.confidence, metric.evidence_refs)

    matches.sort(key=lambda item: (-item["score"], item["name"]))
    top_matches = matches[:top_k]
    return {
        "matches": top_matches,
        "ambiguities": (
            [item for item in top_matches if item["score"] == matches[0]["score"]]
            if matches
            else []
        ),
        "suggested_followups": (
            []
            if len(matches) <= 1
            else ["Use read_doc or inspect_sqlite_schema to confirm the intended meaning."]
        ),
    }
