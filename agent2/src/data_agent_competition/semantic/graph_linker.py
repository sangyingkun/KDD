from __future__ import annotations

from collections import defaultdict

from data_agent_competition.artifacts.schema import GraphNode, SemanticArtifact
from data_agent_competition.semantic.graph_types import (
    GraphCandidate,
    GraphCandidateSet,
    GraphRetrievalResult,
)
from data_agent_competition.semantic.normalization import normalize_identifier, question_ngrams, token_overlap_score


def link_graph_candidates(
    *,
    question: str,
    artifact: SemanticArtifact,
    retrieval: GraphRetrievalResult,
) -> GraphCandidateSet:
    node_index = {node.node_id: node for node in artifact.graph_nodes}
    supporting_edges = defaultdict(set)
    for edge in retrieval.edge_hits:
        supporting_edges[edge.source_node_id].add(edge.edge_id)
        supporting_edges[edge.target_node_id].add(edge.edge_id)

    source_candidates: list[GraphCandidate] = []
    field_candidates: list[GraphCandidate] = []
    join_candidates: list[GraphCandidate] = []
    value_candidates: list[GraphCandidate] = []
    metric_candidates: list[GraphCandidate] = []
    constraint_candidates: list[GraphCandidate] = []
    ambiguity_candidates: list[GraphCandidate] = []
    use_case_candidates: list[GraphCandidate] = []

    for hit in retrieval.node_hits:
        node = node_index.get(hit.node_id)
        if node is None:
            continue
        candidate = _candidate_from_hit(
            question=question,
            node=node,
            hit=hit,
            edge_ids=tuple(sorted(supporting_edges.get(node.node_id, ()))),
        )
        if node.node_type == "source":
            source_candidates.append(candidate)
        elif node.node_type == "field":
            field_candidates.append(candidate)
        elif node.node_type == "join":
            join_candidates.append(candidate)
        elif node.node_type == "value_concept":
            value_candidates.append(candidate)
        elif node.node_type == "metric":
            metric_candidates.append(candidate)
        elif node.node_type == "constraint":
            constraint_candidates.append(candidate)
        elif node.node_type == "ambiguity":
            ambiguity_candidates.append(candidate)
        elif node.node_type == "use_case":
            use_case_candidates.append(candidate)

    linked_sources = _top_unique(source_candidates, key=lambda item: item.source_id or item.node_id, limit=4)
    source_ids = {
        source_id
        for candidate in linked_sources
        for source_id in _candidate_source_ids(candidate)
    }
    linked_fields = _top_unique(
        [
            candidate
            for candidate in field_candidates
            if candidate.source_id in source_ids
            or any(bound_source_id in source_ids for bound_source_id in _candidate_source_ids(candidate))
            or _question_support(question, candidate.label)
        ],
        key=lambda item: (item.source_id or "", item.field_name or item.node_id),
        limit=12,
    )
    linked_joins = _top_unique(
        [
            candidate
            for candidate in join_candidates
            if _join_touches_sources(candidate, source_ids)
        ],
        key=lambda item: item.node_id,
        limit=8,
    )
    linked_values = _top_unique(value_candidates, key=lambda item: item.node_id, limit=8)

    evidence_node_ids = tuple(dict.fromkeys(hit.node_id for hit in retrieval.node_hits))
    evidence_edge_ids = tuple(dict.fromkeys(edge.edge_id for edge in retrieval.edge_hits))
    return GraphCandidateSet(
        source_candidates=tuple(linked_sources),
        field_candidates=tuple(linked_fields),
        join_candidates=tuple(linked_joins),
        value_candidates=tuple(linked_values),
        metric_candidates=tuple(_top_unique(metric_candidates, key=lambda item: item.node_id, limit=8)),
        constraint_candidates=tuple(_top_unique(constraint_candidates, key=lambda item: item.node_id, limit=12)),
        ambiguity_candidates=tuple(_top_unique(ambiguity_candidates, key=lambda item: item.node_id, limit=12)),
        use_case_candidates=tuple(_top_unique(use_case_candidates, key=lambda item: item.node_id, limit=8)),
        evidence_node_ids=evidence_node_ids,
        evidence_edge_ids=evidence_edge_ids,
    )


def _candidate_from_hit(
    *,
    question: str,
    node: GraphNode,
    hit,
    edge_ids: tuple[str, ...],
) -> GraphCandidate:
    lexical_alignment = _alignment_bonus(question, node)
    score = hit.score + lexical_alignment + _metadata_bonus(node)
    source_id = _str_value(node.metadata.get("source_id"))
    field_name = _str_value(node.metadata.get("field_name"))
    resolved_value = _resolved_value(node)
    return GraphCandidate(
        node_id=node.node_id,
        node_type=node.node_type,
        label=node.label,
        canonical_text=node.canonical_text,
        score=score,
        metadata=dict(node.metadata),
        source_id=source_id,
        field_name=field_name,
        resolved_value=resolved_value,
        rationale=hit.rationale,
        evidence_node_ids=(node.node_id,),
        evidence_edge_ids=edge_ids,
    )


def _alignment_bonus(question: str, node: GraphNode) -> float:
    ngrams = question_ngrams(question)
    score = 0.0
    for phrase in ngrams:
        score = max(score, float(token_overlap_score(phrase, node.label)) * 0.15)
        score = max(score, float(token_overlap_score(phrase, node.canonical_text)) * 0.08)
    return score


def _metadata_bonus(node: GraphNode) -> float:
    metadata = node.metadata
    bonus = 0.0
    if metadata.get("aliases"):
        bonus += 0.04
    if metadata.get("semantic_tags"):
        bonus += 0.03
    if metadata.get("source_hint"):
        bonus += 0.05
    return bonus


def _top_unique(candidates: list[GraphCandidate], *, key, limit: int) -> list[GraphCandidate]:
    best: dict[object, GraphCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.node_id)):
        dedupe_key = key(candidate)
        if dedupe_key not in best:
            best[dedupe_key] = candidate
    return list(best.values())[:limit]


def _join_touches_sources(candidate: GraphCandidate, source_ids: set[str | None]) -> bool:
    candidate_source_ids = _candidate_source_ids(candidate)
    if len(candidate_source_ids & {source_id for source_id in source_ids if source_id}) >= 2:
        return True
    normalized = normalize_identifier(candidate.canonical_text)
    matched = 0
    for source_id in source_ids:
        if not source_id:
            continue
        if normalize_identifier(source_id) in normalized:
            matched += 1
    return matched >= 2


def _question_support(question: str, label: str) -> bool:
    return token_overlap_score(question, label) > 0


def _resolved_value(node: GraphNode) -> str | None:
    explicit = _str_value(node.metadata.get("resolved_value"))
    if explicit:
        return explicit
    normalized = normalize_identifier(node.canonical_text)
    if "maps value" in normalized:
        return node.label
    return None


def _str_value(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _candidate_source_ids(candidate: GraphCandidate) -> set[str]:
    source_ids: set[str] = set()
    if candidate.source_id:
        source_ids.add(candidate.source_id)
    for value in candidate.metadata.get("bound_source_ids", []):
        rendered = str(value).strip()
        if rendered:
            source_ids.add(rendered)
    left_source_id = _str_value(candidate.metadata.get("left_source_id"))
    right_source_id = _str_value(candidate.metadata.get("right_source_id"))
    if left_source_id:
        source_ids.add(left_source_id)
    if right_source_id:
        source_ids.add(right_source_id)
    return source_ids
