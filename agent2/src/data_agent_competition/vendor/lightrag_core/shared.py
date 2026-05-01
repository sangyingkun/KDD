from __future__ import annotations

from dataclasses import dataclass

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.semantic.graph_linker import link_graph_candidates
from data_agent_competition.semantic.graph_types import GraphCandidateSet, GraphRetrievalResult
from data_agent_competition.semantic.question_typer import classify_question
from data_agent_competition.semantic.types import QuestionType, TaskBundle
from data_agent_competition.vendor.lightrag_core.base import QueryParam


@dataclass(frozen=True, slots=True)
class LightRAGSemanticResult:
    retrieval: GraphRetrievalResult
    candidates: GraphCandidateSet


def build_query_param(task: TaskBundle) -> QueryParam:
    from data_agent_competition.semantic.normalization import question_terms

    keywords = list(question_terms(task.question))
    high_level = [token for token in keywords if len(token) > 5][:8]
    low_level = keywords[:12]
    question_type = classify_question(task)
    if question_type in {QuestionType.EXPLANATORY, QuestionType.HYBRID}:
        mode = "mix"
    elif question_type in {QuestionType.AGGREGATION, QuestionType.COMPARISON, QuestionType.RANKING}:
        mode = "hybrid"
    else:
        mode = "local"
    return QueryParam(
        mode=mode,
        top_k=12,
        chunk_top_k=8,
        hl_keywords=high_level,
        ll_keywords=low_level,
        include_references=True,
    )


def build_semantic_candidates(
    *,
    question: str,
    artifact: SemanticArtifact,
    retrieval: GraphRetrievalResult,
) -> GraphCandidateSet:
    return link_graph_candidates(
        question=question,
        artifact=artifact,
        retrieval=retrieval,
    )
