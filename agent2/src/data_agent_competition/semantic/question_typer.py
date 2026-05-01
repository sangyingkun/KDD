from __future__ import annotations

from data_agent_competition.semantic.types import QuestionType, TaskBundle


RANKING_HINTS = {"top", "highest", "lowest", "most", "least", "rank", "largest", "smallest"}
AGGREGATION_HINTS = {"count", "sum", "average", "avg", "total", "ratio", "percentage", "percent"}
COMPARISON_HINTS = {"compare", "difference", "versus", "than", "between"}
TEMPORAL_HINTS = {
    "year",
    "month",
    "date",
    "daily",
    "weekly",
    "quarter",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
EXTRACTION_HINTS = {"list", "show", "return", "identify", "find", "which", "what"}


def classify_question(task: TaskBundle) -> QuestionType:
    question = task.question.lower()
    flags: list[QuestionType] = []
    if any(token in question for token in RANKING_HINTS):
        flags.append(QuestionType.RANKING)
    if any(token in question for token in AGGREGATION_HINTS):
        flags.append(QuestionType.AGGREGATION)
    if any(token in question for token in COMPARISON_HINTS):
        flags.append(QuestionType.COMPARISON)
    if any(token in question for token in TEMPORAL_HINTS):
        flags.append(QuestionType.TEMPORAL)
    if any(token in question for token in EXTRACTION_HINTS):
        flags.append(QuestionType.EXTRACTION)
    if not flags:
        return QuestionType.LOOKUP
    if len(set(flags)) > 1:
        return QuestionType.HYBRID
    return flags[0]
