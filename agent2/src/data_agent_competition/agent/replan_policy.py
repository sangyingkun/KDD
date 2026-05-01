from __future__ import annotations

from dataclasses import dataclass

from data_agent_competition.execution.types import ExecutionResult
from data_agent_competition.semantic.types import LogicalPlan
from data_agent_competition.semantic.verifier import VerificationResult


ALLOWED_REPLAN_SIGNATURES = {
    "missing_answer_columns",
    "missing_joins",
    "empty_answer",
    "sparse_answer_column",
    "target_grain_not_projected",
    "duplicate_target_grain",
}


@dataclass(frozen=True, slots=True)
class ReplanDecision:
    should_replan: bool
    signature: str | None
    reason: str


def decide_replan(
    *,
    logical_plan: LogicalPlan | None,
    logical_verification: VerificationResult | None,
    final_verification: VerificationResult | None,
    execution_result: ExecutionResult | None,
    semantic_attempts: int,
    max_semantic_retries: int,
) -> ReplanDecision:
    if semantic_attempts >= max_semantic_retries + 1:
        return ReplanDecision(False, None, "semantic retry budget exhausted")

    signature = _failure_signature(logical_verification, final_verification, execution_result)
    if signature is None:
        return ReplanDecision(False, None, "no bounded semantic failure detected")
    if signature not in ALLOWED_REPLAN_SIGNATURES:
        return ReplanDecision(False, signature, "failure signature not eligible for bounded replan")
    if logical_plan is None:
        return ReplanDecision(False, signature, "logical plan missing")
    return ReplanDecision(True, signature, "bounded semantic replan allowed")


def _failure_signature(
    logical_verification: VerificationResult | None,
    final_verification: VerificationResult | None,
    execution_result: ExecutionResult | None,
) -> str | None:
    for verification in (logical_verification, final_verification):
        if verification is None or verification.ok:
            continue
        return verification.signature
    if execution_result is not None and execution_result.succeeded and not execution_result.answer_rows:
        return "empty_answer"
    return None
