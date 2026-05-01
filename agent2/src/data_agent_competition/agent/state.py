from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_agent_competition.artifacts.schema import SemanticArtifact
from data_agent_competition.execution.types import ExecutionResult, PhysicalPlan
from data_agent_competition.runtime.result_writer import AnswerTable
from data_agent_competition.semantic.types import LogicalPlan, SemanticRoutingSpec, TaskBundle
from data_agent_competition.semantic.verifier import VerificationResult


@dataclass(slots=True)
class ControllerState:
    task: TaskBundle
    semantic_artifact: SemanticArtifact | None = None
    semantic_artifact_mode: str | None = None
    semantic_artifact_path: str | None = None
    semantic_routing: SemanticRoutingSpec | None = None
    logical_plan: LogicalPlan | None = None
    logical_verification: VerificationResult | None = None
    physical_plan: PhysicalPlan | None = None
    execution_result: ExecutionResult | None = None
    answer_table: AnswerTable | None = None
    final_verification: VerificationResult | None = None
    status: str = "initialized"
    failure_reason: str | None = None
    failure_signature: str | None = None
    semantic_attempts: int = 0
    trace_steps: list[dict[str, Any]] = field(default_factory=list)
    orchestration_backend: str = "sequential"
    trace_schema_version: str = "controller_trace.v1"

    def record_step(self, node: str, **payload: Any) -> None:
        self.trace_steps.append(
            {
                "schema_version": self.trace_schema_version,
                "node": node,
                **payload,
            }
        )
