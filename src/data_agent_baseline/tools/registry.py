from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.config import PROJECT_ROOT, RetrievalConfig
from data_agent_baseline.semantic.builder import build_base_semantic_catalog
from data_agent_baseline.semantic.catalog import SemanticCatalog
from data_agent_baseline.semantic.embedding import EmbeddingProvider, create_embedding_provider
from data_agent_baseline.semantic.linker import SchemaLinkResult, link_schema_candidates
from data_agent_baseline.semantic.overlay import apply_overlay, load_overlay_file
from data_agent_baseline.semantic.planner import plan_semantic_query
from data_agent_baseline.semantic.retrieval import TaskRetrievalIndex, build_task_retrieval_index
from data_agent_baseline.semantic.resolver import resolve_business_term
from data_agent_baseline.semantic.verifier import validate_answer_semantics
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import PythonSession, execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30

PROFILE_AGENT_CORE = "agent_core"
PROFILE_SYSTEM_BOOTSTRAP = "system_bootstrap"
PROFILE_VALIDATION = "validation"


class ToolVisibility(str, Enum):
    SYSTEM = "system"
    AGENT = "agent"
    VALIDATION = "validation"


class ToolStage(str, Enum):
    BOOTSTRAP = "bootstrap"
    EXECUTION = "execution"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    visibility: ToolVisibility = ToolVisibility.AGENT
    stage: ToolStage = ToolStage.EXECUTION


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[["ToolRegistry", PublicTask, dict[str, Any]], ToolExecutionResult]


def _build_semantic_catalog(task: PublicTask) -> SemanticCatalog:
    base_catalog = build_base_semantic_catalog(task)
    global_overlay = load_overlay_file(
        PROJECT_ROOT / "configs" / "semantic_overlays" / "global" / "default.yaml"
    )
    task_overlay = load_overlay_file(
        PROJECT_ROOT / "configs" / "semantic_overlays" / "tasks" / f"{task.task_id}.yaml"
    )
    return apply_overlay(apply_overlay(base_catalog, global_overlay), task_overlay)


def _list_context(_: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(_: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(_: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(_: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _inspect_sqlite_schema(_: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(_: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(_: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    registry = _
    if registry.enable_stateful_python_session:
        content = registry.get_python_session(task).execute(
            code,
            timeout_seconds=registry.python_session_timeout_seconds,
        )
    else:
        content = execute_python_code(
            context_root=task.context_dir,
            code=code,
            timeout_seconds=registry.python_session_timeout_seconds,
        )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(registry: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    validation = validate_answer_semantics(
        registry.get_semantic_catalog(task),
        question=str(action_input.get("question", task.question)),
        columns=list(columns),
        rows=normalized_rows,
        derivation_summary=action_input.get("derivation_summary"),
        used_entities=list(action_input.get("used_entities", [])),
        used_measures=list(action_input.get("used_measures", [])),
        used_metrics=list(action_input.get("used_metrics", [])),
        used_relations=list(action_input.get("used_relations", [])),
    )
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


def _describe_semantics(registry: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = registry.get_semantic_catalog(task)
    return ToolExecutionResult(
        ok=True,
        content=catalog.summary(
            max_items_per_section=int(action_input.get("max_items_per_section", 10)),
            include_evidence=bool(action_input.get("include_evidence", False)),
        ),
    )


def _resolve_business_term(registry: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = registry.get_semantic_catalog(task)
    return ToolExecutionResult(
        ok=True,
        content=resolve_business_term(
            catalog,
            term=str(action_input["term"]),
            expected_types=list(action_input.get("expected_types", [])) or None,
            top_k=int(action_input.get("top_k", 5)),
        ),
    )


def _plan_semantic_query(registry: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    question = str(action_input.get("question", task.question))
    target_metric = action_input.get("target_metric")
    target_entity = action_input.get("target_entity")
    feedback = action_input.get("feedback")
    cached_plan = registry.get_semantic_plan(
        task,
        question=question,
        target_metric=target_metric,
        target_entity=target_entity,
        feedback=str(feedback) if feedback is not None else None,
    )
    return ToolExecutionResult(
        ok=True,
        content=cached_plan,
    )


def _link_schema_candidates(
    registry: "ToolRegistry",
    task: PublicTask,
    action_input: dict[str, Any],
) -> ToolExecutionResult:
    question = str(action_input.get("question", task.question))
    content = registry.get_schema_link_result(task, question=question)
    return ToolExecutionResult(ok=True, content=content)


def _validate_answer_semantics(registry: "ToolRegistry", task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = registry.get_semantic_catalog(task)
    return ToolExecutionResult(
        ok=True,
        content=validate_answer_semantics(
            catalog,
            question=str(action_input.get("question", task.question)),
            columns=list(action_input["columns"]),
            rows=list(action_input["rows"]),
            derivation_summary=action_input.get("derivation_summary"),
            used_entities=list(action_input.get("used_entities", [])),
            used_measures=list(action_input.get("used_measures", [])),
            used_metrics=list(action_input.get("used_metrics", [])),
            used_relations=list(action_input.get("used_relations", [])),
        ),
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    enable_stateful_python_session: bool = True
    python_session_timeout_seconds: int = EXECUTE_PYTHON_TIMEOUT_SECONDS
    _semantic_catalog_cache: dict[str, SemanticCatalog] = field(default_factory=dict)
    _semantic_plan_cache: dict[tuple[str, str, str | None, str | None, str | None], dict[str, Any]] = field(default_factory=dict)
    _retrieval_index_cache: dict[str, TaskRetrievalIndex] = field(default_factory=dict)
    _schema_link_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    _python_session_cache: dict[str, PythonSession] = field(default_factory=dict)
    _embedding_provider: EmbeddingProvider | None = None

    def get_specs_for_profile(self, profile: str) -> dict[str, ToolSpec]:
        if profile == PROFILE_SYSTEM_BOOTSTRAP:
            visibility = ToolVisibility.SYSTEM
        elif profile == PROFILE_VALIDATION:
            visibility = ToolVisibility.VALIDATION
        elif profile == PROFILE_AGENT_CORE:
            visibility = ToolVisibility.AGENT
        else:
            raise ValueError(f"Unknown tool profile: {profile}")
        return {
            name: spec
            for name, spec in sorted(self.specs.items())
            if spec.visibility == visibility
        }

    def _schema_from_example(self, value: Any) -> dict[str, Any]:
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string"}
        if value is None:
            return {"type": "string"}
        if isinstance(value, list):
            item_schema = self._schema_from_example(value[0]) if value else {"type": "string"}
            return {"type": "array", "items": item_schema}
        if isinstance(value, dict):
            properties = {key: self._schema_from_example(item) for key, item in value.items()}
            return {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        return {"type": "string"}

    def _tool_parameters_schema(self, spec: ToolSpec) -> dict[str, Any]:
        inferred = self._schema_from_example(spec.input_schema)
        if inferred.get("type") != "object":
            return {
                "type": "object",
                "properties": {"value": inferred},
                "required": ["value"],
                "additionalProperties": False,
            }
        inferred.setdefault("additionalProperties", False)
        return inferred

    def to_openai_tools_format(self, profile: str = PROFILE_AGENT_CORE) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec in self.get_specs_for_profile(profile).values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": self._tool_parameters_schema(spec),
                    },
                }
            )
        return tools

    def describe_for_prompt(self, profile: str = PROFILE_AGENT_CORE) -> str:
        lines = []
        for name, spec in self.get_specs_for_profile(profile).items():
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {json.dumps(spec.input_schema, ensure_ascii=False)}")
        return "\n".join(lines)

    def get_semantic_catalog(self, task: PublicTask) -> SemanticCatalog:
        cached = self._semantic_catalog_cache.get(task.task_id)
        if cached is not None:
            return cached
        catalog = _build_semantic_catalog(task)
        self._semantic_catalog_cache[task.task_id] = catalog
        return catalog

    def get_semantic_plan(
        self,
        task: PublicTask,
        *,
        question: str,
        target_metric: str | None,
        target_entity: str | None,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        cache_key = (task.task_id, question, target_metric, target_entity, feedback)
        cached = self._semantic_plan_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        link_result = self.get_schema_link_result(task, question=question, as_dataclass=True)
        plan = plan_semantic_query(
            self.get_semantic_catalog(task),
            question,
            target_metric=target_metric,
            target_entity=target_entity,
            link_result=link_result,
            feedback=feedback,
        )
        self._semantic_plan_cache[cache_key] = copy.deepcopy(plan)
        return copy.deepcopy(plan)

    def get_retrieval_index(self, task: PublicTask) -> TaskRetrievalIndex:
        cached = self._retrieval_index_cache.get(task.task_id)
        if cached is not None:
            return cached
        index = build_task_retrieval_index(self.get_semantic_catalog(task), self.get_embedding_provider())
        self._retrieval_index_cache[task.task_id] = index
        return index

    def get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = create_embedding_provider(self.retrieval_config)
        return self._embedding_provider

    def get_python_session(self, task: PublicTask) -> PythonSession:
        cached = self._python_session_cache.get(task.task_id)
        if cached is not None:
            return cached
        session = PythonSession(task.context_dir)
        self._python_session_cache[task.task_id] = session
        return session

    def cleanup_task_runtime(self, task_id: str) -> None:
        session = self._python_session_cache.pop(task_id, None)
        if session is not None:
            session.close(force=False)

    def cleanup_all_runtime(self) -> None:
        for task_id in list(self._python_session_cache):
            self.cleanup_task_runtime(task_id)

    def get_schema_link_result(
        self,
        task: PublicTask,
        *,
        question: str,
        as_dataclass: bool = False,
    ) -> dict[str, Any] | SchemaLinkResult:
        cache_key = (task.task_id, question)
        cached = self._schema_link_cache.get(cache_key)
        if cached is None:
            result = link_schema_candidates(
                question,
                self.get_semantic_catalog(task),
                self.get_retrieval_index(task),
                self.get_embedding_provider(),
                self.retrieval_config,
            )
            cached = {
                "query_units": copy.deepcopy(result.query_units),
                "top_entities": copy.deepcopy(result.top_entities),
                "top_fields": copy.deepcopy(result.top_fields),
                "top_knowledge": copy.deepcopy(result.top_knowledge),
                "candidate_bindings": copy.deepcopy(result.candidate_bindings),
                "chosen_bindings": copy.deepcopy(result.chosen_bindings),
                "binding_conflicts": copy.deepcopy(result.binding_conflicts),
                "join_candidates": copy.deepcopy(result.join_candidates),
                "candidate_join_paths": copy.deepcopy(result.candidate_join_paths),
                "required_sources": list(result.required_sources),
                "unresolved_ambiguities": copy.deepcopy(result.unresolved_ambiguities),
                "debug_view": copy.deepcopy(result.debug_view),
            }
            self._schema_link_cache[cache_key] = cached
        if as_dataclass:
            return SchemaLinkResult(
                query_units=copy.deepcopy(cached["query_units"]),
                top_entities=copy.deepcopy(cached["top_entities"]),
                top_fields=copy.deepcopy(cached["top_fields"]),
                top_knowledge=copy.deepcopy(cached["top_knowledge"]),
                candidate_bindings=copy.deepcopy(cached["candidate_bindings"]),
                chosen_bindings=copy.deepcopy(cached["chosen_bindings"]),
                binding_conflicts=copy.deepcopy(cached["binding_conflicts"]),
                join_candidates=copy.deepcopy(cached["join_candidates"]),
                candidate_join_paths=copy.deepcopy(cached["candidate_join_paths"]),
                required_sources=list(cached["required_sources"]),
                unresolved_ambiguities=copy.deepcopy(cached["unresolved_ambiguities"]),
                debug_view=copy.deepcopy(cached["debug_view"]),
            )
        return copy.deepcopy(cached)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](self, task, action_input)


def create_default_tool_registry(
    retrieval_config: RetrievalConfig | None = None,
    *,
    enable_stateful_python_session: bool = True,
    python_session_timeout_seconds: int = EXECUTE_PYTHON_TIMEOUT_SECONDS,
) -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "describe_semantics": ToolSpec(
            name="describe_semantics",
            description="Describe the task-level semantic catalog, including entities, relations, dimensions, measures, metrics, and warnings.",
            input_schema={"include_evidence": False, "max_items_per_section": 10},
            visibility=ToolVisibility.SYSTEM,
            stage=ToolStage.BOOTSTRAP,
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "link_schema_candidates": ToolSpec(
            name="link_schema_candidates",
            description=(
                "Link question phrases to task-scoped entities, fields, knowledge, and candidate join paths "
                "using hybrid retrieval before raw exploration."
            ),
            input_schema={"question": "For patients with severe degree of thrombosis, list their ID and sex."},
            visibility=ToolVisibility.SYSTEM,
            stage=ToolStage.BOOTSTRAP,
        ),
        "plan_semantic_query": ToolSpec(
            name="plan_semantic_query",
            description="Suggest an execution path, required sources, and output grain for a question using the semantic catalog.",
            input_schema={"question": "What is total order amount by region?"},
            visibility=ToolVisibility.SYSTEM,
            stage=ToolStage.BOOTSTRAP,
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
        "resolve_business_term": ToolSpec(
            name="resolve_business_term",
            description="Resolve a business term against entities, dimensions, measures, and metrics in the semantic catalog.",
            input_schema={"term": "customer", "expected_types": ["entity"], "top_k": 5},
        ),
        "validate_answer_semantics": ToolSpec(
            name="validate_answer_semantics",
            description="Validate a candidate answer against semantic grain, join, metric, and filter rules before final submission.",
            input_schema={"columns": ["total_amount"], "rows": [["99.5"]]},
            visibility=ToolVisibility.VALIDATION,
            stage=ToolStage.VALIDATION,
        ),
    }
    handlers = {
        "answer": _answer,
        "describe_semantics": _describe_semantics,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "link_schema_candidates": _link_schema_candidates,
        "plan_semantic_query": _plan_semantic_query,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
        "resolve_business_term": _resolve_business_term,
        "validate_answer_semantics": _validate_answer_semantics,
    }
    return ToolRegistry(
        specs=specs,
        handlers=handlers,
        retrieval_config=retrieval_config or RetrievalConfig(),
        enable_stateful_python_session=enable_stateful_python_session,
        python_session_timeout_seconds=python_session_timeout_seconds,
    )
