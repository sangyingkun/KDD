from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.config import PROJECT_ROOT
from data_agent_baseline.semantic.builder import build_base_semantic_catalog
from data_agent_baseline.semantic.catalog import SemanticCatalog
from data_agent_baseline.semantic.overlay import apply_overlay, load_overlay_file
from data_agent_baseline.semantic.planner import plan_semantic_query
from data_agent_baseline.semantic.resolver import resolve_business_term
from data_agent_baseline.semantic.verifier import validate_answer_semantics
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _build_semantic_catalog(task: PublicTask) -> SemanticCatalog:
    base_catalog = build_base_semantic_catalog(task)
    global_overlay = load_overlay_file(
        PROJECT_ROOT / "configs" / "semantic_overlays" / "global" / "default.yaml"
    )
    task_overlay = load_overlay_file(
        PROJECT_ROOT / "configs" / "semantic_overlays" / "tasks" / f"{task.task_id}.yaml"
    )
    return apply_overlay(apply_overlay(base_catalog, global_overlay), task_overlay)


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
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


def _describe_semantics(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = _build_semantic_catalog(task)
    return ToolExecutionResult(
        ok=True,
        content=catalog.summary(
            max_items_per_section=int(action_input.get("max_items_per_section", 10)),
            include_evidence=bool(action_input.get("include_evidence", False)),
        ),
    )


def _resolve_business_term(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = _build_semantic_catalog(task)
    return ToolExecutionResult(
        ok=True,
        content=resolve_business_term(
            catalog,
            term=str(action_input["term"]),
            expected_types=list(action_input.get("expected_types", [])) or None,
            top_k=int(action_input.get("top_k", 5)),
        ),
    )


def _plan_semantic_query(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = _build_semantic_catalog(task)
    return ToolExecutionResult(
        ok=True,
        content=plan_semantic_query(
            catalog,
            str(action_input.get("question", task.question)),
            target_metric=action_input.get("target_metric"),
            target_entity=action_input.get("target_entity"),
        ),
    )


def _validate_answer_semantics(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = _build_semantic_catalog(task)
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

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry() -> ToolRegistry:
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
        "plan_semantic_query": ToolSpec(
            name="plan_semantic_query",
            description="Suggest an execution path, required sources, and output grain for a question using the semantic catalog.",
            input_schema={"question": "What is total order amount by region?"},
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
        ),
    }
    handlers = {
        "answer": _answer,
        "describe_semantics": _describe_semantics,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "plan_semantic_query": _plan_semantic_query,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
        "resolve_business_term": _resolve_business_term,
        "validate_answer_semantics": _validate_answer_semantics,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
