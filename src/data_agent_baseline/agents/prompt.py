from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Semantic context is already bootstrapped for you before free-form exploration begins.
2. Resolve ambiguity before assuming field, metric, or business-term meanings.
3. Use planning hints before expensive SQL or Python execution.
4. Base your answer only on information you can observe through the provided tools.
5. The task is complete only when you call the `answer` tool.
6. The `answer` tool must receive a table with `columns` and `rows`.
7. Validate high-risk answers before final submission whenever joins, aggregations, or derived metrics are involved.
8. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
9. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
10. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to understand task semantics first:
```json
{"thought":"I should inspect the available files before choosing a data source.","action":"list_context","action_input":{"max_depth":3}}
```

Example response when you need to inspect a database schema:
```json
{"thought":"I should inspect the sqlite schema before writing SQL.","action":"inspect_sqlite_schema","action_input":{"path":"orders.sqlite"}}
```

Example response when you need to inspect the context directly:
```json
{"thought":"I already know the relevant file and need the raw data.","action":"read_csv","action_input":{"path":"orders.csv","max_rows":20}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        "Execution workflow:\n"
        "- Semantic linking and planning context has already been injected before you act.\n"
        "- Use that prior context instead of re-deriving the task from scratch.\n"
        "- Resolve ambiguity before assuming field or metric meanings.\n"
        "- Use planning hints before expensive SQL or Python execution.\n"
        "- Fall back to raw tools when semantic confidence is low.\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    prompt_parts = [f"Observation:\n{rendered}"]
    runtime_feedback = observation.get("runtime_feedback")
    if isinstance(runtime_feedback, dict):
        prompt_lines = runtime_feedback.get("prompt_lines", [])
        if isinstance(prompt_lines, list) and prompt_lines:
            guidance = "\n".join(f"- {str(line)}" for line in prompt_lines if str(line).strip())
            if guidance:
                prompt_parts.append(f"Runtime Guidance:\n{guidance}")
    return "\n\n".join(prompt_parts)
