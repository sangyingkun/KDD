# Agent2 Competition Kernel

`agent2/` is an isolated build area for the next-generation competition kernel.

It is intentionally separated from the existing baseline implementation so the
new semantic and execution architecture can evolve without changing the
original runtime shell contract.

## Layout

- `src/data_agent_competition/`: implementation
- `configs/`: runnable local configs
- `artifacts/runs/`: the shared benchmark-style output location used by the baseline runtime

## Run

```bash
uv run --project agent2 dabench-agent2 status --config configs/react_baseline.example.yaml
uv run --project agent2 dabench-agent2 inspect-task task_11 --config configs/react_baseline.example.yaml
uv run --project agent2 dabench-agent2 run-task task_11 --config configs/react_baseline.example.yaml
```

## Architecture Notes

- `runtime/`: benchmark-facing CLI and output writing
- `agent/`: controller state machine, LangGraph orchestration, bounded replan policy
- `semantic/`: ordered semantic pipeline and verification
- `execution/`: physical planning and deterministic execution
- semantic snapshots are built inside the kernel at runtime and are not part of the public run interface

## Controller Backend

The controller is implemented with LangGraph-compatible nodes. If LangGraph is
available in the Python environment, `agent2` uses it as the orchestration
backend. Otherwise the same node flow runs through a sequential fallback with
the same state shape and trace structure.
