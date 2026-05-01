# Competition Artifacts

Each task gets a committed semantic artifact under:

```text
competition_artifacts/task_xxx/semantic_artifact.json
```

The file is the static semantic input for runtime reasoning. It is not a cache.

## Required Sections

- `task_id`
- `assets`
- `knowledge_facts`
- `sources`
- `join_candidates`
- `doc_chunks`

## Design Rules

- Paths are relative to the task `context/` directory.
- Source descriptors are normalized objects, not raw file dumps.
- Join candidates store explicit left/right field pairs.
- Knowledge facts should be atomic statements when possible.
- Doc chunks should stay small enough for semantic grounding and evidence reuse.
