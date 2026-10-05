# workflow-orchestrator

Deterministic DAG workflow orchestrator.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m workflow_orchestrator version
python3 -m workflow_orchestrator help
python3 -m workflow_orchestrator plan <definition.json>
```

## Definition format

```json
{
  "tasks": [
    {"id": "task-a"},
    {"id": "task-b", "depends_on": ["task-a"]}
  ]
}
```

The top-level object contains only a `tasks` array. Each task has a
non-empty string `id` and an optional string array `depends_on` (defaults to
`[]`). `plan` validates dependencies and prints deterministic execution
levels without executing anything:

```json
{"levels":[["task-a"],["task-b"]]}
```

Tasks without dependencies form the first level; every other task is placed
one level below its deepest dependency. Tasks within a level are ordered by
Unicode code point. The output does not depend on input ordering.

### Exit codes

| Code | stderr |
| --- | --- |
| 1 | `cannot read definition: <path>` |
| 2 | `invalid json` |
| 2 | `invalid definition` |
| 2 | `duplicate task id: <id>` |
| 2 | `unknown dependency: <task-id> -> <dependency-id>` |
| 2 | `self dependency: <id>` |
| 2 | `cycle detected` |

When multiple graph problems exist, the reported one is chosen by Unicode
code point order of task id and dependency id, independent of input order.
On failure nothing is written to stdout.
