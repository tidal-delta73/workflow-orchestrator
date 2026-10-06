# workflow-orchestrator

Deterministic DAG workflow orchestrator.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m workflow_orchestrator version
python3 -m workflow_orchestrator help
python3 -m workflow_orchestrator plan <definition.json>
python3 -m workflow_orchestrator schedule <definition.json> <max-parallel>
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

`schedule <definition.json> <max-parallel>` validates the same definition and
prints deterministic scheduling batches without executing anything. It starts
with the dependency-free tasks; each batch contains up to `max-parallel`
currently-ready tasks chosen by Unicode code point order, and tasks depending
on a batch only become candidates after that whole batch is complete.
Repeated dependency entries count once.

```json
{"batches":[["task-a"],["task-b"]]}
```

An empty task set prints `{"batches":[]}`. `max-parallel` is an unsigned
ASCII decimal integer without leading zeros in the range `1` through
`2147483647`; any other spelling (or an out-of-range value) fails with
`invalid parallelism` on stderr and exit code 2 before the definition file is
read.

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
