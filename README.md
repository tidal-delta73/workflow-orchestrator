# workflow-orchestrator

Deterministic DAG workflow orchestrator.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m workflow_orchestrator version
python3 -m workflow_orchestrator help
python3 -m workflow_orchestrator plan <definition.json>
```

### `plan`

Validates a workflow definition and prints its static execution levels as a
single JSON line, e.g. `{"levels":[["task-a"],["task-b"]]}`. It only analyzes
dependencies; no task is ever executed.

A definition is a JSON object with exactly one key, `tasks`, holding an array
of task objects. Each task requires a non-empty string `id` and may list a
string array `depends_on` (empty when omitted). Tasks without dependencies go
on the first level; every other task goes one level after its deepest
dependency, with tasks on the same level ordered by Unicode code point. The
output does not depend on input ordering and ends with a newline.

Exit codes:

- `0` — levels written to standard output;
- `1` — `cannot read definition: <path>` on standard error;
- `2` — `invalid json`, `invalid definition`, `duplicate task id: <id>`,
  `unknown dependency: <task-id> -> <dependency-id>`, `self dependency: <id>`,
  `cycle detected`, or the usage text, all on standard error with no standard
  output.

