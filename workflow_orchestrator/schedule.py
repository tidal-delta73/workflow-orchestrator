"""Deterministic batch scheduling with a fixed parallelism ceiling.

The scheduler only analyzes dependencies; it never executes tasks. It reuses
the planner's document reading, structural validation, and graph-error
selection, then partitions the tasks into batches: every batch holds up to
``max_parallel`` currently-ready tasks chosen by Unicode code point order,
and successors only become candidates once a whole batch is complete.
"""
import heapq
import json

from . import plan as plan_module

# Inclusive upper bound for an accepted parallelism value.
MAX_PARALLEL = 2147483647


class _InvalidParallelism(Exception):
    """The parallelism text is not an accepted ASCII decimal integer."""


def _parse_parallelism(text):
    # ASCII digits only: no sign, no whitespace, no underscores, no
    # Unicode decimal digits, and no leading zero (except "0" itself,
    # which the range check rejects).
    if not text or not text.isascii() or not text.isdigit():
        raise _InvalidParallelism
    if len(text) > 1 and text[0] == "0":
        raise _InvalidParallelism
    value = int(text)
    if not 1 <= value <= MAX_PARALLEL:
        raise _InvalidParallelism
    return value


def _build_batches(entries, max_parallel):
    # Distinct dependency sets collapse repeated depends_on entries, so a
    # repeated edge counts once toward readiness.
    deps_by_task = {task_id: set(deps) for task_id, deps in entries}
    dependents = {task_id: [] for task_id in deps_by_task}
    for task_id, deps in deps_by_task.items():
        for dep in deps:
            dependents[dep].append(task_id)

    # A min-heap of currently ready ids fixes Unicode code point order
    # without ever depending on set or dict iteration order.
    ready = [
        task_id for task_id, deps in deps_by_task.items() if not deps
    ]
    heapq.heapify(ready)
    batches = []
    while ready:
        # Take the whole batch from what is ready *now*; successors freed
        # by this batch are pushed only afterwards and join the next one.
        batch = [
            heapq.heappop(ready)
            for _ in range(min(max_parallel, len(ready)))
        ]
        batches.append(batch)
        newly_ready = []
        for task_id in batch:
            for successor in dependents[task_id]:
                deps_by_task[successor].discard(task_id)
                if not deps_by_task[successor]:
                    newly_ready.append(successor)
        for task_id in newly_ready:
            heapq.heappush(ready, task_id)
    return batches


def _render(batches) -> str:
    return json.dumps(
        {"batches": batches}, ensure_ascii=False, separators=(",", ":")
    )


def run(path: str, parallelism_text: str, stdout, stderr) -> int:
    """Execute the schedule command, writing to the provided text streams."""
    try:
        max_parallel = _parse_parallelism(parallelism_text)
    except _InvalidParallelism:
        # Reported before the definition file is touched.
        stderr.write("invalid parallelism\n")
        return 2

    try:
        document = plan_module._read_document(path)
    except plan_module._CannotRead:
        stderr.write(f"cannot read definition: {path}\n")
        return 1
    except plan_module._InvalidJson:
        stderr.write("invalid json\n")
        return 2

    try:
        entries = plan_module._parse_tasks(document)
    except plan_module._InvalidDefinition:
        stderr.write("invalid definition\n")
        return 2

    error = plan_module._graph_error(entries)
    if error is not None:
        stderr.write(error + "\n")
        return 2

    stdout.write(_render(_build_batches(entries, max_parallel)) + "\n")
    return 0
