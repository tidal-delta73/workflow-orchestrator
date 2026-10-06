"""Bounded-parallel batch scheduling for workflow definitions.

The scheduler only computes batches; it never executes tasks. Loading,
structural validation, and graph semantics belong to the shared
:mod:`workflow_orchestrator.graph` core, so ``schedule`` reports exactly
the same diagnostics as ``plan`` without reaching into any of plan's
private functions. Scheduling starts from every task with no dependencies;
each batch takes up to ``max_parallel`` of the currently ready tasks in
Unicode code point order, and a successor becomes ready only once the
whole batch containing its dependencies has been scheduled. Duplicate
dependency entries count as one.
"""
import json

from . import graph as graph_module

MAX_PARALLEL_LIMIT = 2147483647


def parse_parallelism(text):
    """Parse the max-parallel argument, or return None when it is invalid.

    Only an ASCII decimal integer between 1 and 2147483647 without a sign
    or leading zeros is accepted.
    """
    if not text or not all("0" <= ch <= "9" for ch in text):
        return None
    if len(text) > 1 and text[0] == "0":
        return None
    value = int(text)
    if not 1 <= value <= MAX_PARALLEL_LIMIT:
        return None
    return value


def _build_batches(g: graph_module.Graph, max_parallel):
    remaining = {task_id: len(g.dependencies[task_id]) for task_id in g.tasks}

    ready = [task_id for task_id in g.tasks if remaining[task_id] == 0]
    batches = []
    while ready:
        batch = ready[:max_parallel]
        batches.append(batch)
        ready = ready[max_parallel:]
        for task_id in batch:
            for nxt in g.dependents[task_id]:
                remaining[nxt] -= 1
                if remaining[nxt] == 0:
                    ready.append(nxt)
        ready.sort()  # Unicode code point order
    return batches


def _render(batches) -> str:
    return json.dumps({"batches": batches}, ensure_ascii=False, separators=(",", ":"))


def run(path: str, max_parallel: int, stdout, stderr) -> int:
    """Execute the schedule command, writing to the provided text streams."""
    try:
        g = graph_module.load_graph(path)
    except graph_module.LoadError as error:
        stderr.write(error.message + "\n")
        return error.exit_code
    stdout.write(_render(_build_batches(g, max_parallel)) + "\n")
    return 0
