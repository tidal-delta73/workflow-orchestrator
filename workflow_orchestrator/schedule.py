"""Bounded-parallel batch scheduling for workflow definitions.

The scheduler only computes batches; it never executes tasks. Loading,
JSON decoding, structural validation, graph validation, and normalization
are shared with ``plan`` through the :mod:`workflow_orchestrator.graph`
core, so both commands consume the same read-only
:class:`~workflow_orchestrator.graph.ValidGraph` and report identical
diagnostics. Scheduling starts from every task with no dependencies; each
batch takes up to ``max_parallel`` of the currently ready tasks in Unicode
code point order, and a successor becomes ready only once the whole batch
containing its dependencies has been scheduled. Duplicate dependency
entries count as one.
"""
import json

from .graph import GraphLoadError, load_graph

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


def build_batches(graph, max_parallel):
    """Compute bounded-parallel batches from a validated graph.

    Uses the graph's de-duplicated forward and reverse dependency relations,
    so duplicate dependency entries count once and nothing here re-parses
    the definition.
    """
    remaining = {
        task_id: len(graph.dependencies[task_id]) for task_id in graph.tasks
    }
    waiting = graph.dependents

    ready = [task_id for task_id in graph.tasks if remaining[task_id] == 0]
    batches = []
    while ready:
        batch = ready[:max_parallel]
        batches.append(batch)
        ready = ready[max_parallel:]
        for task_id in batch:
            for nxt in waiting[task_id]:
                remaining[nxt] -= 1
                if remaining[nxt] == 0:
                    ready.append(nxt)
        ready.sort()  # Unicode code point order
    return batches


def render(batches) -> str:
    return json.dumps({"batches": batches}, ensure_ascii=False, separators=(",", ":"))


def run(path: str, max_parallel: int, stdout, stderr) -> int:
    """Execute the schedule command, writing to the provided text streams."""
    try:
        graph = load_graph(path)
    except GraphLoadError as error:
        stderr.write(error.diagnostic + "\n")
        return error.code
    stdout.write(render(build_batches(graph, max_parallel)) + "\n")
    return 0
