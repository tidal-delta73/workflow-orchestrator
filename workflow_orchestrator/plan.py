"""Static DAG validation and level planning for workflow definitions.

The planner only analyzes dependencies; it never executes tasks. Loading,
structural validation, and graph semantics belong to the shared
:mod:`workflow_orchestrator.graph` core. Given the one validated,
normalized :class:`~workflow_orchestrator.graph.Graph` it produces, this
module computes deterministic execution levels; it never re-parses raw
JSON or re-checks graph errors.
"""
import json

from . import graph as graph_module


def _build_levels(g: graph_module.Graph):
    """Group tasks into levels by deepest dependency.

    A task without dependencies is at level 0; every other task is one
    level below its deepest dependency. Traversal is iterative so deep
    chains never hit the recursion limit.
    """
    depth = {}
    for root in g.tasks:  # already in Unicode code point order
        if root in depth:
            continue
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                deps = g.dependencies[node]
                depth[node] = 0 if not deps else 1 + max(depth[d] for d in deps)
            elif node not in depth:
                stack.append((node, True))
                for dep in g.dependencies[node]:
                    if dep not in depth:
                        stack.append((dep, False))

    levels = []
    for task_id in g.tasks:
        level = depth[task_id]
        while len(levels) <= level:
            levels.append([])
        levels[level].append(task_id)  # g.tasks order is code point order
    return levels


def _render(levels) -> str:
    return json.dumps({"levels": levels}, ensure_ascii=False, separators=(",", ":"))


def run(path: str, stdout, stderr) -> int:
    """Execute the plan command, writing to the provided text streams."""
    try:
        g = graph_module.load_graph(path)
    except graph_module.LoadError as error:
        stderr.write(error.message + "\n")
        return error.exit_code
    stdout.write(_render(_build_levels(g)) + "\n")
    return 0
