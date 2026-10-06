"""Static DAG validation and level planning for workflow definitions.

The planner only analyzes dependencies; it never executes tasks. Loading,
JSON decoding, structural validation, graph validation, and normalization
live in the shared :mod:`workflow_orchestrator.graph` core; this module only
turns a validated :class:`~workflow_orchestrator.graph.ValidGraph` into
deterministic execution levels and translates a core failure into the exit
code and diagnostic ``plan`` has always reported.
"""
import json

from .graph import GraphLoadError, load_graph


def build_levels(graph) -> list[list[str]]:
    """Group tasks into levels by deepest dependency.

    A task with no dependencies sits on level 0; every other task is placed
    one level below its deepest dependency. Tasks within a level are ordered
    by Unicode code point. The walk is iterative, so arbitrarily deep DAGs
    do not depend on the Python recursion limit.
    """
    deps_by_task = graph.dependencies

    depth = {}
    for root in graph.tasks:
        if root in depth:
            continue
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                deps = deps_by_task[node]
                depth[node] = 0 if not deps else 1 + max(depth[d] for d in deps)
            elif node not in depth:
                stack.append((node, True))
                for dep in deps_by_task[node]:
                    if dep not in depth:
                        stack.append((dep, False))

    levels = []
    for task_id, level in depth.items():
        while len(levels) <= level:
            levels.append([])
        levels[level].append(task_id)
    for level in levels:
        level.sort()  # Unicode code point order
    return levels


def render(levels) -> str:
    return json.dumps({"levels": levels}, ensure_ascii=False, separators=(",", ":"))


def run(path: str, stdout, stderr) -> int:
    """Execute the plan command, writing to the provided text streams."""
    try:
        graph = load_graph(path)
    except GraphLoadError as error:
        stderr.write(error.diagnostic + "\n")
        return error.code
    stdout.write(render(build_levels(graph)) + "\n")
    return 0
