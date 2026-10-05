"""Static DAG validation and level planning.

The planner never executes tasks: it validates a parsed workflow definition
and assigns every task to the level immediately after its deepest dependency.
Every outcome is independent of the order in which tasks and dependencies
appear in the definition.
"""
from __future__ import annotations

import heapq
from typing import Any


class DefinitionError(Exception):
    """The workflow definition is structurally invalid or its graph is."""


def plan_levels(definition: Any) -> list[list[str]]:
    """Return the execution levels of a parsed workflow definition.

    The first level holds tasks without dependencies; every other task sits
    one level below its deepest dependency. Tasks within a level are ordered
    by Unicode code point.

    Raises DefinitionError (whose message is the exact CLI error line) for
    any invalid definition or graph-structure problem.
    """
    # --- structural validation -------------------------------------------------
    if not isinstance(definition, dict):
        raise DefinitionError("invalid definition")
    if set(definition) != {"tasks"}:
        raise DefinitionError("invalid definition")
    tasks = definition["tasks"]
    if not isinstance(tasks, list):
        raise DefinitionError("invalid definition")

    parsed: list[tuple[str, list[str]]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise DefinitionError("invalid definition")
        if set(task) - {"id", "depends_on"}:
            raise DefinitionError("invalid definition")
        task_id = task.get("id")
        if not isinstance(task_id, str) or task_id == "":
            raise DefinitionError("invalid definition")
        depends_on = task.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(dep, str) for dep in depends_on
        ):
            raise DefinitionError("invalid definition")
        parsed.append((task_id, depends_on))

    # Any structural problem above short-circuits before graph analysis.

    # --- graph-structure validation -------------------------------------------
    # Keep the first occurrence's dependency list per id; with duplicates the
    # duplicate key below always outranks any edge key on the same id, so the
    # chosen occurrence can never change the reported error.
    id_count: dict[str, int] = {}
    deps_by_id: dict[str, list[str]] = {}
    for task_id, deps in parsed:
        id_count[task_id] = id_count.get(task_id, 0) + 1
        deps_by_id.setdefault(task_id, deps)
    id_set = set(id_count)

    # Every graph problem gets a (task id, dependency id) ordering key; a
    # duplicate id carries an empty dependency slot and therefore precedes
    # any edge problem on the same task. Choosing the minimum key makes the
    # reported error depend only on Unicode code point order, never on input
    # order. Cycles have no identifiable edge and are reported last.
    problems: list[tuple[tuple[str, str], str]] = []
    for task_id in sorted(id_set):
        if id_count[task_id] > 1:
            problems.append(((task_id, ""), f"duplicate task id: {task_id}"))
        for dep in sorted(set(deps_by_id[task_id])):
            if dep == task_id:
                problems.append(
                    ((task_id, dep), f"self dependency: {task_id}")
                )
            elif dep not in id_set:
                problems.append(
                    ((task_id, dep),
                     f"unknown dependency: {task_id} -> {dep}")
                )

    if problems:
        raise DefinitionError(min(problems, key=lambda item: item[0])[1])

    edges: dict[str, set[str]] = {
        task_id: set(deps_by_id[task_id]) for task_id in id_set
    }

    # --- level assignment (Kahn, ties broken by code point) --------------------
    indegree: dict[str, int] = {task_id: len(deps) for task_id, deps in edges.items()}
    dependents: dict[str, list[str]] = {task_id: [] for task_id in id_set}
    for task_id, deps in edges.items():
        for dep in deps:
            dependents[dep].append(task_id)

    level: dict[str, int] = {}
    ready = [task_id for task_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    while ready:
        current = heapq.heappop(ready)
        current_level = level.get(current, 0)
        level[current] = current_level
        for dependent in dependents[current]:
            candidate = current_level + 1
            if candidate > level.get(dependent, 0):
                level[dependent] = candidate
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(level) != len(id_set):
        # All edges point at existing nodes and self loops were rejected, so
        # any remaining nodes belong to a cycle of two or more tasks.
        raise DefinitionError("cycle detected")

    if not id_set:
        return []

    ordered = sorted(id_set, key=lambda task_id: (level[task_id], task_id))
    levels: list[list[str]] = []
    for task_id in ordered:
        task_level = level[task_id]
        if task_level == len(levels):
            levels.append([])
        levels[task_level].append(task_id)
    return levels
