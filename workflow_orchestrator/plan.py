"""Static DAG validation and level planning for workflow definitions.

The planner only analyzes dependencies; it never executes tasks. Given a
definition document it either produces deterministic execution levels or a
single, deterministically chosen error message.
"""
import json
from collections import Counter, defaultdict


class _CannotRead(Exception):
    """The definition file could not be read."""


class _InvalidJson(Exception):
    """The definition file is not valid JSON."""


class _InvalidDefinition(Exception):
    """The JSON value does not have the required definition structure."""


def _read_document(path: str):
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:  # missing file, permission denied, directory, ...
        raise _CannotRead(path) from exc
    try:
        return json.loads(raw)
    except ValueError as exc:  # json.JSONDecodeError and UnicodeDecodeError
        raise _InvalidJson from exc


def _parse_tasks(document):
    # Top level must be an object containing exactly one key: "tasks".
    if not isinstance(document, dict) or set(document) != {"tasks"}:
        raise _InvalidDefinition
    raw_tasks = document["tasks"]
    if not isinstance(raw_tasks, list):
        raise _InvalidDefinition

    entries = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            raise _InvalidDefinition
        if not set(task) <= {"id", "depends_on"}:
            raise _InvalidDefinition
        task_id = task.get("id")
        if not isinstance(task_id, str) or task_id == "":
            raise _InvalidDefinition
        depends_on = task.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(dep, str) for dep in depends_on
        ):
            raise _InvalidDefinition
        entries.append((task_id, list(depends_on)))
    return entries


def _reachable(adj, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _graph_error(entries):
    """Return the single error message to report, or None when the DAG is valid.

    Every structural graph problem is keyed by the (task id, dependency id)
    pair it involves and candidates are compared by Unicode code point order,
    so the choice never depends on input ordering.
    """
    counts = Counter(task_id for task_id, _ in entries)
    known = set(counts)

    # (first id, second id, kind rank, message); kind rank only breaks exact
    # pair ties deterministically.
    candidates = []
    for task_id in counts:
        if counts[task_id] > 1:
            candidates.append((task_id, "", 0, f"duplicate task id: {task_id}"))

    adj = defaultdict(set)
    for task_id, deps in entries:
        for dep in set(deps):
            if dep == task_id:
                candidates.append(
                    (task_id, task_id, 1, f"self dependency: {task_id}")
                )
            elif dep not in known:
                candidates.append(
                    (task_id, dep, 2, f"unknown dependency: {task_id} -> {dep}")
                )
            else:
                adj[task_id].add(dep)

    # An edge u -> v (u != v) participates in a cycle iff v can reach u.
    reach_cache = {}
    cycle_key = None
    for task_id, deps in adj.items():
        for dep in deps:
            if dep not in reach_cache:
                reach_cache[dep] = _reachable(adj, dep)
            if task_id in reach_cache[dep]:
                key = (task_id, dep)
                if cycle_key is None or key < cycle_key:
                    cycle_key = key
    if cycle_key is not None:
        candidates.append(
            (cycle_key[0], cycle_key[1], 3, "cycle detected")
        )

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[3]


def _build_levels(entries):
    deps_by_task = {}
    for task_id, deps in entries:
        deps_by_task[task_id] = set(deps)

    depth = {}
    for root in deps_by_task:
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


def _render(levels) -> str:
    return json.dumps({"levels": levels}, ensure_ascii=False, separators=(",", ":"))


def run(path: str, stdout, stderr) -> int:
    """Execute the plan command, writing to the provided text streams."""
    try:
        document = _read_document(path)
    except _CannotRead:
        stderr.write(f"cannot read definition: {path}\n")
        return 1
    except _InvalidJson:
        stderr.write("invalid json\n")
        return 2

    try:
        entries = _parse_tasks(document)
    except _InvalidDefinition:
        stderr.write("invalid definition\n")
        return 2

    error = _graph_error(entries)
    if error is not None:
        stderr.write(error + "\n")
        return 2

    stdout.write(_render(_build_levels(entries)) + "\n")
    return 0
