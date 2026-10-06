"""Shared loading and validation pipeline for workflow definition graphs.

This module is the single place that understands how a definition file
becomes a usable graph. It is organized as four stages with explicit
boundaries:

1. :func:`read_bytes`  -- file system access only;
2. :func:`decode_document` -- JSON decoding of those bytes only;
3. :func:`parse_definition` -- structural validation of the JSON value;
4. :func:`build_graph` -- graph semantics (duplicates, dependencies, cycles)
   and normalization.

:func:`load_graph` runs the four stages in order. The module never writes
to stdout or stderr: every failure is a :class:`LoadError` carrying the
exit code and diagnostic text, and the command entry points decide how to
present it. Successful loading returns one read-only, fully validated
:class:`Graph`; downstream planning logic must never re-parse raw JSON or
re-check graph errors.

All graph analysis is iterative, so arbitrarily deep DAGs do not depend on
the Python recursion limit.
"""
import json
from collections import Counter, defaultdict
from types import MappingProxyType

CANNOT_READ_CODE = 1
INVALID_CODE = 2


class LoadError(Exception):
    """A determinate loading failure: the exit code and diagnostic text.

    Core code raises this instead of touching an output stream; the command
    entry point translates it into an exit code and one stderr line.
    """

    def __init__(self, exit_code: int, message: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.message = message


# ---------------------------------------------------------------------------
# Stage 1: file system access.
# ---------------------------------------------------------------------------


def read_bytes(path: str) -> bytes:
    """Read the raw bytes of *path*.

    Raises :class:`LoadError` with code 1 when the file cannot be read
    (missing file, permission denied, a directory, ...).
    """
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise LoadError(CANNOT_READ_CODE, f"cannot read definition: {path}") from exc


# ---------------------------------------------------------------------------
# Stage 2: JSON decoding.
# ---------------------------------------------------------------------------


def decode_document(raw: bytes):
    """Decode raw bytes as JSON; no structural assumptions are made here."""
    try:
        return json.loads(raw)
    except ValueError as exc:  # json.JSONDecodeError and UnicodeDecodeError
        raise LoadError(INVALID_CODE, "invalid json") from exc


# ---------------------------------------------------------------------------
# Stage 3: definition structure.
# ---------------------------------------------------------------------------


def parse_definition(document):
    """Validate the JSON value's structure and extract task entries.

    The top level must be an object containing exactly the key ``tasks``,
    whose value is a list of task objects. Each task object may contain only
    ``id`` (a non-empty string) and ``depends_on`` (a list of strings,
    defaulting to empty). Returns ``[(task_id, [dep, ...]), ...]`` in
    document order; entries are not yet semantically checked or de-duplicated.
    """
    if not isinstance(document, dict) or set(document) != {"tasks"}:
        raise LoadError(INVALID_CODE, "invalid definition")
    raw_tasks = document["tasks"]
    if not isinstance(raw_tasks, list):
        raise LoadError(INVALID_CODE, "invalid definition")

    entries = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            raise LoadError(INVALID_CODE, "invalid definition")
        if not set(task) <= {"id", "depends_on"}:
            raise LoadError(INVALID_CODE, "invalid definition")
        task_id = task.get("id")
        if not isinstance(task_id, str) or task_id == "":
            raise LoadError(INVALID_CODE, "invalid definition")
        depends_on = task.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(dep, str) for dep in depends_on
        ):
            raise LoadError(INVALID_CODE, "invalid definition")
        entries.append((task_id, list(depends_on)))
    return entries


# ---------------------------------------------------------------------------
# Stage 4: graph semantics and normalization.
# ---------------------------------------------------------------------------


def _reachable(adj, start):
    """Iterative reachability set from *start* following *adj*."""
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
    """Return the single error message to report, or None for a valid DAG.

    Every structural graph problem is keyed by the (task id, dependency id)
    pair it involves and candidates are compared by Unicode code point order,
    so the choice never depends on input or set iteration order.
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


class Graph:
    """A read-only, validated, normalized workflow DAG.

    Attributes:
        tasks: every task id in Unicode code point order;
        dependencies: task id -> tuple of its unique dependency ids, in
            Unicode code point order (duplicate listings count as one);
        dependents: task id -> tuple of task ids that depend on it, in
            Unicode code point order.

    The graph is immutable: attributes cannot be reassigned, the mapping
    views cannot be changed, and every value is a tuple. Planning logic
    only reads it and never re-parses the original JSON or re-checks
    graph errors.
    """

    __slots__ = ("tasks", "dependencies", "dependents")

    def __init__(self, entries):
        unique_deps = {}
        for task_id, deps in entries:
            unique_deps[task_id] = set(deps)

        dependents = {task_id: set() for task_id in unique_deps}
        for task_id, deps in unique_deps.items():
            for dep in deps:
                dependents[dep].add(task_id)

        object.__setattr__(self, "tasks", tuple(sorted(unique_deps)))
        object.__setattr__(
            self,
            "dependencies",
            MappingProxyType(
                {
                    task_id: tuple(sorted(deps))
                    for task_id, deps in unique_deps.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "dependents",
            MappingProxyType(
                {
                    task_id: tuple(sorted(successors))
                    for task_id, successors in dependents.items()
                }
            ),
        )

    def __setattr__(self, name, value):
        raise AttributeError(f"Graph is read-only: cannot set {name!r}")

    def __delattr__(self, name):
        raise AttributeError(f"Graph is read-only: cannot delete {name!r}")

    def __repr__(self):
        return f"Graph(tasks={self.tasks!r})"


def build_graph(entries) -> Graph:
    """Check graph semantics of parsed *entries* and normalize them.

    Raises :class:`LoadError` with code 2 for duplicate task ids, unknown
    dependencies, self dependencies, and cycles. On success returns the
    single normalized :class:`Graph` shared by every command.
    """
    error = _graph_error(entries)
    if error is not None:
        raise LoadError(INVALID_CODE, error)
    return Graph(entries)


# ---------------------------------------------------------------------------
# Whole pipeline.
# ---------------------------------------------------------------------------


def load_graph(path: str) -> Graph:
    """Run all four stages and return the validated, normalized graph.

    The first failure encountered becomes a :class:`LoadError`; nothing is
    written to any stream.
    """
    raw = read_bytes(path)
    document = decode_document(raw)
    entries = parse_definition(document)
    return build_graph(entries)
