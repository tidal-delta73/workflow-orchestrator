"""Shared loading, validation, and normalization of workflow definitions.

This module is the single place that touches a definition's raw bytes.
Both the ``plan`` and ``schedule`` commands consume the same
:class:`ValidGraph`; neither re-parses JSON or re-judges graph errors.

The pipeline is split into four strict layers, each with one failure mode:

1. :func:`read_file`       -- file system access (bytes or :class:`CannotRead`)
2. :func:`decode_json`     -- JSON syntax (a JSON value or :class:`InvalidJson`)
3. :func:`parse_definition`-- definition structure (entries or
                              :class:`InvalidDefinition`)
4. :func:`build_graph`     -- graph semantics (a :class:`ValidGraph`, or one
                              deterministically chosen :class:`GraphError`)

:func:`load_graph` runs all four in order. No function in this module reads
from or writes to stdout/stderr; failures are carried by exceptions whose
``code`` and ``diagnostic`` the command entry points turn into exit codes and
diagnostic lines.
"""
import json
from collections import Counter, defaultdict
from types import MappingProxyType
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Failures: a closed hierarchy of deterministic results, no streams involved.
# ---------------------------------------------------------------------------

class GraphLoadError(Exception):
    """Base class for every definition-loading failure.

    Each failure fixes its own process exit ``code`` and the exact
    ``diagnostic`` text (without trailing newline) the commands must print.
    """

    code: int = 2

    @property
    def diagnostic(self) -> str:
        raise NotImplementedError


class CannotRead(GraphLoadError):
    """The definition file could not be read. Exit code 1."""

    code = 1

    def __init__(self, path: str):
        super().__init__(path)
        self.path = path

    @property
    def diagnostic(self) -> str:
        return f"cannot read definition: {self.path}"


class InvalidJson(GraphLoadError):
    """The definition file is not valid JSON (or not valid UTF-8)."""

    @property
    def diagnostic(self) -> str:
        return "invalid json"


class InvalidDefinition(GraphLoadError):
    """The JSON value does not have the required definition structure."""

    @property
    def diagnostic(self) -> str:
        return "invalid definition"


class GraphError(GraphLoadError):
    """The definition is structurally well formed but the graph is invalid."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    @property
    def diagnostic(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# The normalized result.
# ---------------------------------------------------------------------------

class ValidGraph(NamedTuple):
    """A read-only, validated, normalized view of a workflow definition.

    ``tasks`` is every task id in Unicode code point order.
    ``dependencies`` maps each task to its de-duplicated dependency ids.
    ``dependents`` is the reverse relation, likewise de-duplicated. Every
    contained mapping is wrapped to reject mutation and every relation is a
    tuple, so consumers cannot alter the shared graph.
    """

    tasks: tuple[str, ...]
    dependencies: MappingProxyType
    dependents: MappingProxyType


# ---------------------------------------------------------------------------
# Layer 1: file system.
# ---------------------------------------------------------------------------

def read_file(path: str) -> bytes:
    """Read the raw definition bytes, or raise :class:`CannotRead`."""
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:  # missing file, permission denied, directory, ...
        raise CannotRead(path) from exc


# ---------------------------------------------------------------------------
# Layer 2: JSON syntax.
# ---------------------------------------------------------------------------

def decode_json(raw: bytes):
    """Decode JSON bytes, or raise :class:`InvalidJson`.

    ``json.loads`` accepts ``bytes`` and reports invalid UTF-8 as a
    ``ValueError`` alongside syntax errors, so one clause covers both.
    """
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise InvalidJson from exc


# ---------------------------------------------------------------------------
# Layer 3: definition structure.
# ---------------------------------------------------------------------------

def parse_definition(document):
    """Validate the JSON value's shape; return ``[(task_id, [deps]), ...]``.

    Only structural rules live here: the top-level object, the task list,
    field presence/names, and value types. Graph semantics (duplicates,
    unknown/self dependencies, cycles) are deferred to :func:`build_graph`.
    """
    # Top level must be an object containing exactly one key: "tasks".
    if not isinstance(document, dict) or set(document) != {"tasks"}:
        raise InvalidDefinition
    raw_tasks = document["tasks"]
    if not isinstance(raw_tasks, list):
        raise InvalidDefinition

    entries = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            raise InvalidDefinition
        if not set(task) <= {"id", "depends_on"}:
            raise InvalidDefinition
        task_id = task.get("id")
        if not isinstance(task_id, str) or task_id == "":
            raise InvalidDefinition
        depends_on = task.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(dep, str) for dep in depends_on
        ):
            raise InvalidDefinition
        entries.append((task_id, list(depends_on)))
    return entries


# ---------------------------------------------------------------------------
# Layer 4: graph semantics.
# ---------------------------------------------------------------------------

def _reachable(adj, start):
    """Iterative reachability from ``start`` over ``adj`` (no recursion)."""
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def select_graph_error(entries):
    """Return the one graph-error message to report, or None if the DAG is valid.

    Every structural graph problem is keyed by the (task id, dependency id)
    pair it involves and candidates are compared by Unicode code point order,
    so the choice never depends on input ordering or hash iteration order.
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


def build_graph(entries) -> ValidGraph:
    """Validate graph semantics and normalize valid entries.

    Raises :class:`GraphError` with the single deterministically chosen
    diagnostic when the graph is invalid. Duplicate dependency entries count
    as one. Relations are sorted by Unicode code point order so the result
    depends only on graph content, never on listing or iteration order.
    """
    message = select_graph_error(entries)
    if message is not None:
        raise GraphError(message)

    tasks = tuple(sorted({task_id for task_id, _ in entries}))

    # Build the maps in the canonical (Unicode-sorted) task order so the
    # whole object is independent of how tasks were listed.
    unique_deps = {
        task_id: tuple(sorted(set(deps))) for task_id, deps in entries
    }
    dependencies = MappingProxyType(
        {task_id: unique_deps[task_id] for task_id in tasks}
    )

    dependents = {task_id: [] for task_id in tasks}
    for task_id in tasks:
        for dep in unique_deps[task_id]:
            dependents[dep].append(task_id)

    return ValidGraph(
        tasks=tasks,
        dependencies=dependencies,
        dependents=MappingProxyType(
            {task_id: tuple(dependents[task_id]) for task_id in tasks}
        ),
    )


# ---------------------------------------------------------------------------
# Full pipeline.
# ---------------------------------------------------------------------------

def load_graph(path: str) -> ValidGraph:
    """Read, decode, structurally validate, and semantically validate a file.

    Returns the shared read-only :class:`ValidGraph`. Raises one of the
    :class:`GraphLoadError` subclasses on failure; checks run strictly in the
    order read -> decode -> structure -> graph.
    """
    raw = read_file(path)
    document = decode_json(raw)
    entries = parse_definition(document)
    return build_graph(entries)
