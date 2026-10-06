"""Seeded generative cases and an independent reference oracle.

This module supports the generative regression suite in
``test_generative.py``. It provides:

* :class:`Rng` -- deterministic randomness derived only from
  ``random.Random.random()``, the one generator method CPython guarantees
  to keep stable for a given seed. Seeds are strings (never tuples or
  other objects, whose ``hash()`` varies with ``PYTHONHASHSEED``), so a
  fixed seed reproduces the exact same case sequence on any machine,
  regardless of hash seed, locale, or filesystem ordering.
* case generators -- :func:`valid_cases` builds seeded DAGs (empty graph,
  single task, multiple roots, wide graph, long chain, diamond join,
  independent subgraphs, random DAGs) plus equivalent JSON documents per
  graph (permuted task order, permuted ``depends_on`` order, duplicated
  dependency entries); :func:`invalid_cases` builds seeded duplicate-task,
  unknown-dependency, self-dependency, cycle, and mixed-problem documents.
* a reference oracle -- :func:`reference_levels`, :func:`reference_batches`
  and :func:`reference_error` implement the documented public contract
  from scratch. They deliberately import nothing from
  ``workflow_orchestrator`` so the tests compare the product against an
  independent statement of the semantics.
* invariant checkers -- :func:`check_levels` and :func:`check_batches`
  restate the output contract as standalone predicates over parsed stdout,
  so a bug shared by product and oracle shape cannot pass silently.

Nothing here touches the network, the clock, or the environment.
"""
import random
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Deterministic randomness.
# ---------------------------------------------------------------------------

class Rng:
    """A small deterministic RNG built only on ``Random.random()``.

    ``random()`` is the sole method whose sequence is guaranteed stable
    across CPython versions for a given seed, so every draw here is routed
    through it. Seeds must be ``int``/``str``/``bytes``; seeding with a
    tuple would fall back to ``hash()`` and break reproducibility under a
    randomized ``PYTHONHASHSEED``.
    """

    def __init__(self, seed):
        self._random = random.Random(seed)

    def below(self, n):
        """A deterministic integer in ``range(n)`` for ``n >= 1``."""
        return min(n - 1, int(self._random.random() * n))

    def chance(self, p):
        return self._random.random() < p

    def choice(self, items):
        return items[self.below(len(items))]

    def shuffle(self, items):
        """A shuffled copy of ``items`` (Fisher-Yates over ``below``)."""
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items

    def sample(self, items, k):
        return self.shuffle(items)[:k]


# ---------------------------------------------------------------------------
# Reference oracle: the documented semantics, restated independently.
# ---------------------------------------------------------------------------

def reference_levels(deps):
    """Levels by deepest dependency; tasks within a level Unicode-sorted.

    ``deps`` maps each task id to an iterable of dependency ids.
    """
    memo = {}

    def depth(node):
        if node not in memo:
            own = deps[node]
            memo[node] = 0 if not own else 1 + max(depth(d) for d in own)
        return memo[node]

    levels = []
    for node in deps:
        level = depth(node)
        while len(levels) <= level:
            levels.append([])
        levels[level].append(node)
    return [sorted(level) for level in levels]


def reference_batches(deps, max_parallel):
    """Greedy bounded-parallel batches per the documented schedule rule."""
    remaining = {task: len(set(own)) for task, own in deps.items()}
    dependents = {task: [] for task in deps}
    for task, own in deps.items():
        for dep in set(own):
            dependents[dep].append(task)
    ready = sorted(task for task in deps if remaining[task] == 0)
    batches = []
    while ready:
        batch = ready[:max_parallel]
        batches.append(batch)
        ready = ready[max_parallel:]
        for task in batch:
            for nxt in dependents[task]:
                remaining[nxt] -= 1
                if remaining[nxt] == 0:
                    ready.append(nxt)
        ready.sort()
    return batches


def _reaches(adjacency, start, target):
    """True when ``target`` is reachable from ``start`` over ``adjacency``."""
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in adjacency.get(node, ()):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def reference_error(entries):
    """The one deterministic graph-error message, or None for a valid DAG.

    ``entries`` is a sequence of ``(task_id, deps)`` pairs as listed in the
    document (duplicate task ids and duplicate dependency entries kept).
    Candidates are keyed by (task id, dependency id, kind rank) and the
    smallest key in Unicode code point order wins, per the README contract.
    """
    counts = {}
    for task_id, _ in entries:
        counts[task_id] = counts.get(task_id, 0) + 1
    known = set(counts)

    best = None

    def offer(key, message):
        nonlocal best
        if best is None or key < best[0]:
            best = (key, message)

    for task_id, count in counts.items():
        if count > 1:
            offer((task_id, "", 0), "duplicate task id: %s" % task_id)

    adjacency = {}
    for task_id, deps in entries:
        for dep in set(deps):
            if dep == task_id:
                offer((task_id, task_id, 1), "self dependency: %s" % task_id)
            elif dep not in known:
                offer(
                    (task_id, dep, 2),
                    "unknown dependency: %s -> %s" % (task_id, dep),
                )
            else:
                adjacency.setdefault(task_id, set()).add(dep)

    # An edge u -> v (u != v) lies on a cycle iff v can reach u.
    cycle_edges = [
        (task_id, dep)
        for task_id, deps in adjacency.items()
        for dep in deps
        if _reaches(adjacency, dep, task_id)
    ]
    if cycle_edges:
        offer(min(cycle_edges) + (3,), "cycle detected")

    return None if best is None else best[1]


# ---------------------------------------------------------------------------
# Output-contract invariant checkers (return a list of violation strings).
# ---------------------------------------------------------------------------

def check_levels(deps, levels):
    """Violations of the plan contract in a parsed ``levels`` output."""
    problems = []
    flat = [task for level in levels for task in level]
    if sorted(flat) != sorted(deps):
        problems.append("levels do not cover each task exactly once: %r" % flat)
        return problems
    place = {task: i for i, level in enumerate(levels) for task in level}
    for i, level in enumerate(levels):
        if list(level) != sorted(level):
            problems.append("level %d not in Unicode order: %r" % (i, level))
    for task, own in deps.items():
        want = 0 if not own else 1 + max(place[d] for d in own)
        if place[task] != want:
            problems.append(
                "task %r on level %d, deepest dependency wants %d"
                % (task, place[task], want)
            )
    return problems


def check_batches(deps, batches, max_parallel):
    """Violations of the schedule contract in a parsed ``batches`` output."""
    problems = []
    flat = [task for batch in batches for task in batch]
    if sorted(flat) != sorted(deps):
        problems.append("batches do not cover each task exactly once: %r" % flat)
        return problems
    place = {task: i for i, batch in enumerate(batches) for task in batch}
    for task, own in deps.items():
        for dep in own:
            if place[dep] >= place[task]:
                problems.append(
                    "dependency %r (batch %d) not strictly before %r (batch %d)"
                    % (dep, place[dep], task, place[task])
                )
    done = set()
    for i, batch in enumerate(batches):
        if not batch:
            problems.append("batch %d is empty" % i)
        if len(batch) > max_parallel:
            problems.append(
                "batch %d holds %d tasks, max-parallel is %d"
                % (i, len(batch), max_parallel)
            )
        ready = sorted(
            task
            for task in deps
            if task not in done and all(d in done for d in deps[task])
        )
        if list(batch) != ready[:max_parallel]:
            problems.append(
                "batch %d is %r, but the ready tasks at batch start were %r"
                % (i, batch, ready[:max_parallel])
            )
        done.update(batch)
    return problems


# ---------------------------------------------------------------------------
# Valid-case generation.
# ---------------------------------------------------------------------------

DOCUMENT_VARIANTS = 3


class ValidCase(NamedTuple):
    name: str
    deps: dict        # task id -> tuple of dependency ids (canonical)
    parallelisms: tuple
    documents: tuple  # equivalent JSON documents


def _canonical(deps):
    return {
        task: tuple(sorted(set(own))) for task, own in sorted(deps.items())
    }


def _frontier_width(deps):
    """The largest number of tasks ready at any single batch start."""
    done = set()
    width = 0
    while len(done) < len(deps):
        ready = [
            task
            for task in deps
            if task not in done and all(d in done for d in deps[task])
        ]
        if not ready:
            break
        width = max(width, len(ready))
        done.update(ready)
    return width


def _shapes(rng):
    """Hand-shaped DAG templates; sizes drawn from ``rng`` where useful."""
    shapes = [("empty", {}), ("single", {"only": ()})]

    shapes.append((
        "multi-root",
        {
            "root-a": (),
            "root-b": (),
            "root-c": (),
            "join-left": ("root-a", "root-b"),
            "leaf-c": ("root-c",),
            "join-all": ("join-left", "leaf-c"),
        },
    ))

    # Wide: many roots, plus ids whose code points straddle the ASCII
    # boundary ("Zulu" < lowercase, "éclair" beyond ASCII) to exercise
    # Unicode ordering and non-ASCII output bytes.
    wide = {"n%02d" % i: () for i in range(10 + rng.below(4))}
    wide["Zulu"] = ()
    wide["éclair"] = ()
    shapes.append(("wide", wide))

    chain = {}
    prev = None
    for i in range(12 + rng.below(8)):
        task = "c%02d" % i
        chain[task] = () if prev is None else (prev,)
        prev = task
    shapes.append(("long-chain", chain))

    shapes.append((
        "diamond",
        {
            "top": (),
            "left": ("top",),
            "right": ("top",),
            "bottom": ("left", "right"),
        },
    ))

    shapes.append((
        "independent-subgraphs",
        {
            "a0": (),
            "a1": ("a0",),
            "a2": ("a1",),
            "b0": (),
            "b1": ("b0",),
            "d0": (),
            "d1": ("d0",),
            "d2": ("d0",),
            "d3": ("d1", "d2"),
        },
    ))
    return shapes


def _random_dag(rng, label):
    """A random DAG: each task depends on a random subset of earlier tasks."""
    n = 5 + rng.below(9)
    ids = ["%s-%02d" % (label, i) for i in range(n)]
    order = rng.shuffle(ids)
    deps = {task: [] for task in ids}
    for pos, task in enumerate(order):
        earlier = order[:pos]
        if earlier:
            deps[task] = rng.sample(earlier, rng.below(min(3, len(earlier)) + 1))
    return deps


def _equivalent_documents(deps, rng, count):
    """JSON documents equivalent to ``deps``: permuted orders, dup deps."""
    documents = []
    for _ in range(count):
        tasks = []
        for task in rng.shuffle(list(deps)):
            own = rng.shuffle(deps[task])
            if own and rng.chance(0.5):
                # Repeat one dependency entry; duplicates are semantically
                # irrelevant and must not change the output.
                own = own + [own[rng.below(len(own))]]
            entry = {"id": task}
            if own or rng.chance(0.5):
                entry["depends_on"] = own
            tasks.append(entry)
        documents.append({"tasks": tasks})
    return documents


def valid_cases(seed):
    """The deterministic sequence of valid-DAG cases for ``seed``."""
    rng = Rng("valid:%s" % seed)
    shapes = _shapes(rng)
    for index in range(3):
        shapes.append(("random-%d" % index, _random_dag(rng, "r%d" % index)))

    cases = []
    for name, deps in shapes:
        deps = _canonical(deps)
        # Boundary parallelisms: 1, exactly the largest ready frontier,
        # and more than the total task count (deduplicated, order kept).
        parallelisms = tuple(
            dict.fromkeys((1, max(1, _frontier_width(deps)), len(deps) + 1))
        )
        doc_rng = Rng("valid-docs:%s:%s" % (seed, name))
        documents = tuple(_equivalent_documents(deps, doc_rng, DOCUMENT_VARIANTS))
        cases.append(ValidCase(name, deps, parallelisms, documents))
    return tuple(cases)


# ---------------------------------------------------------------------------
# Invalid-case generation.
# ---------------------------------------------------------------------------

INVALID_DOCUMENT_VARIANTS = 2


class InvalidCase(NamedTuple):
    name: str
    message: str      # the expected diagnostic, without trailing newline
    documents: tuple  # equivalent JSON documents (all must fail identically)


def _duplicate_entries(rng):
    entries = [["t0", []], ["t1", ["t0"]], ["t2", ["t0"]]]
    victim = rng.choice([entry[0] for entry in entries])
    entries.append([victim, rng.sample(["t0", "t1"], rng.below(2))])
    return entries


def _unknown_entries(rng):
    ghosts = rng.sample(["ghost-a", "ghost-m", "ghost-z"], 2)
    return [
        ["t0", []],
        ["t1", ["t0", ghosts[0]]],
        ["t2", ["t0", ghosts[1]]],
    ]


def _self_entries(rng):
    entries = [["t0", []], ["t1", ["t0"]], ["t2", ["t0"]]]
    victim = rng.choice(entries)
    victim[1].append(victim[0])
    return entries


def _cycle_entries(rng):
    n = 3 + rng.below(3)
    order = rng.shuffle(["ring-%d" % i for i in range(n)])
    entries = [[order[i], [order[(i + 1) % n]]] for i in range(n)]
    entries.append(["solo", []])
    return entries


def _entry_of(entries, task_id):
    for entry in entries:
        if entry[0] == task_id:
            return entry
    raise KeyError(task_id)


def _mixed_entries(rng, index):
    """A valid random DAG with 1-3 distinct problem kinds injected."""
    dag = _random_dag(rng, "m%d" % index)
    entries = [[task, list(own)] for task, own in dag.items()]
    ids = [entry[0] for entry in entries]
    kinds = rng.sample(
        ["duplicate", "unknown", "self", "cycle"], 1 + rng.below(3)
    )
    for kind in kinds:
        if kind == "duplicate":
            entries.append([rng.choice(ids), rng.sample(ids, rng.below(2))])
        elif kind == "unknown":
            rng.choice(entries)[1].append(
                "ghost-%d-%d" % (index, rng.below(100))
            )
        elif kind == "self":
            victim = rng.choice(entries)
            victim[1].append(victim[0])
        else:  # cycle: wire two distinct tasks to depend on each other
            first, second = rng.sample(ids, 2)
            _entry_of(entries, first)[1].append(second)
            _entry_of(entries, second)[1].append(first)
    return entries


def _entry_documents(entries, rng, count):
    """Documents listing ``entries`` in permuted task/dependency orders."""
    documents = []
    for _ in range(count):
        tasks = []
        for task, own in rng.shuffle(entries):
            entry = {"id": task}
            own = rng.shuffle(own)
            if own or rng.chance(0.5):
                entry["depends_on"] = own
            tasks.append(entry)
        documents.append({"tasks": tasks})
    return documents


def invalid_cases(seed):
    """The deterministic sequence of invalid-definition cases for ``seed``."""
    rng = Rng("invalid:%s" % seed)
    specs = [
        ("duplicate-task-id", _duplicate_entries(rng)),
        ("unknown-dependency", _unknown_entries(rng)),
        ("self-dependency", _self_entries(rng)),
        ("cycle", _cycle_entries(rng)),
    ]
    for index in range(4):
        specs.append(("mixed-%d" % index, _mixed_entries(rng, index)))

    cases = []
    for name, entries in specs:
        entries = [(task, list(own)) for task, own in entries]
        message = reference_error(entries)
        if message is None:
            raise AssertionError(
                "generator produced a valid graph for case %r" % name
            )
        doc_rng = Rng("invalid-docs:%s:%s" % (seed, name))
        documents = tuple(
            _entry_documents(entries, doc_rng, INVALID_DOCUMENT_VARIANTS)
        )
        cases.append(InvalidCase(name, message, documents))
    return tuple(cases)
