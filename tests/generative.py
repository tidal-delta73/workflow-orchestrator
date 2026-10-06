"""Seed-driven graph generators and independent reference oracles.

This module is the test-only backbone for the generative regression suite.
It is deliberately self-contained:

* it never imports ``workflow_orchestrator`` -- the expected results below
  are derived from scratch from the documented semantics, so a shared
  implementation bug cannot make oracle and product agree,
* it uses only the Python standard library,
* every random choice comes from a ``random.Random`` seeded with an integer
  (the Mersenne Twister seeding/method sequence is stable across supported
  Python versions), and every set is explicitly sorted before its order can
  influence a choice, so the generated sequence does not depend on
  ``PYTHONHASHSEED``, dict iteration order, or the locale.

Two shapes of data are produced:

* ``entries`` -- ``[{"id": ..., "depends_on"?}]`` task lists used as graph
  content. Valid-case entries have unique ids and reference only known
  tasks; invalid-case entries may break graph-semantic rules but stay
  structurally well formed.
* ``documents`` -- concrete definition JSON values derived from entries by
  permuting task order, dependency order, and duplicating dependency items,
  so equivalent content is presented through non-identical spellings.
"""
import json
import random

# Mixed ASCII/non-ASCII letters, deliberately not listed in code point
# order. Code points: 9='9'(57) a(97) m(109) z(122) e-acute(233)
# alpha(945) euro(8364) zhong(20013).
LETTERS = ("m", "中", "a", "9", "é", "z", "α", "€")


class Seed:
    """A labelled integer seed with readable per-archetype derivation."""

    def __init__(self, value, label):
        self.value = value
        self.label = label

    def child(self, tag):
        # Mix a fixed tag into the parent seed without relying on hash().
        mixed = (self.value * 2862933555777941757 + 1 + _tag_salt(tag)) & 0xFFFFFFFFFFFFFFFF
        return Seed(mixed, f"{self.label}/{tag}")

    def rng(self):
        return random.Random(self.value)


def _tag_salt(tag):
    total = len(tag)
    for ch in tag:
        total = (total * 131 + ord(ch)) & 0xFFFFFFFFFFFFFFFF
    return total


# ---------------------------------------------------------------------------
# IDs and small helpers
# ---------------------------------------------------------------------------

def make_id(rng, index):
    """A unique-per-index task id mixing a seeded letter choice."""
    return f"{rng.choice(LETTERS)}-{index:03d}"


def _task(task_id, deps, *, omit_empty=True):
    task = {"id": task_id}
    if deps or not omit_empty:
        task["depends_on"] = list(deps)
    return task


def sorted_unique_deps(entries):
    """``{id: sorted_tuple(set(deps))}`` merged over all occurrences."""
    merged = {}
    for task in entries:
        merged.setdefault(task["id"], set()).update(task.get("depends_on", ()))
    return {task_id: tuple(sorted(deps)) for task_id, deps in merged.items()}


def document(entries, *, omit_empty=True):
    """A compact definition document from entries."""
    return {
        "tasks": [
            _task(task["id"], task.get("depends_on", []), omit_empty=omit_empty)
            for task in entries
        ]
    }


def dump(document_value):
    """Deterministic compact JSON text; used for files and failure output."""
    return json.dumps(document_value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Valid-DAG archetypes.
#
# Every generator returns entries forming a legal DAG. Dependency edges only
# point at nodes constructed earlier inside the generator, so acyclicity
# holds by construction; ids are unique.
# ---------------------------------------------------------------------------

def gen_empty(rng):
    return []


def gen_multi_root(rng):
    roots = [make_id(rng, i) for i in range(rng.randint(2, 6))]
    entries = [{"id": root} for root in roots]
    for j in range(rng.randint(1, 3)):
        count = rng.randint(1, len(roots))
        deps = rng.sample(roots, count)
        entries.append({"id": make_id(rng, 100 + j), "depends_on": deps})
    return entries


def gen_wide(rng):
    layers = rng.randint(2, 4)
    width = rng.randint(3, 6)
    entries = []
    previous = []
    for layer in range(layers):
        current = [make_id(rng, layer * 100 + j) for j in range(width)]
        for k, node in enumerate(current):
            if layer == 0:
                entries.append({"id": node})
            else:
                # Every non-root node depends on at least one prior-layer
                # node; additional edges are picked independently.
                deps = {rng.choice(previous)}
                for candidate in previous:
                    if rng.random() < 0.35:
                        deps.add(candidate)
                entries.append({"id": node, "depends_on": sorted(deps)})
        previous = current
    return entries


def gen_chain(rng):
    length = rng.randint(4, 40)
    ids = [make_id(rng, i) for i in range(length)]
    entries = [{"id": ids[0]}]
    for i in range(1, length):
        deps = [ids[i - 1]]
        if i >= 2 and rng.random() < 0.3:
            deps.append(ids[i - 2])  # extra backward edge, still acyclic
        entries.append({"id": ids[i], "depends_on": deps})
    return entries


def gen_diamond(rng):
    # widths [1, w1, w2, 1]: a root, two independent middle layers, and a
    # single join depending on every branch tail -- a diamond convergence.
    w1, w2 = rng.randint(2, 4), rng.randint(2, 4)
    root = make_id(rng, 0)
    left = [make_id(rng, 10 + i) for i in range(w1)]
    right = [make_id(rng, 50 + i) for i in range(w2)]
    join = make_id(rng, 99)
    entries = [{"id": root}]
    for node in left:
        entries.append({"id": node, "depends_on": [root]})
    for node in right:
        entries.append({"id": node, "depends_on": left})
    entries.append({"id": join, "depends_on": right})
    return entries


def gen_independent_subgraphs(rng):
    # Two or three pairwise-disjoint components (a chain and small diamonds)
    # with no cross edges at all.
    components = rng.randint(2, 3)
    entries = []
    for c in range(components):
        root = f"g{c}-root"
        entries.append({"id": root})
        mid = f"g{c}-mid"
        leaf = f"g{c}-leaf"
        if rng.random() < 0.5:
            entries.append({"id": mid, "depends_on": [root]})
            entries.append({"id": leaf, "depends_on": [mid]})
        else:
            a, b = f"g{c}-a", f"g{c}-b"
            entries.append({"id": a, "depends_on": [root]})
            entries.append({"id": b, "depends_on": [root]})
            entries.append({"id": leaf, "depends_on": [a, b]})
    return entries


def gen_random_dag(rng):
    # General DAG: nodes are born in index order and each may depend on any
    # strictly earlier node. Roots, chains, diamonds, and disjoint pieces
    # all arise as special cases.
    n = rng.randint(4, 12)
    ids = [make_id(rng, i) for i in range(n)]
    edge_prob = rng.choice((0.15, 0.3, 0.45))
    entries = []
    for i, node in enumerate(ids):
        deps = [earlier for earlier in ids[:i] if rng.random() < edge_prob]
        if i > 0 and not deps and rng.random() < 0.7:
            deps = [rng.choice(ids[:i])]
        entries.append(
            {"id": node} if not deps else {"id": node, "depends_on": deps}
        )
    return entries


VALID_ARCHETYPES = {
    "empty": gen_empty,
    "multi-root": gen_multi_root,
    "wide": gen_wide,
    "chain": gen_chain,
    "diamond": gen_diamond,
    "subgraphs": gen_independent_subgraphs,
    "random": gen_random_dag,
}


# ---------------------------------------------------------------------------
# Equivalent (and non-equivalent-but-still-valid) spellings of one graph.
# ---------------------------------------------------------------------------

def equivalent_documents(entries, rng, *, variants=5):
    """Present one valid graph through ``1 + variants`` different JSON docs.

    The first document is a fixed extreme permutation (reverse task order,
    reverse dependency order). Remaining documents independently shuffle
    task order and each dependency list. Half of the shuffled documents
    additionally repeat one dependency item per non-empty dependency list,
    and one document omits empty ``depends_on`` keys differently. All docs
    are valid and semantically identical.
    """
    ids = [task["id"] for task in entries]
    deps_by_id = {
        task["id"]: list(task.get("depends_on", [])) for task in entries
    }

    def build(order, dep_orders, *, duplicate=False, omit_empty=True):
        built = []
        for task_id in order:
            deps = list(dep_orders[task_id])
            if duplicate and deps:
                # Repeat the first item at the end (non-adjacent when the
                # list has several entries).
                deps.append(deps[0])
            built.append(_task(task_id, deps, omit_empty=omit_empty))
        return {"tasks": built}

    documents = []
    # Fixed extreme permutation: no rng consumption, always identical.
    documents.append(
        build(
            list(reversed(ids)),
            {tid: list(reversed(deps_by_id[tid])) for tid in ids},
        )
    )
    for v in range(variants):
        order = list(ids)
        rng.shuffle(order)
        dep_orders = {}
        for task_id in ids:
            deps = list(deps_by_id[task_id])
            rng.shuffle(deps)
            dep_orders[task_id] = deps
        documents.append(
            build(
                order,
                dep_orders,
                duplicate=(v % 2 == 0),
                omit_empty=(v != variants - 1),
            )
        )
    return documents


# ---------------------------------------------------------------------------
# Invalid-graph generators. Entries stay structurally valid so only graph
# semantics (layer 4) are exercised.
# ---------------------------------------------------------------------------

def _gen_duplicate(rng):
    ids = [make_id(rng, i) for i in range(rng.randint(2, 5))]
    victim = rng.choice(ids)
    entries = [{"id": task_id} for task_id in ids]
    entries.insert(rng.randrange(len(entries) + 1), {"id": victim})
    if rng.random() < 0.4:
        # Give one of the other ids a normal dependency for variety.
        other = rng.choice([t for t in ids if t != victim])
        entries.append({"id": other + "-x", "depends_on": [other]})
    return entries


def _gen_unknown(rng):
    ids = [make_id(rng, i) for i in range(rng.randint(2, 5))]
    entries = [{"id": task_id} for task_id in ids]
    missing = f"ghost-{rng.randrange(1000):03d}"
    target = rng.choice(ids)
    deps = [missing]
    if rng.random() < 0.5:
        deps.append(rng.choice([t for t in ids if t != target]))
    entries.append({"id": target + "-u", "depends_on": deps})
    if rng.random() < 0.3:
        # A second unknown edge; the oracle must pick the smaller pair.
        entries.append(
            {"id": target + "-w", "depends_on": [f"ghost-{rng.randrange(1000):03d}"]}
        )
    return entries


def _gen_self(rng):
    ids = [make_id(rng, i) for i in range(rng.randint(1, 4))]
    entries = [{"id": task_id} for task_id in ids]
    target = rng.choice(ids)
    self_id = target + "-s"
    deps = [self_id]
    others = [t for t in ids if t != target]
    if others and rng.random() < 0.5:
        deps.append(rng.choice(others))  # an ordinary edge alongside the self edge
    entries.append({"id": self_id, "depends_on": deps})
    return entries


def _gen_cycle(rng):
    size = rng.choice((2, 3))
    cyc = [make_id(rng, 100 + i) for i in range(size)]
    entries = []
    if rng.random() < 0.5:
        # An unrelated root feeding into the cycle.
        entries.append({"id": make_id(rng, 0)})
        entries.append({"id": cyc[0], "depends_on": [entries[0]["id"], cyc[-1]]})
    else:
        entries.append({"id": cyc[0], "depends_on": [cyc[-1]]})
    for i in range(1, size):
        entries.append({"id": cyc[i], "depends_on": [cyc[i - 1]]})
    if rng.random() < 0.4:
        # A node leaving the cycle (does not break it).
        entries.append({"id": make_id(rng, 900), "depends_on": [cyc[0]]})
    return entries


def _gen_mixed(rng):
    """One document containing several defect kinds at once.

    The winner varies with the seed because letters and phantom ids are
    seeded; the independent oracle decides which diagnostic must surface.
    """
    entries = []

    # A duplicate id, using a deliberately small-code-point letter pool so
    # it can win or lose depending on the other ids drawn.
    dup_id = f"{rng.choice(('a', 'm', 'z', 'é'))}-dup"
    entries.append({"id": dup_id})
    entries.append({"id": dup_id})

    # A self dependency on a separate id.
    self_id = f"{rng.choice(('a', 'm', 'z', 'é'))}-self"
    entries.append({"id": self_id, "depends_on": [self_id]})

    # An unknown dependency on a separate id.
    unk_task = f"{rng.choice(('a', 'm', 'z', 'é'))}-unk"
    entries.append(
        {"id": unk_task, "depends_on": [f"ghost-{rng.randrange(1000):03d}"]}
    )

    # A 2-node cycle under yet another id prefix.
    cx, cy = f"{rng.choice(('a', 'm', 'z', 'é'))}-c0", f"{rng.choice(('b', 'n', 'α'))}-c1"
    entries.append({"id": cx, "depends_on": [cy]})
    entries.append({"id": cy, "depends_on": [cx]})

    # An innocent bystander task.
    entries.append({"id": "ok-root"})
    rng.shuffle(entries)
    return entries


INVALID_ARCHETYPES = {
    "duplicate": _gen_duplicate,
    "unknown": _gen_unknown,
    "self": _gen_self,
    "cycle": _gen_cycle,
    "mixed": _gen_mixed,
}


def listing_variants_invalid(entries, rng, *, variants=2):
    """Task/dep permutations of an invalid document.

    Duplicate task occurrences are preserved; the chosen diagnostic must be
    identical for every listing.
    """
    documents = [document(entries)]
    for _ in range(variants):
        shuffled = list(entries)
        rng.shuffle(shuffled)
        shuffled_entries = []
        for task in shuffled:
            deps = list(task.get("depends_on", ()))
            rng.shuffle(deps)
            shuffled_entries.append(
                {"id": task["id"], **({"depends_on": deps} if "depends_on" in task else {})}
            )
        documents.append({"tasks": shuffled_entries})
    return documents


# ---------------------------------------------------------------------------
# Independent reference oracles.
# ---------------------------------------------------------------------------

def _reach(adj, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def reference_levels(entries):
    """Expected ``plan`` levels: deepest-dependency depth, code point order.

    Independent Kahn-style longest pass over sorted queues; never reads the
    product implementation.
    """
    deps = sorted_unique_deps(entries)
    indegree = {task_id: len(need) for task_id, need in deps.items()}
    dependents = {task_id: [] for task_id in deps}
    for task_id, need in deps.items():
        for dep in need:
            dependents[dep].append(task_id)

    depth = {}
    queue = sorted(task_id for task_id, n in indegree.items() if n == 0)
    while queue:
        node = queue.pop(0)
        need = deps[node]
        depth[node] = 0 if not need else 1 + max(depth[d] for d in need)
        for nxt in sorted(dependents[node]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    levels = []
    for task_id in sorted(depth, key=lambda t: (depth[t], t)):
        while len(levels) <= depth[task_id]:
            levels.append([])
        levels[depth[task_id]].append(task_id)
    return levels


def reference_batches(entries, max_parallel):
    """Expected ``schedule`` batches under the documented greedy rule.

    Each batch takes up to ``max_parallel`` of the tasks ready at batch
    start in Unicode code point order; successors become ready only once
    the whole batch has been scheduled.
    """
    deps = sorted_unique_deps(entries)
    ids = sorted(deps)
    dependents = {task_id: [] for task_id in ids}
    for task_id in ids:
        for dep in deps[task_id]:
            dependents[dep].append(task_id)

    remaining = {task_id: len(deps[task_id]) for task_id in ids}
    ready = sorted(task_id for task_id in ids if remaining[task_id] == 0)
    batches = []
    while ready:
        batch = ready[:max_parallel]
        batches.append(batch)
        ready = ready[max_parallel:]
        for task_id in batch:
            for nxt in dependents[task_id]:
                remaining[nxt] -= 1
                if remaining[nxt] == 0:
                    ready.append(nxt)
        ready.sort()
    return batches


def reference_diagnostic(entries):
    """The one graph-semantic error message the commands must report.

    Selection, derived directly from the README contract: candidates are
    keyed by (task id, dependency id, kind rank) with duplicate < self <
    unknown < cycle on an exact-pair tie; the minimum key wins. Returns
    None for valid graphs.
    """
    counts = {}
    for task in entries:
        counts[task["id"]] = counts.get(task["id"], 0) + 1
    known = set(counts)

    candidates = []
    for task_id in sorted(counts):
        if counts[task_id] > 1:
            candidates.append((task_id, "", 0, f"duplicate task id: {task_id}"))

    adj = {task_id: set() for task_id in known}
    for task in entries:
        task_id = task["id"]
        for dep in set(task.get("depends_on", ())):
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

    cycle_key = None
    for task_id in sorted(adj):
        for dep in sorted(adj[task_id]):
            if task_id in _reach(adj, dep):
                key = (task_id, dep)
                if cycle_key is None or key < cycle_key:
                    cycle_key = key
    if cycle_key is not None:
        candidates.append((cycle_key[0], cycle_key[1], 3, "cycle detected"))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[3]


# ---------------------------------------------------------------------------
# Seeded invalid max-parallel spellings.
# ---------------------------------------------------------------------------

def invalid_parallelism_spellings(seed=20240917):
    """Deterministic batch of malformed max-parallel arguments.

    Mixes a fixed edge list (boundaries, signs, whitespace, non-ASCII
    digits) with seeded random junk, so future loosening of the parser is
    caught without any dependence on machine state.
    """
    fixed = [
        "0", "-1", "+1", "01", "00", "1.0", "1e3", " 1", "1 ",
        "abc", "", "１２", "2147483648", "99999999999999999999999",
    ]
    rng = random.Random(seed)
    alphabet = "0123456789+- .xXabcde_"
    for _ in range(16):
        length = rng.randint(0, 5)
        text = "".join(rng.choice(alphabet) for _ in range(length))
        # Anything the generator accidentally makes valid is replaced by a
        # guaranteed-invalid wrapping; reference check is done by the caller
        # via the product's own rule duplication below.
        if text.isdigit() and text and not (len(text) > 1 and text[0] == "0"):
            value = int(text)
            if 1 <= value <= 2147483647:
                text = text + "x"
        fixed.append(text)
    # De-duplicate while preserving the stable generation order.
    return list(dict.fromkeys(fixed))


# ---------------------------------------------------------------------------
# Deterministic case builders (the fixed "case sequence").
# ---------------------------------------------------------------------------

VALID_CASE_COUNT_PER_ARCHETYPE = 4
INVALID_CASE_COUNT_PER_KIND = 4
MASTER_SEED = 0xEC0006


def build_valid_cases(seed=MASTER_SEED, *, variants=2):
    """Every valid (archetype, index) case, in a stable order.

    ``variants`` shuffled spellings are produced in addition to the one
    fixed extreme permutation, so each case carries ``variants + 1``
    equivalent documents.
    """
    root = Seed(seed, "valid")
    cases = []
    for name in sorted(VALID_ARCHETYPES):
        for i in range(VALID_CASE_COUNT_PER_ARCHETYPE):
            case_seed = root.child(f"{name}:{i}")
            rng = case_seed.rng()
            entries = VALID_ARCHETYPES[name](rng)
            docs = equivalent_documents(entries, rng, variants=variants)
            cases.append(
                {
                    "seed": case_seed.value,
                    "label": case_seed.label,
                    "archetype": name,
                    "index": i,
                    "entries": entries,
                    "documents": docs,
                    "levels": reference_levels(entries),
                    "n_tasks": len({task["id"] for task in entries}),
                }
            )
    return cases


def build_invalid_cases(seed=MASTER_SEED, *, variants=2):
    """Every invalid (kind, index) case plus its listing variants.

    ``variants`` shuffled listings are produced in addition to the original
    ordering, so each case carries ``variants + 1`` documents.
    """
    root = Seed(seed, "invalid")
    cases = []
    for name in sorted(INVALID_ARCHETYPES):
        for i in range(INVALID_CASE_COUNT_PER_KIND):
            case_seed = root.child(f"{name}:{i}")
            rng = case_seed.rng()
            entries = INVALID_ARCHETYPES[name](rng)
            docs = listing_variants_invalid(entries, rng, variants=variants)
            cases.append(
                {
                    "seed": case_seed.value,
                    "label": case_seed.label,
                    "kind": name,
                    "index": i,
                    "documents": docs,
                    "diagnostic": reference_diagnostic(entries),
                }
            )
    return cases


def canonical_case_text(cases):
    """Stable serialization of a built case list (for reproducibility)."""
    return json.dumps(cases, ensure_ascii=False, sort_keys=True)
