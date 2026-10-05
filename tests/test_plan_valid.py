"""Tests for successful ``plan`` runs.

Covers the documented success contract:

* exit status 0, stderr empty,
* stdout is exactly the compact ``{"levels":[...]}`` document plus one
  newline,
* a task lands one level below its *deepest* dependency,
* tasks inside a level are ordered by Unicode code point,
* output is byte-for-byte independent of task order, depends_on order,
  duplicate dependency entries, set/hash iteration order (hash seed),
  locale, and (for a large fixed graph) does not rely on recursion.
"""
import json

import pytest

from conftest import assert_success, make_env, render_levels
from variants import equivalent_documents, spread


# ---------------------------------------------------------------------------
# Small fixed graphs with hard-coded expected plans (independent oracle).
# ---------------------------------------------------------------------------

def test_empty_tasks(cli, write_def):
    path = write_def({"tasks": []})
    assert_success(cli.plan(path), render_levels([]))


def test_empty_tasks_compact_spelling(cli, tmp_path):
    # The on-disk spelling must not leak into the compact output.
    path = tmp_path / "pretty.json"
    path.write_text('{\n  "tasks": [ ]\n}\n', encoding="utf-8")
    result = cli.plan(path)
    assert_success(result, b'{"levels":[]}\n')


def test_single_node(cli, write_def):
    path = write_def({"tasks": [{"id": "only"}]})
    assert_success(cli.plan(path), render_levels([["only"]]))


def test_readme_example(cli, write_def):
    doc = {
        "tasks": [
            {"id": "task-a"},
            {"id": "task-b", "depends_on": ["task-a"]},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.plan(path), b'{"levels":[["task-a"],["task-b"]]}\n'
    )


def test_deepest_dependency_determines_level(cli, write_def):
    # c depends on both a (depth 0) and b (depth 1); it must sit at 2.
    doc = {
        "tasks": [
            {"id": "a"},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["a", "b"]},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.plan(path), render_levels([["a"], ["b"], ["c"]])
    )


def test_multi_layer_and_shared_dependency(cli, write_def):
    # Diamond: s is shared by x and y; t joins both branches.
    doc = {
        "tasks": [
            {"id": "t", "depends_on": ["x", "y"]},
            {"id": "y", "depends_on": ["s"]},
            {"id": "x", "depends_on": ["s"]},
            {"id": "s"},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.plan(path),
        render_levels([["s"], ["x", "y"], ["t"]]),
    )


def test_same_level_ordered_by_unicode_code_point(cli, write_def):
    # Code points: 9='9'(57) 'A'(65) 'a'(97) 'e'(101) e-acute(233)
    # alpha(945) euro(8364) zhong(20013). None have dependencies.
    ids = ["中", "a", "€", "9", "é", "A", "α", "e"]
    doc = {"tasks": [{"id": task_id} for task_id in ids]}
    path = write_def(doc)
    expected = [["9", "A", "a", "e", "é", "α", "€", "中"]]
    assert_success(cli.plan(path), render_levels(expected))


# ---------------------------------------------------------------------------
# Full permutation enumeration on small graphs.
# ---------------------------------------------------------------------------

def _all_plans_byte_identical(cli, write_def, tasks, expected):
    """Every task/dep permutation (plus duplicate-dep variants) plans equal."""
    docs = equivalent_documents(tasks, duplicate_deps=True)
    paths = [write_def(doc) for doc in docs]
    results = cli.plan_many(paths)
    assert len(results) == len(docs)
    for result, doc in zip(results, docs):
        assert_success(result, expected), json.dumps(
            doc, ensure_ascii=False
        )


def test_permutation_invariance_ascii_diamond(cli, write_def):
    tasks = [
        {"id": "a"},
        {"id": "b", "depends_on": ["a"]},
        {"id": "c", "depends_on": ["a", "b"]},  # shallow + deep dep
        {"id": "d", "depends_on": ["a"]},
    ]
    # 4! task orders * 2 dep orders for c = 48, plus 24 duplicate-dep docs.
    _all_plans_byte_identical(
        cli, write_def, tasks,
        render_levels([["a"], ["b", "d"], ["c"]]),
    )


def test_permutation_invariance_unicode(cli, write_def):
    # "A" and "中" share level 2; their relative order is by code point.
    tasks = [
        {"id": "α"},
        {"id": "a", "depends_on": ["α"]},
        {"id": "中", "depends_on": ["α", "a"]},
        {"id": "A", "depends_on": ["a"]},
    ]
    _all_plans_byte_identical(
        cli, write_def, tasks,
        render_levels([["α"], ["a"], ["A", "中"]]),
    )


def test_duplicate_dependency_entries_are_ignored(cli, write_def):
    # Repeated, non-adjacent duplicates in several lists.
    doc = {
        "tasks": [
            {"id": "a"},
            {"id": "b", "depends_on": ["a", "a"]},
            {"id": "c", "depends_on": ["a", "b", "a", "b"]},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.plan(path), render_levels([["a"], ["b"], ["c"]])
    )


def test_sampled_permutations_richer_graph(cli, write_def):
    # 7 tasks, 4 levels, a shared root, and Unicode ids across layers.
    # Full enumeration would be 5040 * 4 documents; sample spread orders.
    tasks = [
        {"id": "root"},
        {"id": "a", "depends_on": ["root"]},
        {"id": "mid", "depends_on": ["root"]},
        {"id": "中", "depends_on": ["root"]},
        {"id": "b", "depends_on": ["root", "a"]},
        {"id": "c", "depends_on": ["a"]},
        {"id": "λ", "depends_on": ["中", "mid"]},
        {"id": "d", "depends_on": ["b", "c"]},
    ]
    expected = render_levels(
        [["root"], ["a", "mid", "中"], ["b", "c", "λ"], ["d"]]
    )
    docs = spread(
        equivalent_documents(tasks, duplicate_deps=True), 48
    )
    results = cli.plan_many([write_def(doc) for doc in docs])
    for result in results:
        assert_success(result, expected)


# ---------------------------------------------------------------------------
# Locale and hash-seed independence.
# ---------------------------------------------------------------------------

UNICODE_GRAPH = {
    "tasks": [
        {"id": "α"},
        {"id": "中", "depends_on": ["α"]},
        {"id": "é", "depends_on": ["α"]},
    ]
}
UNICODE_EXPECTED = render_levels([["α"], ["é", "中"]])


@pytest.mark.parametrize(
    "env",
    [
        # Pinned C locale with UTF-8 mode.
        make_env(),
        # UTF-8 mode off but IO encoding explicitly UTF-8: still no locale
        # dependence through the standard streams.
        make_env(PYTHONUTF8="0"),
        # No locale variables at all, UTF-8 mode still forced.
        make_env(LC_ALL=None, LANG=None, PYTHONUTF8="1"),
    ],
)
def test_unicode_output_under_c_locale(cli, write_def, env):
    path = write_def(UNICODE_GRAPH)
    assert_success(cli.plan(path, env=env), UNICODE_EXPECTED)


@pytest.mark.parametrize("seed", ["0", "1", "2", "7", "42"])
def test_output_independent_of_hash_seed(cli, write_def, seed):
    path = write_def(UNICODE_GRAPH)
    assert_success(
        cli.plan(path, env=make_env(PYTHONHASHSEED=seed)),
        UNICODE_EXPECTED,
    )


def test_output_independent_of_random_hash_seed(cli, write_def):
    # Three runs each let the interpreter choose a fresh random seed.
    paths = [write_def(UNICODE_GRAPH) for _ in range(3)]
    results = cli.plan_many(
        paths, env=make_env(PYTHONHASHSEED=None)
    )
    for result in results:
        assert_success(result, UNICODE_EXPECTED)


# ---------------------------------------------------------------------------
# Large fixed graph: deep chain (beyond the recursion limit) plus wide
# layers with shared dependencies.
# ---------------------------------------------------------------------------

# Fixed graph sizes. The chain is three times Python's default recursion
# limit (1000) plus one node: any recursive DAG walk would blow the stack,
# while the product code's iterative traversal handles it.
CHAIN_LENGTH = 3001
WIDE_LEVELS = 10
WIDE_WIDTH = 30


def _large_deps():
    """Fixed large graph as id -> frozenset(deps), built independently of
    any listing order."""
    deps = {}
    for i in range(CHAIN_LENGTH):
        node = f"chain-{i:04d}"
        deps[node] = frozenset(
            [f"chain-{i - 1:04d}"] if i > 0 else []
        )
    for level in range(WIDE_LEVELS):
        for j in range(WIDE_WIDTH):
            node = f"wide-{level:02d}-{j:02d}"
            if level == 0:
                deps[node] = frozenset()
            else:
                # Dense shared dependency: every node depends on every node
                # in the previous wide layer.
                deps[node] = frozenset(
                    f"wide-{level - 1:02d}-{k:02d}"
                    for k in range(WIDE_WIDTH)
                )
    return deps


def _reference_levels(deps):
    """Independent oracle: iterative Kahn longest-path, then group/sort."""
    indegree = {node: len(need) for node, need in deps.items()}
    dependents = {node: [] for node in deps}
    for node, need in deps.items():
        for dep in need:
            dependents[dep].append(node)

    depth = {}
    queue = sorted(node for node, n in indegree.items() if n == 0)
    while queue:
        node = queue.pop(0)
        depth[node] = 0 if not deps[node] else 1 + max(
            depth[dep] for dep in deps[node]
        )
        for nxt in dependents[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    assert len(depth) == len(deps)  # sanity: oracle saw every node

    levels = []
    for node, level in depth.items():
        while len(levels) <= level:
            levels.append([])
        levels[level].append(node)
    for level in levels:
        level.sort()
    return levels


def _listings(deps):
    """Three fixed, very different task/dependency listings of one graph."""
    nodes = sorted(deps)
    reversed_nodes = list(reversed(nodes))
    stride = 7
    # Fixed strided listing: concatenate every residue class modulo the
    # stride, which is still a permutation of all nodes.
    strided = [node for r in range(stride) for node in nodes[r::stride]]
    listings = []
    for order_index, order in enumerate([nodes, reversed_nodes, strided]):
        tasks = []
        for pos, node in enumerate(order):
            need = sorted(deps[node])
            if order_index == 1:
                need = list(reversed(need))
            elif order_index == 2:
                # Fixed interleaving dep order, distinct from both above.
                need = need[::-2] + need[-2::-2]
            task = {"id": node}
            if need:
                task["depends_on"] = need
            tasks.append(task)
        listings.append(tasks)
    return listings


def test_large_graph_all_orderings_and_seeds(cli, tmp_path):
    deps = _large_deps()
    expected = render_levels(_reference_levels(deps))

    # The chain alone is more than three times Python's default recursion
    # limit (1000); a recursive implementation could not survive this depth.
    assert CHAIN_LENGTH > 3000

    paths = []
    for i, tasks in enumerate(_listings(deps)):
        path = tmp_path / f"large-{i}.json"
        path.write_text(
            json.dumps({"tasks": tasks}, ensure_ascii=False),
            encoding="utf-8",
        )
        paths.append(path)

    results = cli.plan_many(paths, timeout=300)
    for result in results:
        assert result.returncode == 0, result.stderr
        assert result.stderr == b""
        assert result.stdout == expected

    # Same forward listing under a fixed and a random hash seed: set
    # iteration order during analysis must never reach the output.
    seeded = cli.plan_many(
        [paths[0]] * 2,
        env=make_env(PYTHONHASHSEED="12345"),
        timeout=300,
    )
    seeded += cli.plan_many(
        [paths[0]] * 2,
        env=make_env(PYTHONHASHSEED=None),
        timeout=300,
    )
    for result in seeded:
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected
