"""Tests for successful ``schedule`` runs.

Covers the documented success contract:

* exit status 0, stderr empty,
* stdout is exactly the compact ``{"batches":[...]}`` document plus one
  newline,
* scheduling starts with every dependency-free task,
* each batch takes up to max-parallel currently-ready tasks by Unicode
  code point order, and successors only join the next batch once the
  whole previous batch is complete,
* repeated dependency entries count once,
* output is byte-for-byte independent of task order, depends_on order,
  duplicate dependency entries, set/hash iteration order (hash seed),
  and locale,
* the scheduler is iterative (a chain beyond the recursion limit works).
"""
import json

import pytest

from conftest import assert_success, make_env, render_batches
from variants import equivalent_documents, spread


# ---------------------------------------------------------------------------
# Small fixed graphs with hard-coded expected schedules.
# ---------------------------------------------------------------------------

def test_empty_tasks(cli, write_def):
    path = write_def({"tasks": []})
    for parallel in (1, 2, 2147483647):
        assert_success(
            cli.schedule(path, parallel), render_batches([])
        )


def test_empty_tasks_compact_spelling(cli, tmp_path):
    path = tmp_path / "pretty.json"
    path.write_text('{\n  "tasks": [ ]\n}\n', encoding="utf-8")
    result = cli.schedule(path, 1)
    assert_success(result, b'{"batches":[]}\n')


def test_single_node(cli, write_def):
    path = write_def({"tasks": [{"id": "only"}]})
    assert_success(cli.schedule(path, 1), render_batches([["only"]]))


def test_chain_at_parallel_one(cli, write_def):
    doc = {
        "tasks": [
            {"id": "a"},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["b"]},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 1),
        render_batches([["a"], ["b"], ["c"]]),
    )


def test_independent_roots_split_into_batches(cli, write_def):
    ids = ["a", "b", "c", "d", "e"]
    path = write_def({"tasks": [{"id": i} for i in ids]})
    assert_success(
        cli.schedule(path, 2),
        render_batches([["a", "b"], ["c", "d"], ["e"]]),
    )


def test_batch_barrier_holds_back_ready_task(cli, write_def):
    # Roots a, c, e; b depends on a; d depends on a and c.
    # At parallelism 2 the first batch is [a, c]: e stays ready but
    # unchosen, and b/d only become candidates afterwards.
    doc = {
        "tasks": [
            {"id": "e"},
            {"id": "d", "depends_on": ["c", "a"]},
            {"id": "c"},
            {"id": "b", "depends_on": ["a"]},
            {"id": "a"},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 2),
        render_batches([["a", "c"], ["b", "d"], ["e"]]),
    )


def test_unselected_ready_task_waits_with_freed_successor(cli, write_def):
    # Roots a, b, c and x depending on a. Parallelism 2:
    # batch 1 [a, b] leaves c pending; after the barrier x is ready too,
    # so batch 2 is [c, x].
    doc = {
        "tasks": [
            {"id": "x", "depends_on": ["a"]},
            {"id": "c"},
            {"id": "b"},
            {"id": "a"},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 2),
        render_batches([["a", "b"], ["c", "x"]]),
    )


def test_parallelism_larger_than_width_matches_levels(cli, write_def):
    doc = {
        "tasks": [
            {"id": "a"},
            {"id": "c"},
            {"id": "b", "depends_on": ["a"]},
            {"id": "d", "depends_on": ["a", "c"]},
        ]
    }
    path = write_def(doc)
    expected = render_batches([["a", "c"], ["b", "d"]])
    for parallel in (2, 3, 100, 2147483647):
        assert_success(cli.schedule(path, parallel), expected)


def test_within_batch_ordered_by_unicode_code_point(cli, write_def):
    # Code points: 9='9'(57) 'A'(65) 'a'(97) 'e'(101) e-acute(233)
    # alpha(945) euro(8364) zhong(20013). None have dependencies.
    ids = ["中", "a", "€", "9", "é", "A", "α", "e"]
    doc = {"tasks": [{"id": task_id} for task_id in ids]}
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 3),
        render_batches(
            [["9", "A", "a"], ["e", "é", "α"], ["€", "中"]]
        ),
    )


def test_duplicate_dependency_entries_count_once(cli, write_def):
    doc = {
        "tasks": [
            {"id": "a"},
            {"id": "c"},
            {"id": "b", "depends_on": ["a", "a"]},
            {"id": "d", "depends_on": ["a", "c", "a", "c", "a"]},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 4),
        render_batches([["a", "c"], ["b", "d"]]),
    )


def test_diamond_parallel_one_and_two(cli, write_def):
    doc = {
        "tasks": [
            {"id": "s"},
            {"id": "x", "depends_on": ["s"]},
            {"id": "y", "depends_on": ["s"]},
            {"id": "t", "depends_on": ["x", "y"]},
        ]
    }
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 1),
        render_batches([["s"], ["x"], ["y"], ["t"]]),
    )
    assert_success(
        cli.schedule(path, 2),
        render_batches([["s"], ["x", "y"], ["t"]]),
    )


# ---------------------------------------------------------------------------
# Full permutation enumeration on small graphs.
# ---------------------------------------------------------------------------

def _all_schedules_byte_identical(cli, write_def, tasks, parallel, expected):
    docs = equivalent_documents(tasks, duplicate_deps=True)
    pairs = [(write_def(doc), parallel) for doc in docs]
    results = cli.schedule_many(pairs)
    assert len(results) == len(docs)
    for result, doc in zip(results, docs):
        assert_success(result, expected), json.dumps(
            doc, ensure_ascii=False
        )


def test_permutation_invariance_four_task_graph(cli, write_def):
    tasks = [
        {"id": "a"},
        {"id": "b", "depends_on": ["a"]},
        {"id": "c"},
        {"id": "d", "depends_on": ["a", "c"]},
    ]
    cases = {
        1: render_batches([["a"], ["b"], ["c"], ["d"]]),
        2: render_batches([["a", "c"], ["b", "d"]]),
        4: render_batches([["a", "c"], ["b", "d"]]),
    }
    for parallel, expected in cases.items():
        _all_schedules_byte_identical(
            cli, write_def, tasks, parallel, expected
        )


def test_sampled_permutations_richer_graph(cli, write_def):
    # 7 tasks: two roots, branching and joining, with a Unicode id.
    tasks = [
        {"id": "root"},
        {"id": "e"},
        {"id": "a", "depends_on": ["root"]},
        {"id": "中", "depends_on": ["root"]},
        {"id": "b", "depends_on": ["root", "a"]},
        {"id": "c", "depends_on": ["a"]},
        {"id": "d", "depends_on": ["b", "c", "中", "e"]},
    ]
    # p=3:
    # batch 1 [e, root, ...] -> roots are "e" and "root": [e, root].
    # then ready [a, 中] (both only need root): batch 2 [a, 中].
    # c needs a; b needs root+a -> both freed: batch 3 [b, c].
    # d waits for 中 (batch 2) and e (batch 1) and b,c (batch 3):
    # batch 4 [d].
    expected = render_batches(
        [["e", "root"], ["a", "中"], ["b", "c"], ["d"]]
    )
    docs = spread(
        equivalent_documents(tasks, duplicate_deps=True), 48
    )
    results = cli.schedule_many(
        [(write_def(doc), 3) for doc in docs]
    )
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
UNICODE_EXPECTED = render_batches([["α"], ["é", "中"]])


@pytest.mark.parametrize(
    "env",
    [
        make_env(),
        make_env(PYTHONUTF8="0"),
        make_env(LC_ALL=None, LANG=None, PYTHONUTF8="1"),
    ],
)
def test_unicode_output_under_c_locale(cli, write_def, env):
    path = write_def(UNICODE_GRAPH)
    assert_success(cli.schedule(path, 2, env=env), UNICODE_EXPECTED)


@pytest.mark.parametrize("seed", ["0", "1", "2", "7", "42"])
def test_output_independent_of_hash_seed(cli, write_def, seed):
    path = write_def(UNICODE_GRAPH)
    assert_success(
        cli.schedule(path, 2, env=make_env(PYTHONHASHSEED=seed)),
        UNICODE_EXPECTED,
    )


def test_output_independent_of_random_hash_seed(cli, write_def):
    paths = [write_def(UNICODE_GRAPH) for _ in range(3)]
    results = cli.schedule_many(
        [(p, 2) for p in paths], env=make_env(PYTHONHASHSEED=None)
    )
    for result in results:
        assert_success(result, UNICODE_EXPECTED)


# ---------------------------------------------------------------------------
# Large fixed graph: a chain beyond the recursion limit plus many
# independent roots, checked against an independent oracle.
# ---------------------------------------------------------------------------

CHAIN_LENGTH = 3001
ROOT_COUNT = 400
PARALLEL = 4


def _large_deps():
    deps = {}
    for i in range(CHAIN_LENGTH):
        node = f"chain-{i:04d}"
        deps[node] = {f"chain-{i - 1:04d}"} if i > 0 else set()
    for j in range(ROOT_COUNT):
        deps[f"root-{j:03d}"] = set()
    return deps


def _reference_batches(deps, parallel):
    """Independent oracle: sorted ready-set slicing with a batch barrier."""
    pending = {node: set(need) for node, need in deps.items()}
    dependents = {node: [] for node in deps}
    for node, need in deps.items():
        for dep in need:
            dependents[dep].append(node)

    ready = sorted(node for node, need in pending.items() if not need)
    batches = []
    while ready:
        batch = ready[:parallel]
        batches.append(batch)
        ready = ready[parallel:]
        for node in batch:
            for successor in dependents[node]:
                pending[successor].discard(node)
                if not pending[successor]:
                    ready.append(successor)
        ready.sort()
    assert sum(map(len, batches)) == len(deps)
    return batches


def test_large_graph_orderings_and_seeds(cli, tmp_path):
    deps = _large_deps()
    expected = render_batches(_reference_batches(deps, PARALLEL))

    assert CHAIN_LENGTH > 3000  # recursive walks cannot survive this

    nodes = sorted(deps)
    listings = [
        nodes,
        list(reversed(nodes)),
    ]
    paths = []
    for i, order in enumerate(listings):
        tasks = []
        for node in order:
            need = sorted(deps[node])
            if i == 1:
                need = list(reversed(need)) * 2  # also repeat each dep
            task = {"id": node}
            if need:
                task["depends_on"] = need
            tasks.append(task)
        path = tmp_path / f"large-{i}.json"
        path.write_text(
            json.dumps({"tasks": tasks}, ensure_ascii=False),
            encoding="utf-8",
        )
        paths.append(path)

    results = cli.schedule_many(
        [(p, PARALLEL) for p in paths], timeout=300
    )
    for result in results:
        assert result.returncode == 0, result.stderr
        assert result.stderr == b""
        assert result.stdout == expected

    seeded = cli.schedule_many(
        [(paths[0], PARALLEL)],
        env=make_env(PYTHONHASHSEED="12345"),
        timeout=300,
    )
    seeded += cli.schedule_many(
        [(paths[0], PARALLEL)],
        env=make_env(PYTHONHASHSEED=None),
        timeout=300,
    )
    for result in seeded:
        assert result.stdout == expected
