"""Tests for invalid definitions and unreadable files.

Every failure must satisfy the public contract:

* the documented exit code (1 for unreadable files, 2 for everything else),
* nothing at all on stdout,
* exactly one diagnostic line on stderr, chosen by Unicode code point order
  of (task id, dependency id) when several graph problems coexist,
* that choice is invariant over every listing permutation of the document.
"""
import itertools
import os

import pytest

from conftest import assert_failure, make_env
from variants import unique_permutations


def diag(text: str) -> bytes:
    return (text + "\n").encode("utf-8")


def all_listings(tasks):
    """All distinct task orders x depends_on orders for a task list."""
    docs = []
    for perm in unique_permutations(tasks):
        options = []
        for task in perm:
            deps = task.get("depends_on")
            if deps and len(deps) >= 2:
                orders = unique_permutations(deps)
                options.append(
                    [{**task, "depends_on": list(order)} for order in orders]
                )
            else:
                options.append([task])
        for combo in itertools.product(*options):
            docs.append({"tasks": list(combo)})
    return docs


def assert_single_diagnostic_everywhere(cli, write_def, tasks, message, *, code=2):
    docs = all_listings(tasks)
    assert docs  # guard: the permutation machinery produced cases
    paths = [write_def(doc) for doc in docs]
    results = cli.plan_many(paths)
    assert len(results) == len(docs)
    for result in results:
        assert_failure(result, diag(message), code=code)


# ---------------------------------------------------------------------------
# The file itself cannot be read (exit code 1).
# ---------------------------------------------------------------------------

def test_missing_file(cli):
    missing = "/tmp/definitely-not-here-echo002.json"
    assert not os.path.exists(missing)
    result = cli.plan(missing)
    assert_failure(
        result, diag(f"cannot read definition: {missing}"), code=1
    )


def test_missing_file_path_echoed_verbatim(cli, tmp_path):
    # Spaces and non-ASCII characters in the path are echoed byte-for-byte.
    path = tmp_path / "missing 定義.json"
    result = cli.plan(path)
    assert_failure(
        result,
        ("cannot read definition: " + str(path) + "\n").encode("utf-8"),
        code=1,
    )


def test_directory_is_not_readable_as_definition(cli, tmp_path):
    result = cli.plan(tmp_path)
    assert_failure(
        result,
        ("cannot read definition: " + str(tmp_path) + "\n").encode("utf-8"),
        code=1,
    )


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="permission bits are not enforced for root",
)
def test_unreadable_file_permissions(cli, tmp_path):
    path = tmp_path / "noperm.json"
    path.write_text('{"tasks":[]}', encoding="utf-8")
    os.chmod(path, 0o000)
    try:
        result = cli.plan(path)
        assert_failure(
            result,
            ("cannot read definition: " + str(path) + "\n").encode("utf-8"),
            code=1,
        )
    finally:
        os.chmod(path, 0o644)


# ---------------------------------------------------------------------------
# Unreadable file beats every other check and never writes to stdout.
# ---------------------------------------------------------------------------

def test_unreadable_file_has_empty_stdout_even_with_hash_seed(cli):
    result = cli.plan(
        "/tmp/no-such-echo002-file.json",
        env=make_env(PYTHONHASHSEED="99"),
    )
    assert result.returncode == 1
    assert result.stdout == b""


# ---------------------------------------------------------------------------
# The file is readable but not JSON (exit code 2, "invalid json").
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"not json at all",
        b'{"tasks": [',
        b'{"tasks": [}]}',
        b"\xff\xfe{not valid utf8}",
        b"\x00\x01\x02",
        b"undefined",
        b"'single quotes are not json'",
        b"{tasks: []}",
    ],
)
def test_invalid_json(cli, write_def, raw):
    path = write_def(raw=raw)
    assert_failure(result=cli.plan(path), expected_stderr=diag("invalid json"))


def test_invalid_json_stdout_empty(cli, write_def):
    path = write_def(raw=b'{"tasks": [broken')
    result = cli.plan(path)
    assert result.stdout == b""


# ---------------------------------------------------------------------------
# JSON parses but the value is not a definition (exit 2, "invalid definition").
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "document",
    [
        None,
        True,
        42,
        3.14,
        "a string",
        ["tasks"],
        [],
        {},
        {"other": []},
        {"tasks": [], "extra": 1},
        {"Tasks": []},
        {"tasks": None},
        {"tasks": {}},
        {"tasks": "tasks"},
        {"tasks": 42},
        {"tasks": [None]},
        {"tasks": ["a"]},
        {"tasks": [42]},
        {"tasks": [True]},
        {"tasks": [[]]},
        {"tasks": [{"id": "a"}, "not-an-object"]},
        {"tasks": [{"id": ""}]},
        {"tasks": [{"id": 42}]},
        {"tasks": [{"id": None}]},
        {"tasks": [{"id": ["a"]}]},
        {"tasks": [{"id": "a", "depends_on": []}, {"id": "b", "x": 1}]},
        {"tasks": [{"id": "a", "depends_on": {}}]},
        {"tasks": [{"id": "a", "depends_on": "x"}]},
        {"tasks": [{"id": "a", "depends_on": None}]},
        {"tasks": [{"id": "a", "depends_on": [1]}]},
        {"tasks": [{"id": "a", "depends_on": [None]}]},
        {"tasks": [{"id": "a", "depends_on": [[]]}]},
        {"tasks": [{"id": "a", "depends_on": [{}]}]},
        {"tasks": [{"id": "a", "depends_on": ["ok", 2]}]},
    ],
)
def test_invalid_definition(cli, write_def, document):
    path = write_def(document)
    assert_failure(result=cli.plan(path), expected_stderr=diag("invalid definition"))


# ---------------------------------------------------------------------------
# Single graph problems.
# ---------------------------------------------------------------------------

def test_duplicate_task_id(cli, write_def):
    path = write_def({"tasks": [{"id": "a"}, {"id": "a"}]})
    assert_failure(result=cli.plan(path), expected_stderr=diag("duplicate task id: a"))


def test_duplicate_task_id_three_times(cli, write_def):
    path = write_def(
        {"tasks": [{"id": "z"}, {"id": "z"}, {"id": "z"}]}
    )
    assert_failure(result=cli.plan(path), expected_stderr=diag("duplicate task id: z"))


def test_duplicate_unicode_task_id(cli, write_def):
    path = write_def({"tasks": [{"id": "中"}, {"id": "中"}]})
    assert_failure(
        result=cli.plan(path), expected_stderr=diag("duplicate task id: 中")
    )


def test_unknown_dependency(cli, write_def):
    path = write_def(
        {"tasks": [{"id": "a", "depends_on": ["missing"]}]}
    )
    assert_failure(
        result=cli.plan(path),
        expected_stderr=diag("unknown dependency: a -> missing"),
    )


def test_unknown_dependency_empty_string_id(cli, write_def):
    # A dependency may structurally be any string; "" names no task.
    path = write_def({"tasks": [{"id": "a", "depends_on": [""]}]})
    assert_failure(
        result=cli.plan(path),
        expected_stderr=diag("unknown dependency: a -> "),
    )


def test_unknown_dependency_unicode(cli, write_def):
    path = write_def({"tasks": [{"id": "α", "depends_on": ["未知"]}]})
    result = cli.plan(path)
    assert_failure(
        result, diag("unknown dependency: α -> 未知")
    )
    # Raw UTF-8 bytes, regardless of the machine locale.
    assert result.stderr == "unknown dependency: α -> 未知\n".encode("utf-8")


def test_self_dependency(cli, write_def):
    path = write_def({"tasks": [{"id": "a", "depends_on": ["a"]}]})
    assert_failure(
        result=cli.plan(path), expected_stderr=diag("self dependency: a")
    )


def test_self_dependency_repeated_still_one_diagnostic(cli, write_def):
    path = write_def({"tasks": [{"id": "é", "depends_on": ["é", "é"]}]})
    assert_failure(
        result=cli.plan(path), expected_stderr=diag("self dependency: é")
    )


def test_two_node_cycle(cli, write_def):
    doc = {
        "tasks": [
            {"id": "a", "depends_on": ["b"]},
            {"id": "b", "depends_on": ["a"]},
        ]
    }
    assert_failure(result=cli.plan(write_def(doc)), expected_stderr=diag("cycle detected"))


def test_three_node_cycle(cli, write_def):
    doc = {
        "tasks": [
            {"id": "a", "depends_on": ["b"]},
            {"id": "b", "depends_on": ["c"]},
            {"id": "c", "depends_on": ["a"]},
        ]
    }
    assert_failure(result=cli.plan(write_def(doc)), expected_stderr=diag("cycle detected"))


def test_cycle_with_entering_and_leaving_edges(cli, write_def):
    # root enters the b<->c cycle; leaf leaves it. The cycle still reports.
    doc = {
        "tasks": [
            {"id": "root"},
            {"id": "a", "depends_on": ["b"]},
            {"id": "b", "depends_on": ["c", "root"]},
            {"id": "c", "depends_on": ["b"]},
        ]
    }
    assert_failure(result=cli.plan(write_def(doc)), expected_stderr=diag("cycle detected"))


# ---------------------------------------------------------------------------
# Multiple simultaneous problems: one diagnostic, picked by Unicode order
# of (task id, dependency id), stable under every listing permutation.
# ---------------------------------------------------------------------------

def test_multi_problem_self_beats_unknown_dup_and_cycle(cli, write_def):
    # dup b; self a->a; unknown a->x; cycle c<->d.
    # Ordered pairs: (a,a,self) < (a,x,unknown) < (b,'',dup) < (c,d,cycle).
    tasks = [
        {"id": "b"},
        {"id": "b"},
        {"id": "a", "depends_on": ["x", "a"]},
        {"id": "c", "depends_on": ["d"]},
        {"id": "d", "depends_on": ["c"]},
    ]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "self dependency: a"
    )


def test_multi_problem_unknown_beats_duplicate_by_task_id(cli, write_def):
    # unknown (a -> m) sorts before duplicate (z, "") because a < z.
    tasks = [
        {"id": "z"},
        {"id": "z"},
        {"id": "a", "depends_on": ["m"]},
    ]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "unknown dependency: a -> m"
    )


def test_multi_problem_duplicate_beats_unknown_by_task_id(cli, write_def):
    # duplicate (a, "") sorts before unknown (z -> m) because a < z.
    tasks = [
        {"id": "a"},
        {"id": "a"},
        {"id": "z", "depends_on": ["m"]},
    ]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "duplicate task id: a"
    )


def test_multi_problem_self_vs_unknown_same_task_dep_order(cli, write_def):
    # One task with two bad deps; all_listings enumerates both depends_on
    # orders. Self pair (a,a) precedes unknown pair (a,x), so self wins in
    # every listing. (No duplicate task id is introduced.)
    tasks = [{"id": "a", "depends_on": ["a", "x"]}]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "self dependency: a"
    )


def test_multi_problem_two_unknowns_same_task_pick_smaller_dep(cli, write_def):
    # Both dependency ids unknown; the smaller dep id is reported,
    # independent of the depends_on listing order.
    tasks = [{"id": "a", "depends_on": ["z", "y"]}]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "unknown dependency: a -> y"
    )


def test_multi_problem_unknown_beats_cycle_when_task_id_smaller(cli, write_def):
    # unknown (a -> x) vs cycle edge (m, n): a < m, so unknown wins.
    tasks = [
        {"id": "a", "depends_on": ["x"]},
        {"id": "m", "depends_on": ["n"]},
        {"id": "n", "depends_on": ["m"]},
    ]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "unknown dependency: a -> x"
    )


def test_multi_problem_cycle_beats_unknown_when_task_id_smaller(cli, write_def):
    # cycle edge (a, b) vs unknown (z -> q): a < z, so the cycle wins.
    tasks = [
        {"id": "a", "depends_on": ["b"]},
        {"id": "b", "depends_on": ["a"]},
        {"id": "z", "depends_on": ["q"]},
    ]
    assert_single_diagnostic_everywhere(cli, write_def, tasks, "cycle detected")


def test_multi_problem_two_cycles_report_smallest_edge(cli, write_def):
    tasks = [
        {"id": "a", "depends_on": ["b"]},
        {"id": "b", "depends_on": ["a"]},
        {"id": "m", "depends_on": ["n"]},
        {"id": "n", "depends_on": ["m"]},
    ]
    assert_single_diagnostic_everywhere(cli, write_def, tasks, "cycle detected")


def test_multi_problem_smallest_of_two_duplicates(cli, write_def):
    tasks = [
        {"id": "a"},
        {"id": "a"},
        {"id": "b"},
        {"id": "b"},
        {"id": "c", "depends_on": ["d"]},
    ]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "duplicate task id: a"
    )


def test_multi_problem_unicode_code_point_order(cli, write_def):
    # unknown (a -> z); self é -> é; duplicate 中.
    # Code points: a(97) < é(233) < 中(20013), so unknown wins.
    tasks = [
        {"id": "中"},
        {"id": "中"},
        {"id": "é", "depends_on": ["é"]},
        {"id": "a", "depends_on": ["z"]},
    ]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "unknown dependency: a -> z"
    )


def test_multi_problem_unicode_self_beats_unicode_dup(cli, write_def):
    # self (é, é) precedes duplicate (中, "") by code point.
    tasks = [
        {"id": "中"},
        {"id": "中"},
        {"id": "é", "depends_on": ["é"]},
    ]
    assert_single_diagnostic_everywhere(
        cli, write_def, tasks, "self dependency: é"
    )


def test_multi_problem_diagnostic_stable_under_hash_seed(cli, write_def):
    tasks = [
        {"id": "b"},
        {"id": "b"},
        {"id": "a", "depends_on": ["x", "a"]},
        {"id": "c", "depends_on": ["d"]},
        {"id": "d", "depends_on": ["c"]},
    ]
    docs = all_listings(tasks)[:12]
    for seed in ("0", "31", "2024"):
        results = cli.plan_many(
            [write_def(doc) for doc in docs],
            env=make_env(PYTHONHASHSEED=seed),
        )
        for result in results:
            assert_failure(result, diag("self dependency: a"))


def test_structural_failure_precedes_graph_checks(cli, write_def):
    # Even with duplicates/unknown deps present, a malformed task structure
    # reports "invalid definition" and nothing reaches stdout.
    path = write_def(
        {
            "tasks": [
                {"id": "a"},
                {"id": "a"},
                {"id": "b", "depends_on": ["ghost"]},
                {"id": 123},
            ]
        }
    )
    result = cli.plan(path)
    assert_failure(result, diag("invalid definition"))
