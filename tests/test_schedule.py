"""Tests for the ``schedule`` command.

Covers the documented contract:

* exit status 0, stderr empty, stdout is exactly the compact
  ``{"batches":[...]}`` document plus one newline,
* each batch holds at most ``max-parallel`` tasks, chosen from the
  currently ready tasks in Unicode code point order,
* a successor becomes ready only after the whole batch containing its
  dependencies; duplicate dependency entries count as one,
* output is byte-for-byte independent of task order, depends_on order,
  duplicate dependency entries, hash seed, and locale,
* ``max-parallel`` accepts only an ASCII decimal integer in
  1..2147483647 without sign or leading zeros, validated before the
  definition file is read,
* definition errors reuse plan's diagnostics, exit codes, and stream
  rules exactly.
"""
import pytest

from conftest import assert_failure, assert_success, make_env, render_batches
from variants import equivalent_documents, spread

INVALID_PARALLELISM = b"invalid parallelism\n"


# ---------------------------------------------------------------------------
# Small fixed graphs with hard-coded expected batches (independent oracle).
# ---------------------------------------------------------------------------

def test_empty_tasks(cli, write_def):
    path = write_def({"tasks": []})
    assert_success(cli.schedule(path, 1), b'{"batches":[]}\n')


def test_single_node(cli, write_def):
    path = write_def({"tasks": [{"id": "only"}]})
    assert_success(cli.schedule(path, 3), render_batches([["only"]]))


def test_independent_tasks_split_by_parallelism(cli, write_def):
    doc = {"tasks": [{"id": t} for t in ["a", "b", "c", "d", "e"]]}
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 2),
        render_batches([["a", "b"], ["c", "d"], ["e"]]),
    )


def test_parallelism_one_serializes_in_code_point_order(cli, write_def):
    doc = {"tasks": [{"id": t} for t in ["b", "a", "c"]]}
    path = write_def(doc)
    assert_success(cli.schedule(path, 1), render_batches([["a"], ["b"], ["c"]]))


def test_parallelism_covers_all_ready(cli, write_def):
    doc = {"tasks": [{"id": t} for t in ["c", "a", "b"]]}
    path = write_def(doc)
    assert_success(cli.schedule(path, 10), render_batches([["a", "b", "c"]]))


def test_successor_waits_for_whole_batch(cli, write_def):
    # x depends only on a, but a is batched with b; x must wait for batch 2
    # even though a alone would have sufficed.
    doc = {
        "tasks": [
            {"id": "x", "depends_on": ["a"]},
            {"id": "b"},
            {"id": "a"},
        ]
    }
    path = write_def(doc)
    assert_success(cli.schedule(path, 2), render_batches([["a", "b"], ["x"]]))


def test_ready_tasks_refill_between_batches(cli, write_def):
    # b becomes ready after batch 1 and joins the leftover c in batch 2.
    doc = {
        "tasks": [
            {"id": "b", "depends_on": ["a"]},
            {"id": "c"},
            {"id": "a"},
        ]
    }
    path = write_def(doc)
    assert_success(cli.schedule(path, 2), render_batches([["a", "c"], ["b"]]))


def test_diamond(cli, write_def):
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
        cli.schedule(path, 5), render_batches([["s"], ["x", "y"], ["t"]])
    )


def test_duplicate_dependency_counts_once(cli, write_def):
    doc = {
        "tasks": [
            {"id": "b", "depends_on": ["a", "a", "a"]},
            {"id": "a"},
        ]
    }
    path = write_def(doc)
    assert_success(cli.schedule(path, 9), render_batches([["a"], ["b"]]))


def test_unicode_ids_sorted_by_code_point(cli, write_def):
    doc = {"tasks": [{"id": t} for t in ["é", "z", "中", "a"]]}
    path = write_def(doc)
    assert_success(
        cli.schedule(path, 4), render_batches([["a", "z", "é", "中"]])
    )


def test_max_parallel_upper_bound_accepted(cli, write_def):
    path = write_def({"tasks": [{"id": "a"}, {"id": "b"}]})
    assert_success(cli.schedule(path, 2147483647), render_batches([["a", "b"]]))


# ---------------------------------------------------------------------------
# Determinism across equivalent inputs and environments.
# ---------------------------------------------------------------------------

GRAPH = [
    {"id": "root"},
    {"id": "left", "depends_on": ["root"]},
    {"id": "right", "depends_on": ["root"]},
    {"id": "join", "depends_on": ["left", "right"]},
    {"id": "free"},
]

EXPECTED = render_batches([["free", "root"], ["left", "right"], ["join"]])


@pytest.mark.parametrize(
    "doc", spread(equivalent_documents(GRAPH, duplicate_deps=True), 12)
)
def test_equivalent_documents_produce_identical_bytes(cli, write_def, doc):
    path = write_def(doc)
    assert_success(cli.schedule(path, 2), EXPECTED)


@pytest.mark.parametrize("seed", ["0", "1", "555", None])
def test_identical_across_hash_seeds(cli, write_def, seed):
    path = write_def({"tasks": list(GRAPH)})
    env = make_env(PYTHONHASHSEED=seed)
    assert_success(cli.schedule(path, 2, env=env), EXPECTED)


def test_identical_across_locales(cli, write_def):
    path = write_def({"tasks": list(GRAPH)})
    for locale in ("C", "C.UTF-8", "en_US.UTF-8"):
        env = make_env(LC_ALL=locale, LANG=locale)
        assert_success(cli.schedule(path, 2, env=env), EXPECTED)


# ---------------------------------------------------------------------------
# max-parallel argument validation (before the file is ever read).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",
        "0",
        "00",
        "01",
        "+1",
        "-1",
        " 1",
        "1 ",
        "1.0",
        "1e3",
        "abc",
        "１２",  # full-width digits are not ASCII decimal
        "2147483648",
        "99999999999999999999",
    ],
)
def test_invalid_parallelism(cli, write_def, text):
    path = write_def({"tasks": []})
    result = cli.schedule(path, text)
    assert_failure(result, INVALID_PARALLELISM)


@pytest.mark.parametrize("text", ["1", "2", "2147483647"])
def test_valid_parallelism(cli, write_def, text):
    path = write_def({"tasks": [{"id": "a"}]})
    assert_success(cli.schedule(path, text), render_batches([["a"]]))


def test_invalid_parallelism_checked_before_file_read(cli, tmp_path):
    # The definition path does not exist; parallelism must be rejected first.
    result = cli.schedule(tmp_path / "missing.json", "nope")
    assert_failure(result, INVALID_PARALLELISM)


# ---------------------------------------------------------------------------
# Argument count and definition-error behavior (shared with plan).
# ---------------------------------------------------------------------------

def test_wrong_argument_count(cli, write_def):
    from workflow_orchestrator.__main__ import USAGE

    path = write_def({"tasks": []})
    for argv in (["schedule"], ["schedule", str(path)], ["schedule", str(path), "1", "x"]):
        result = cli.run(argv)
        assert result.returncode == 2
        assert result.stdout == b""
        assert result.stderr == USAGE.encode("utf-8")


def test_cannot_read_definition(cli, tmp_path):
    missing = tmp_path / "missing.json"
    assert_failure(
        cli.schedule(missing, 1),
        f"cannot read definition: {missing}\n".encode("utf-8"),
        code=1,
    )


def test_invalid_json(cli, write_def):
    path = write_def(raw=b"{not json")
    assert_failure(cli.schedule(path, 1), b"invalid json\n")


def test_invalid_definition(cli, write_def):
    path = write_def({"tasks": [{"id": "a", "extra": 1}]})
    assert_failure(cli.schedule(path, 1), b"invalid definition\n")


def test_graph_errors_match_plan(cli, write_def):
    cases = [
        (
            {"tasks": [{"id": "a"}, {"id": "a"}]},
            b"duplicate task id: a\n",
        ),
        (
            {"tasks": [{"id": "a", "depends_on": ["a"]}]},
            b"self dependency: a\n",
        ),
        (
            {"tasks": [{"id": "a", "depends_on": ["ghost"]}]},
            b"unknown dependency: a -> ghost\n",
        ),
        (
            {
                "tasks": [
                    {"id": "a", "depends_on": ["b"]},
                    {"id": "b", "depends_on": ["a"]},
                ]
            },
            b"cycle detected\n",
        ),
    ]
    for doc, diagnostic in cases:
        path = write_def(doc)
        assert_failure(cli.schedule(path, 2), diagnostic)
