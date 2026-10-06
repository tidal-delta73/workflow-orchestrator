"""Regression tests: ``plan`` and ``schedule`` fail identically.

Both commands consume the same validated graph core, so for every broken
definition they must report the same exit code, the same single stderr
diagnostic, and an empty stdout, regardless of task/dependency listing
order or hash seed. These are black-box tests through the public CLI,
mirroring the boundary guarantees the core tests pin in-process.
"""
import itertools

import pytest

from conftest import assert_failure, make_env
from variants import unique_permutations


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


def assert_commands_agree(cli, path, expected_stderr, code, *, env=None):
    plan_result = cli.plan(path, env=env)
    schedule_result = cli.schedule(path, 1, env=env)
    assert_failure(plan_result, expected_stderr, code=code)
    assert_failure(schedule_result, expected_stderr, code=code)
    # The raw streams must be byte-for-byte the same between the commands.
    assert plan_result.returncode == schedule_result.returncode
    assert plan_result.stdout == schedule_result.stdout
    assert plan_result.stderr == schedule_result.stderr


def diag(text: str) -> bytes:
    return (text + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Each error class, end to end through both commands.
# ---------------------------------------------------------------------------


def test_commands_agree_on_missing_file(cli, tmp_path):
    missing = tmp_path / "missing-é.json"
    expected = f"cannot read definition: {missing}\n".encode("utf-8")
    assert_commands_agree(cli, missing, expected, code=1)


def test_commands_agree_on_directory(cli, tmp_path):
    expected = f"cannot read definition: {tmp_path}\n".encode("utf-8")
    assert_commands_agree(cli, tmp_path, expected, code=1)


def test_commands_agree_on_invalid_json(cli, write_def):
    path = write_def(raw=b'{"tasks": [broken')
    assert_commands_agree(cli, path, b"invalid json\n", code=2)


def test_commands_agree_on_invalid_definition(cli, write_def):
    path = write_def({"tasks": [{"id": "a", "extra": 1}]})
    assert_commands_agree(cli, path, b"invalid definition\n", code=2)


@pytest.mark.parametrize(
    "tasks,message",
    [
        ([{"id": "a"}, {"id": "a"}], "duplicate task id: a"),
        ([{"id": "中"}, {"id": "中"}], "duplicate task id: 中"),
        ([{"id": "a", "depends_on": ["a"]}], "self dependency: a"),
        (
            [{"id": "a", "depends_on": ["ghost"]}],
            "unknown dependency: a -> ghost",
        ),
        (
            [{"id": "α", "depends_on": ["未知"]}],
            "unknown dependency: α -> 未知",
        ),
        (
            [
                {"id": "a", "depends_on": ["b"]},
                {"id": "b", "depends_on": ["a"]},
            ],
            "cycle detected",
        ),
    ],
)
def test_commands_agree_on_each_graph_error(cli, write_def, tasks, message):
    path = write_def({"tasks": tasks})
    assert_commands_agree(cli, path, diag(message), code=2)


# ---------------------------------------------------------------------------
# The shared single-diagnostic choice holds for both commands under every
# listing permutation and several hash seeds.
# ---------------------------------------------------------------------------


def test_commands_agree_under_all_permutations(cli, write_def):
    # dup b; self a->a; unknown a->x; cycle c<->d -> self (a,a) always wins.
    tasks = [
        {"id": "b"},
        {"id": "b"},
        {"id": "a", "depends_on": ["x", "a"]},
        {"id": "c", "depends_on": ["d"]},
        {"id": "d", "depends_on": ["c"]},
    ]
    docs = all_listings(tasks)
    assert docs
    for doc in docs:
        path = write_def(doc)
        assert_commands_agree(cli, path, diag("self dependency: a"), code=2)


def test_commands_agree_across_hash_seeds(cli, write_def):
    # A graph carrying two error kinds at once: duplicate (a, "") sorts
    # before the unknown edge (z -> m) because a < z, under every seed.
    doc = {
        "tasks": [
            {"id": "a"},
            {"id": "a"},
            {"id": "z", "depends_on": ["m"]},
        ]
    }
    for seed in ("0", "7", "99", None):
        path = write_def(doc)
        assert_commands_agree(
            cli,
            path,
            diag("duplicate task id: a"),
            code=2,
            env=make_env(PYTHONHASHSEED=seed),
        )


def test_commands_agree_unicode_code_point_order(cli, write_def):
    # unknown (a -> z); self é -> é; duplicate 中 -> unknown wins by code point.
    doc = {
        "tasks": [
            {"id": "中"},
            {"id": "中"},
            {"id": "é", "depends_on": ["é"]},
            {"id": "a", "depends_on": ["z"]},
        ]
    }
    path = write_def(doc)
    assert_commands_agree(
        cli, path, diag("unknown dependency: a -> z"), code=2
    )


# ---------------------------------------------------------------------------
# Error priority: invalid parallelism is detected before the file is read,
# so schedule must never fall through to a definition diagnostic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parallelism", ["0", "01", "-1", "abc", "1.0", ""])
def test_invalid_parallelism_precedes_every_file_error(
    cli, tmp_path, parallelism
):
    # The file does not exist and is not valid; the parallelism diagnostic
    # must win and stdout must stay empty.
    result = cli.schedule(tmp_path / "missing.json", parallelism)
    assert_failure(result, b"invalid parallelism\n", code=2)


def test_invalid_parallelism_precedes_invalid_json(cli, write_def):
    path = write_def(raw=b"{not json")
    result = cli.schedule(path, "nope")
    assert_failure(result, b"invalid parallelism\n", code=2)


def test_valid_parallelism_then_definition_error_matches_plan(cli, write_def):
    # With legal parallelism, schedule's definition error equals plan's.
    path = write_def(raw=b"{not json")
    plan_result = cli.plan(path)
    schedule_result = cli.schedule(path, "3")
    assert plan_result.returncode == schedule_result.returncode == 2
    assert plan_result.stdout == schedule_result.stdout == b""
    assert plan_result.stderr == schedule_result.stderr == b"invalid json\n"


# ---------------------------------------------------------------------------
# Success contract preserved for both commands: compact JSON with exactly
# one trailing newline, nothing on stderr.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        make_env(),
        make_env(PYTHONHASHSEED=None),
        make_env(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8"),
    ],
)
def test_success_outputs_have_single_trailing_newline(
    cli, write_def, env
):
    doc = {
        "tasks": [
            {"id": "中"},
            {"id": "a", "depends_on": ["中", "中"]},  # duplicate dep
        ]
    }
    path = write_def(doc)
    plan_payload = '{"levels":[["中"],["a"]]}\n'.encode("utf-8")
    schedule_payload = '{"batches":[["中"],["a"]]}\n'.encode("utf-8")
    for result, payload in (
        (cli.plan(path, env=env), plan_payload),
        (cli.schedule(path, 4, env=env), schedule_payload),
    ):
        assert result.returncode == 0
        assert result.stderr == b""
        assert result.stdout == payload
        assert result.stdout.endswith(b"\n")
        assert not result.stdout.endswith(b"\n\n")
        assert result.stdout.count(b"\n") == 1  # exactly one trailing newline


def test_empty_graph_contract_for_both_commands(cli, write_def):
    path = write_def({"tasks": []})
    plan_result = cli.plan(path)
    schedule_result = cli.schedule(path, 2)
    assert plan_result.stdout == b'{"levels":[]}\n'
    assert schedule_result.stdout == b'{"batches":[]}\n'
    assert plan_result.stderr == schedule_result.stderr == b""
    assert plan_result.returncode == schedule_result.returncode == 0
