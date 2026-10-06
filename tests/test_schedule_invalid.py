"""Tests for ``schedule`` argument handling and failure paths.

The success contract guarantees nothing reaches stdout on any failure:

* wrong argument count prints the full usage text to stderr and exits 2,
* a malformed or out-of-range parallelism prints exactly
  ``invalid parallelism`` to stderr, exits 2, and is decided *before* the
  definition file is read (an unreadable path still reports the
  parallelism error),
* once the parallelism is valid, every definition problem reuses plan's
  exact diagnostic text, precedence, and exit code (1 for an unreadable
  file, 2 otherwise).
"""
import io
import os
import sys

import pytest

from conftest import make_env
from workflow_orchestrator.__main__ import USAGE, main

USAGE_BYTES = USAGE.encode("utf-8")
INVALID_PARALLELISM = b"invalid parallelism\n"


def run_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        rc = main(list(argv))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out.getvalue().encode("utf-8"), err.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Argument count: full usage on stderr, exit 2, nothing on stdout.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "argv",
    [
        ("schedule",),
        ("schedule", "a.json"),
        ("schedule", "a.json", "1", "x"),
    ],
)
def test_wrong_argc_subprocess(cli, argv):
    result = cli.run(list(argv))
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == USAGE_BYTES


@pytest.mark.parametrize(
    "argv",
    [
        ("schedule",),
        ("schedule", "a.json"),
        ("schedule", "a.json", "1", "x"),
    ],
)
def test_wrong_argc_inproc(argv):
    rc, out, err = run_main(*argv)
    assert (rc, out, err) == (2, b"", USAGE_BYTES)


# ---------------------------------------------------------------------------
# Parallelism text: exact ASCII-decimal grammar, signedness, range.
# ---------------------------------------------------------------------------

VALID = ["1", "2", "3", "10", "1000000", "2147483647"]


@pytest.mark.parametrize("text", VALID)
def test_valid_parallelism_accepts_file(cli, write_def, text):
    path = write_def({"tasks": []})
    result = cli.schedule(path, text)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b'{"batches":[]}\n'
    assert result.stderr == b""


INVALID_TEXTS = [
    "",              # empty
    "0",             # below minimum
    "-1",            # sign
    "+1",            # plus sign
    "01",            # leading zero
    "00",
    "0001",
    " 1",            # surrounding whitespace
    "1 ",
    "\t1",
    "1\n",
    "1.0",           # not an integer spelling
    "1e3",
    "0x1",
    "1_000",
    "9999999999999999999999",  # above the upper bound
    "2147483648",    # bound + 1
    "4294967296",
    "１２",          # Unicode fullwidth digits
    "١٢",            # Arabic-Indic digits
    "①",
    "一",
    "nan",
    "inf",
    "true",
    "0b1",
]


@pytest.mark.parametrize("text", INVALID_TEXTS)
def test_invalid_parallelism_subprocess(cli, write_def, text):
    path = write_def({"tasks": []})
    result = cli.schedule(path, text)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == INVALID_PARALLELISM


def test_invalid_parallelism_check_precedes_file_read(cli):
    # The file does not exist, yet the parallelism error wins and nothing
    # attempts to open it.
    missing = "/tmp/definitely-not-here-echo004-schedule.json"
    assert not os.path.exists(missing)
    result = cli.schedule(missing, "0")
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == INVALID_PARALLELISM


def test_invalid_parallelism_with_directory_path(cli, tmp_path):
    result = cli.schedule(tmp_path, "01")
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == INVALID_PARALLELISM


def test_invalid_parallelism_even_when_document_broken(cli, write_def):
    path = write_def(raw=b"not json")
    result = cli.schedule(path, "x")
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == INVALID_PARALLELISM


@pytest.mark.parametrize("text", INVALID_TEXTS)
def test_invalid_parallelism_inproc(write_def, text):
    path = str(write_def({"tasks": []}))
    rc, out, err = run_main("schedule", path, text)
    assert (rc, out, err) == (2, b"", INVALID_PARALLELISM)


def test_boundary_values(cli, write_def):
    path = write_def({"tasks": [{"id": "a"}]})
    ok = cli.schedule(path, "2147483647")
    assert ok.returncode == 0
    assert ok.stdout == b'{"batches":[["a"]]}\n'
    bad = cli.schedule(path, "2147483648")
    assert bad.returncode == 2
    assert bad.stdout == b""
    assert bad.stderr == INVALID_PARALLELISM


# ---------------------------------------------------------------------------
# Once parallelism is valid: definition diagnostics mirror plan exactly.
# ---------------------------------------------------------------------------

def diag(text: str) -> bytes:
    return (text + "\n").encode("utf-8")


def test_missing_file(cli):
    missing = "/tmp/definitely-not-here-echo004-schedule.json"
    assert not os.path.exists(missing)
    result = cli.schedule(missing, 1)
    assert result.returncode == 1
    assert result.stdout == b""
    assert result.stderr == diag(f"cannot read definition: {missing}")


def test_directory_not_readable(cli, tmp_path):
    result = cli.schedule(tmp_path, 2)
    assert result.returncode == 1
    assert result.stdout == b""
    assert result.stderr == diag(f"cannot read definition: {tmp_path}")


@pytest.mark.parametrize("raw", [b"", b"{", b"not json", b"\xff\xfe{"])
def test_invalid_json(cli, write_def, raw):
    path = write_def(raw=raw)
    result = cli.schedule(path, 3)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == diag("invalid json")


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"other": []},
        {"tasks": [], "extra": 1},
        {"tasks": None},
        {"tasks": [42]},
        {"tasks": [{"id": ""}]},
        {"tasks": [{"id": "a", "depends_on": "x"}]},
        {"tasks": [{"id": "a", "depends_on": [1]}]},
    ],
)
def test_invalid_definition(cli, write_def, document):
    path = write_def(document)
    result = cli.schedule(path, 5)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == diag("invalid definition")


def test_duplicate_task_id(cli, write_def):
    path = write_def({"tasks": [{"id": "中"}, {"id": "中"}]})
    result = cli.schedule(path, 1)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == diag("duplicate task id: 中")


def test_unknown_dependency(cli, write_def):
    path = write_def({"tasks": [{"id": "a", "depends_on": ["missing"]}]})
    result = cli.schedule(path, 1)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == diag("unknown dependency: a -> missing")


def test_self_dependency(cli, write_def):
    path = write_def({"tasks": [{"id": "a", "depends_on": ["a"]}]})
    result = cli.schedule(path, 1)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == diag("self dependency: a")


def test_cycle(cli, write_def):
    doc = {
        "tasks": [
            {"id": "a", "depends_on": ["b"]},
            {"id": "b", "depends_on": ["a"]},
        ]
    }
    result = cli.schedule(write_def(doc), 1)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == diag("cycle detected")


def test_graph_error_precedence_matches_plan(cli, write_def):
    # Same multi-problem graph as plan's precedence test: self a->a wins
    # over unknown a->x, duplicate b, and the c<->d cycle.
    doc = {
        "tasks": [
            {"id": "b"},
            {"id": "b"},
            {"id": "a", "depends_on": ["x", "a"]},
            {"id": "c", "depends_on": ["d"]},
            {"id": "d", "depends_on": ["c"]},
        ]
    }
    path = write_def(doc)
    for parallel in (1, 2, 2147483647):
        result = cli.schedule(path, parallel)
        assert result.returncode == 2
        assert result.stdout == b""
        assert result.stderr == diag("self dependency: a"), parallel


def test_definition_error_hash_seed_stable(cli, write_def):
    path = write_def(
        {
            "tasks": [
                {"id": "a", "depends_on": ["b"]},
                {"id": "b", "depends_on": ["a"]},
            ]
        }
    )
    for seed in ("0", "31", "2024"):
        result = cli.schedule(path, 4, env=make_env(PYTHONHASHSEED=seed))
        assert result.returncode == 2
        assert result.stdout == b""
        assert result.stderr == diag("cycle detected")


def test_help_carries_schedule_line():
    # The new line supplements (never reorders or rewrites) the old ones.
    lines = USAGE.splitlines()
    assert lines[0] == "usage: python3 -m workflow_orchestrator <command>"
    assert lines[2] == "commands:"
    assert lines[3].strip() == "version                  print the package version"
    assert lines[4].strip() == "help                     print this message"
    assert lines[5].strip() == (
        "plan <definition.json>   print static execution levels for a workflow"
    )
    assert any(
        line.strip() == "schedule <definition.json> <max-parallel>"
        for line in lines
    )
    assert any(
        "print deterministic scheduling batches" in line for line in lines
    )
    assert USAGE.endswith("\n")
    assert not USAGE.endswith("\n\n")


@pytest.mark.parametrize("text", ["\x00", "\x001", "\x00" + "1", "1" + "\x00"])
def test_parser_rejects_control_characters_directly(text):
    # NUL cannot be passed through a POSIX argv, so exercise the parser
    # itself; every control character is a non-digit and must be rejected.
    from workflow_orchestrator.schedule import _parse_parallelism, _InvalidParallelism

    with pytest.raises(_InvalidParallelism):
        _parse_parallelism(text)
