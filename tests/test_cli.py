"""Tests for the public command surface outside ``plan`` itself.

Covers ``version``, ``help``/``-h``/``--help``, no command (defaults to
help), unknown commands, and ``plan`` argument-count errors, asserting the
documented return values and which stream (stdout vs stderr) is used.

These are checked both end-to-end through ``python -m
workflow_orchestrator`` and in-process via ``main(argv)`` with captured
streams; both entries must agree.
"""
import io
import sys

import pytest

from conftest import make_env
from workflow_orchestrator import __version__
from workflow_orchestrator.__main__ import USAGE, main

USAGE_BYTES = USAGE.encode("utf-8")
VERSION_BYTES = (__version__ + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# In-process entry point (exercises main(argv) directly).
# ---------------------------------------------------------------------------

def run_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        # Pass [] explicitly (rather than None): None makes main read
        # sys.argv[1:], which under pytest contains the runner's own flags
        # and would not represent "no command". An empty list takes the
        # same default-to-help branch.
        rc = main(list(argv))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out.getvalue().encode("utf-8"), err.getvalue().encode("utf-8")


def test_main_version_inproc():
    rc, out, err = run_main("version")
    assert (rc, out, err) == (0, VERSION_BYTES, b"")


def test_main_help_variants_inproc():
    for argv in [("help",), ("-h",), ("--help",), ()]:
        rc, out, err = run_main(*argv)
        assert (rc, out, err) == (0, USAGE_BYTES, b""), argv


def test_main_unknown_command_inproc():
    rc, out, err = run_main("frobnicate")
    assert rc == 2
    assert out == b""
    assert err == b"unknown command: frobnicate\n" + USAGE_BYTES


def test_main_plan_wrong_argc_inproc():
    for argv in [("plan",), ("plan", "a", "b"), ("plan", "a", "b", "c")]:
        rc, out, err = run_main(*argv)
        assert (rc, out, err) == (2, b"", USAGE_BYTES), argv


# ---------------------------------------------------------------------------
# End-to-end through the documented public entry point.
# ---------------------------------------------------------------------------

def test_version_subprocess(cli):
    result = cli.run(["version"])
    assert result.returncode == 0
    assert result.stdout == VERSION_BYTES
    assert result.stderr == b""


@pytest.mark.parametrize("command", ["help", "-h", "--help"])
def test_help_subprocess(cli, command):
    result = cli.run([command])
    assert result.returncode == 0
    assert result.stdout == USAGE_BYTES
    assert result.stderr == b""


def test_no_command_prints_help_to_stdout(cli):
    result = cli.run([])
    assert result.returncode == 0
    assert result.stdout == USAGE_BYTES
    assert result.stderr == b""


def test_help_text_is_documented_spelling():
    # Guard the exact public help content/README-advertised command lines.
    assert USAGE.startswith(
        "usage: python3 -m workflow_orchestrator <command>\n"
    )
    for line in (
        "  version                  print the package version",
        "  help                     print this message",
        "  plan <definition.json>   print static execution levels",
    ):
        assert line in USAGE
    # No trailing newline beyond the message's own final newline-less print:
    # __main__ prints with end="", so USAGE itself ends right after the
    # closing newline.
    assert USAGE.endswith("\n")
    assert not USAGE.endswith("\n\n")


def test_unknown_command_stderr_and_exit_code(cli):
    result = cli.run(["nope"])
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"unknown command: nope\n" + USAGE_BYTES


def test_unknown_command_echoed_verbatim(cli):
    result = cli.run(["--bogus"])
    assert result.returncode == 2
    assert result.stderr.startswith(b"unknown command: --bogus\n")


def test_plan_without_argument(cli):
    result = cli.run(["plan"])
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == USAGE_BYTES


def test_plan_with_too_many_arguments(cli):
    result = cli.run(["plan", "a.json", "b.json"])
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == USAGE_BYTES


def test_commands_share_return_codes_across_hash_seeds(cli, write_def):
    path = write_def({"tasks": []})
    env = make_env(PYTHONHASHSEED="555")
    version = cli.run(["version"], env=env)
    bad_plan = cli.run(["plan"], env=env)
    good_plan = cli.plan(path, env=env)
    assert version.returncode == 0
    assert bad_plan.returncode == 2
    assert good_plan.returncode == 0
