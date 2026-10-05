"""Shared black-box test infrastructure.

Tests enter through the documented public entry point
(``python -m workflow_orchestrator``) with definition files on disk, and
assert on the process exit status and raw stdout/stderr bytes.

The subprocess environment is whitelisted and pinned so results cannot
depend on the invoking machine's locale, Python hash seed, or directory
ordering:

* ``LC_ALL=LANG=C`` with UTF-8 mode / IO encoding forced on (a test-time
  configuration; the installed library keeps zero runtime dependencies),
* a fixed ``PYTHONHASHSEED`` by default, with dedicated tests varying it,
* ``cwd`` and ``PYTHONPATH`` pinned at the repository root.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", "/tmp"),
    "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    "LC_ALL": "C",
    "LANG": "C",
    # Pin the standard stream encoding to UTF-8 on every supported Python
    # regardless of the C locale above.
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONHASHSEED": "0",
    "PYTHONPATH": REPO_ROOT,
}


def make_env(**overrides):
    """A subprocess environment based on BASE_ENV.

    A value of None removes the key entirely (e.g. PYTHONHASHSEED=None to
    let the interpreter pick a random seed).
    """
    env = dict(BASE_ENV)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


class CliRunner:
    """Run the public CLI as a subprocess and capture raw byte streams."""

    def run(self, argv, *, env=None, timeout=120):
        return subprocess.run(
            [sys.executable, "-m", "workflow_orchestrator", *argv],
            cwd=REPO_ROOT,
            env=BASE_ENV if env is None else env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

    def plan(self, path, *, env=None, timeout=120):
        return self.run(["plan", str(path)], env=env, timeout=timeout)

    def plan_many(self, paths, *, env=None, timeout=120, workers=8):
        """Run plan over many definition files concurrently, in order.

        Subprocess startup dominates these cases; waiting on child processes
        releases the GIL, so a thread pool keeps the suite fast without any
        shared state between runs.
        """
        paths = list(paths)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(
                    lambda p: self.plan(p, env=env, timeout=timeout), paths
                )
            )


def render_levels(levels) -> bytes:
    """The exact compact success-output byte contract (plus newline)."""
    return (
        json.dumps(
            {"levels": levels}, ensure_ascii=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


@pytest.fixture(scope="session")
def cli():
    return CliRunner()


@pytest.fixture
def write_def(tmp_path):
    """Write a definition document (or raw bytes) and return its path."""
    counter = 0

    def _write(doc=None, *, raw=None, name=None, indent=False):
        nonlocal counter
        path = tmp_path / (name or f"definition-{counter}.json")
        counter += 1
        if raw is not None:
            path.write_bytes(raw)
        else:
            text = json.dumps(doc, ensure_ascii=False, indent=2 if indent else None)
            path.write_text(text, encoding="utf-8")
        return path

    return _write


def assert_success(result, expected_stdout: bytes):
    """Assert the full success contract: rc 0, exact stdout, empty stderr."""
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected_stdout
    assert result.stderr == b""


def assert_failure(result, expected_stderr: bytes, code=2):
    """Assert the full failure contract: given rc, empty stdout, one diagnostic."""
    assert result.returncode == code
    assert result.stdout == b"", "failure path must write nothing to stdout"
    assert result.stderr == expected_stderr
    # Every documented failure reports exactly one line on stderr.
    assert expected_stderr.endswith(b"\n")
    assert expected_stderr.count(b"\n") == 1
