"""Shared helpers for the regression tests.

Every test enters the product the same way a user does: a definition file on
disk passed through the public ``python -m workflow_orchestrator`` command
entry. :func:`cli` exercises the real process boundary (exit status, stdout
and stderr bytes); :func:`main_plan` calls the public ``main`` entry point
in-process with redirected streams so that exhaustive permutation checks
stay fast while still going through definition-file -> command -> output.
"""
import contextlib
import io
import itertools
import json
import os
import subprocess
import sys

from workflow_orchestrator.__main__ import main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_json(tmpdir, document, name="definition.json"):
    """Write a definition document as UTF-8 JSON bytes and return its path."""
    path = os.path.join(tmpdir, name)
    with open(path, "wb") as handle:
        handle.write(json.dumps(document, ensure_ascii=False).encode("utf-8"))
    return path


def write_raw(tmpdir, raw, name="definition.json"):
    """Write raw definition bytes and return the path."""
    path = os.path.join(tmpdir, name)
    with open(path, "wb") as handle:
        handle.write(raw)
    return path


def cli(*args, env=None, cwd=REPO_ROOT):
    """Run the public module entry point in a fresh interpreter."""
    full_env = dict(os.environ)
    parent = full_env.get("PYTHONPATH", "")
    full_env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + parent if parent else "")
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "workflow_orchestrator", *args],
        cwd=cwd,
        env=full_env,
        capture_output=True,
    )


def main_plan(path):
    """Invoke ``main(["plan", path])`` and return ``(returncode, out, err)``."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        returncode = main(["plan", path])
    return returncode, stdout.getvalue(), stderr.getvalue()


def index_orders(n):
    """A fixed, exhaustive set of index permutations for small n.

    Exhaustive for up to six elements (720 orderings); larger sequences use
    a fixed list of systematic reorderings. Nothing here depends on random
    seeds, time, or filesystem ordering.
    """
    if n <= 6:
        return list(itertools.permutations(range(n)))
    seq = list(range(n))
    variants = [
        seq,
        list(reversed(seq)),
        seq[1:] + seq[:1],
        seq[-1:] + seq[:-1],
        [seq[-1], *seq[1:-1], seq[0]],
        seq[::2] + seq[1::2],
        seq[n // 2:] + seq[: n // 2],
    ]
    unique = []
    for variant in variants:
        if variant not in unique:
            unique.append(variant)
    return [tuple(v) for v in unique]


def task_permutations(document):
    """Yield the document with every systematic task ordering."""
    tasks = document["tasks"]
    for order in index_orders(len(tasks)):
        yield {"tasks": [tasks[i] for i in order]}


def dependency_variants(dependencies):
    """All orderings of a dependency list plus duplicated-entry variants."""
    deps = list(dependencies)
    variants = [tuple(deps[i] for i in order) for order in index_orders(len(deps))]
    if deps:  # duplicate the first dependency at start, middle, and end
        duplicated = deps[0]
        insertion_points = {0, len(deps) // 2, len(deps)}
        for point in insertion_points:
            variants.append(tuple(deps[:point] + [duplicated] + deps[point:]))
    seen = set()
    result = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            result.append(list(variant))
    return result


def with_dependencies(document, task_index, new_deps):
    """Copy a document replacing one task's depends_on list."""
    tasks = [dict(task) for task in document["tasks"]]
    tasks[task_index] = {**tasks[task_index], "depends_on": list(new_deps)}
    return {"tasks": tasks}


# Fixed valid graphs together with their documented compact plan output.
VALID_GRAPHS = {
    "empty": (
        {"tasks": []},
        '{"levels":[]}\n',
    ),
    "single": (
        {"tasks": [{"id": "n"}]},
        '{"levels":[["n"]]}\n',
    ),
    "multi_layer_diamond": (
        {"tasks": [
            {"id": "build"},
            {"id": "test", "depends_on": ["build"]},
            {"id": "lint", "depends_on": ["build"]},
            {"id": "deploy", "depends_on": ["test", "lint"]},
        ]},
        '{"levels":[["build"],["lint","test"],["deploy"]]}\n',
    ),
    "deepest_dependency_rule": (
        {"tasks": [
            {"id": "a"},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["a"]},
            {"id": "d", "depends_on": ["c", "a"]},
            {"id": "e", "depends_on": ["d", "b"]},
        ]},
        '{"levels":[["a"],["b","c"],["d"],["e"]]}\n',
    ),
    "shared_dependency": (
        {"tasks": [
            {"id": "x", "depends_on": ["r2", "r1"]},
            {"id": "r1"},
            {"id": "r2"},
        ]},
        '{"levels":[["r1","r2"],["x"]]}\n',
    ),
    "unicode_same_level": (
        {"tasks": [
            {"id": "中"},
            {"id": "a"},
            {"id": "aa"},
            {"id": "あ"},
            {"id": "é"},
        ]},
        '{"levels":[["a","aa","é","あ","中"]]}\n',
    ),
    "complex_six_nodes": (
        {"tasks": [
            {"id": "a"},
            {"id": "b"},
            {"id": "c", "depends_on": ["a", "b"]},
            {"id": "d", "depends_on": ["a"]},
            {"id": "e", "depends_on": ["c", "d", "a"]},
            {"id": "f", "depends_on": ["e", "b"]},
        ]},
        '{"levels":[["a","b"],["c","d"],["e"],["f"]]}\n',
    ),
    "duplicate_dependency_entry": (
        {"tasks": [
            {"id": "b", "depends_on": ["a", "a"]},
            {"id": "a"},
        ]},
        '{"levels":[["a"],["b"]]}\n',
    ),
}

USAGE = (
    "usage: python3 -m workflow_orchestrator <command>\n"
    "\n"
    "commands:\n"
    "  version                  print the package version\n"
    "  help                     print this message\n"
    "  plan <definition.json>   print static execution levels for a workflow\n"
)
USAGE_BYTES = USAGE.encode("utf-8")
VERSION_BYTES = b"0.1.0\n"
