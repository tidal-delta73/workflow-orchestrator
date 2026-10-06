"""White-box regression tests for the shared graph core and its boundaries.

These pin the architecture the refactor introduced:

* loading is four distinct stages (read bytes / decode JSON / parse the
  definition structure / check graph semantics), each testable alone;
* the core expresses failure as a :class:`~workflow_orchestrator.graph.LoadError`
  with an exit code and message -- it never writes to stdout or stderr;
* success yields one read-only, normalized
  :class:`~workflow_orchestrator.graph.Graph` (tasks, de-duplicated
  dependencies, reverse dependencies, all in Unicode code point order);
* ``plan`` and ``schedule`` consume the ``Graph``, never raw entries or
  JSON, and ``schedule`` no longer reaches into plan's private loader;
* all graph walks stay iterative on a chain far beyond the recursion limit.
"""
import inspect
import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, make_env
from workflow_orchestrator import graph as graph_module
from workflow_orchestrator import plan as plan_module
from workflow_orchestrator import schedule as schedule_module
from workflow_orchestrator.graph import Graph, LoadError


def build_from_document(document):
    """Run stages 3 and 4 over an already-decoded JSON value."""
    return graph_module.build_graph(graph_module.parse_definition(document))


# ---------------------------------------------------------------------------
# Stage 1: file reading.
# ---------------------------------------------------------------------------


def test_read_bytes_returns_file_contents(tmp_path):
    path = tmp_path / "d.json"
    path.write_bytes(b'{"tasks":[]}')
    assert graph_module.read_bytes(str(path)) == b'{"tasks":[]}'


@pytest.mark.parametrize("use_path", ["missing", "directory"])
def test_read_bytes_failure_is_load_error_code_1(tmp_path, use_path):
    target = tmp_path / "no-such-file.json" if use_path == "missing" else tmp_path
    with pytest.raises(LoadError) as excinfo:
        graph_module.read_bytes(str(target))
    err = excinfo.value
    assert err.exit_code == 1
    assert err.message == f"cannot read definition: {target}"


# ---------------------------------------------------------------------------
# Stage 2: JSON decoding is independent of reading.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [b"", b"{", b"not json", b"\xff\xfe{bogus}", b"'single quotes'"],
)
def test_decode_document_invalid_json(raw):
    with pytest.raises(LoadError) as excinfo:
        graph_module.decode_document(raw)
    assert excinfo.value.exit_code == 2
    assert excinfo.value.message == "invalid json"


def test_decode_document_makes_no_structural_assumptions():
    # decode_document accepts structurally odd JSON; later stages reject it.
    assert graph_module.decode_document(b'{"tasks":[]}') == {"tasks": []}
    assert graph_module.decode_document(b"42") == 42


# ---------------------------------------------------------------------------
# Stage 3: structural validation is independent of graph semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [
        None,
        42,
        "a string",
        [],
        {},
        {"other": []},
        {"tasks": [], "extra": 1},
        {"tasks": None},
        {"tasks": {}},
        {"tasks": [None]},
        {"tasks": ["a"]},
        {"tasks": [42]},
        {"tasks": [{"id": ""}]},
        {"tasks": [{"id": 42}]},
        {"tasks": [{"id": "a", "x": 1}]},
        {"tasks": [{"id": "a", "depends_on": "x"}]},
        {"tasks": [{"id": "a", "depends_on": [1]}]},
        {"tasks": [{"id": "a", "depends_on": [None]}]},
    ],
)
def test_parse_definition_rejects_structure(document):
    with pytest.raises(LoadError) as excinfo:
        graph_module.parse_definition(document)
    assert excinfo.value.exit_code == 2
    assert excinfo.value.message == "invalid definition"


def test_parse_definition_keeps_document_order_and_duplicates():
    document = {
        "tasks": [
            {"id": "b", "depends_on": ["a", "a"]},
            {"id": "a"},
        ]
    }
    entries = graph_module.parse_definition(document)
    # Structure stage does not dedupe or reorder; normalization comes later.
    assert entries == [("b", ["a", "a"]), ("a", [])]


def test_structural_failure_precedes_graph_semantics():
    # Duplicate ids and an unknown dep are present, but a structurally bad
    # task must be reported as "invalid definition" by the structure stage.
    document = {
        "tasks": [
            {"id": "a"},
            {"id": "a"},
            {"id": "b", "depends_on": ["ghost"]},
            {"id": 123},
        ]
    }
    with pytest.raises(LoadError) as excinfo:
        build_from_document(document)
    assert excinfo.value.message == "invalid definition"


# ---------------------------------------------------------------------------
# Stage 4: graph semantics -> the single LoadError, by Unicode order.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entries,message",
    [
        ([("a", []), ("a", [])], "duplicate task id: a"),
        ([("a", ["a"])], "self dependency: a"),
        ([("a", ["missing"])], "unknown dependency: a -> missing"),
        ([("a", ["b"]), ("b", ["a"])], "cycle detected"),
    ],
)
def test_build_graph_reports_each_graph_error(entries, message):
    with pytest.raises(LoadError) as excinfo:
        graph_module.build_graph(entries)
    assert excinfo.value.exit_code == 2
    assert excinfo.value.message == message


def test_build_graph_multi_problem_picks_unicode_smallest_pair():
    # dup b; self a->a; unknown a->x; cycle c<->d -> self (a,a) wins.
    entries = [
        ("b", []),
        ("b", []),
        ("a", ["x", "a"]),
        ("c", ["d"]),
        ("d", ["c"]),
    ]
    with pytest.raises(LoadError) as excinfo:
        graph_module.build_graph(entries)
    assert excinfo.value.message == "self dependency: a"


# ---------------------------------------------------------------------------
# The core never touches output streams.
# ---------------------------------------------------------------------------


class _BoomStream:
    def write(self, *args, **kwargs):
        raise AssertionError("core code must not write to a stream")

    def flush(self):
        raise AssertionError("core code must not flush a stream")


def test_core_never_writes_to_streams(tmp_path, monkeypatch):
    missing = tmp_path / "gone.json"
    bad_json = tmp_path / "bad.json"
    bad_json.write_bytes(b"{")
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"tasks": [{"id": 1}]}), encoding="utf-8")
    cyclic = tmp_path / "cycle.json"
    cyclic.write_text(
        json.dumps(
            {"tasks": [
                {"id": "a", "depends_on": ["b"]},
                {"id": "b", "depends_on": ["a"]},
            ]}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "stdout", _BoomStream())
    monkeypatch.setattr(sys, "stderr", _BoomStream())

    for path in (missing, bad_json, malformed, cyclic):
        with pytest.raises(LoadError):
            graph_module.load_graph(str(path))

    # A successful load stays silent too.
    ok = tmp_path / "ok.json"
    ok.write_text('{"tasks":[{"id":"a"}]}', encoding="utf-8")
    g = graph_module.load_graph(str(ok))
    assert g.tasks == ("a",)


def test_graph_module_never_references_streams_or_print():
    # Boundary by code, not by comments/docstrings: no stdout/stderr/print
    # references and no sys import anywhere in the core module.
    import ast

    tree = ast.parse(inspect.getsource(graph_module))
    forbidden_names = {"stdout", "stderr", "print"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            pytest.fail(f"core references {node.id} at line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr in ("stdout", "stderr"):
            pytest.fail(f"core references .{node.attr} at line {node.lineno}")
        if isinstance(node, ast.Import):
            assert all(alias.name != "sys" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "sys"


# ---------------------------------------------------------------------------
# The normalized Graph: sorted, de-duplicated, reverse edges, read-only.
# ---------------------------------------------------------------------------


def test_graph_normalizes_deduplicates_and_reverses():
    entries = [
        ("t", ["x", "y", "x", "y"]),  # duplicates count once
        ("y", ["s", "s"]),
        ("x", ["s"]),
        ("s", []),
        ("中", []),
    ]
    g = graph_module.build_graph(entries)

    assert g.tasks == ("s", "t", "x", "y", "中")  # Unicode code point order
    assert g.dependencies["t"] == ("x", "y")
    assert g.dependencies["y"] == ("s",)
    assert g.dependencies["s"] == ()
    # Reverse dependencies: every successor, de-duplicated, code point order.
    assert g.dependents["s"] == ("x", "y")
    assert g.dependents["x"] == ("t",)
    assert g.dependents["y"] == ("t",)
    assert g.dependents["t"] == ()
    assert g.dependents["中"] == ()
    # Every task is a key in both views.
    assert set(g.dependencies) == set(g.tasks)
    assert set(g.dependents) == set(g.tasks)


def test_graph_values_are_tuples():
    g = graph_module.build_graph([("a", []), ("b", ["a", "a"])])
    assert isinstance(g.tasks, tuple)
    assert isinstance(g.dependencies["b"], tuple)
    assert isinstance(g.dependents["a"], tuple)


def test_graph_attributes_are_read_only():
    g = graph_module.build_graph([("a", [])])

    with pytest.raises(AttributeError):
        g.tasks = ("x",)
    with pytest.raises(AttributeError):
        del g.tasks
    # The mapping views cannot be mutated.
    with pytest.raises(TypeError):
        g.dependencies["a"] = ("x",)
    with pytest.raises(TypeError):
        g.dependents["a"] = ("x",)
    with pytest.raises(TypeError):
        del g.dependencies["a"]
    with pytest.raises(AttributeError):
        g.dependencies.clear()
    # The tuple values cannot be mutated.
    with pytest.raises(AttributeError):
        g.tasks.append("x")
    with pytest.raises(AttributeError):
        g.dependencies["a"].append("x")


def test_graph_is_independent_of_listing_order():
    canonical = graph_module.build_graph(
        [
            ("t", ["y", "x"]),
            ("y", ["s"]),
            ("x", ["s"]),
            ("s", []),
        ]
    )
    g = graph_module.build_graph(
        [
            ("s", []),
            ("x", ["s"]),
            ("t", ["x", "y", "x"]),  # plus a duplicate dependency
            ("y", ["s"]),
        ]
    )
    assert g.tasks == canonical.tasks
    assert dict(g.dependencies) == dict(canonical.dependencies)
    assert dict(g.dependents) == dict(canonical.dependents)


def test_graph_normalization_stable_across_hash_seeds():
    # Sets/dicts drive normalization; the serialized graph must be identical
    # under fixed and random hash seeds.
    script = (
        "import json, sys;"
        "from workflow_orchestrator.graph import build_graph;"
        "entries = [('t',['y','x','x']),('y',['s']),('x',['s']),('s',[])];"
        "g = build_graph(entries);"
        "out = (list(g.tasks),"
        " {k: list(v) for k, v in g.dependencies.items()},"
        " {k: list(v) for k, v in g.dependents.items()});"
        "sys.stdout.write(json.dumps(out, ensure_ascii=False, sort_keys=True))"
    )
    observed = set()
    for seed in ("0", "31", "2024", None):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=make_env(PYTHONHASHSEED=seed),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        observed.add(proc.stdout)
    assert len(observed) == 1
    assert json.loads(next(iter(observed)))[0] == ["s", "t", "x", "y"]


# ---------------------------------------------------------------------------
# Commands consume the Graph; neither re-parses JSON nor shares a loader.
# ---------------------------------------------------------------------------


def test_plan_and_schedule_have_no_private_loader():
    # The old plan-owned loader and schedule's back-door into plan are gone.
    assert not hasattr(plan_module, "_load_entries")
    assert not hasattr(plan_module, "_read_document")
    assert not hasattr(plan_module, "_parse_tasks")
    assert not hasattr(schedule_module, "plan_module")
    schedule_source = inspect.getsource(schedule_module)
    assert "from . import plan" not in schedule_source
    assert "_load_entries" not in schedule_source


def test_planners_require_a_graph_not_raw_entries():
    raw_entries = [("b", ["a"]), ("a", [])]
    # Raw JSON-derived tuples must not be accepted in place of a Graph.
    with pytest.raises((AttributeError, TypeError, KeyError)):
        plan_module._build_levels(raw_entries)
    with pytest.raises((AttributeError, TypeError, KeyError)):
        schedule_module._build_batches(raw_entries, 1)


def test_both_planners_consume_the_same_graph(tmp_path):
    # Both commands produce their results from the same normalized object;
    # a duplicate-listing input is already collapsed when they see it.
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps(
            {"tasks": [
                {"id": "b", "depends_on": ["a", "a"]},
                {"id": "a"},
            ]}
        ),
        encoding="utf-8",
    )
    g = graph_module.load_graph(str(path))
    assert isinstance(g, Graph)
    assert g.dependencies["b"] == ("a",)
    assert plan_module._build_levels(g) == [["a"], ["b"]]
    assert schedule_module._build_batches(g, 5) == [["a"], ["b"]]


# ---------------------------------------------------------------------------
# Iteration, not recursion: a chain beyond the recursion limit.
# ---------------------------------------------------------------------------


def test_deep_chain_loads_and_plans_without_recursion():
    n = 3500
    assert n > 3 * sys.getrecursionlimit() // 2
    entries = [
        (f"n{i:05d}", [f"n{i - 1:05d}"] if i else [])
        for i in range(n)
    ]
    g = graph_module.build_graph(entries)  # cycle detection walks deeply too
    assert len(g.tasks) == n

    levels = plan_module._build_levels(g)
    assert len(levels) == n
    assert [level[0] for level in levels] == list(g.tasks)

    batches = schedule_module._build_batches(g, 1)
    assert len(batches) == n
    assert [batch[0] for batch in batches] == list(g.tasks)
