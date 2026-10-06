"""Regression tests for the shared graph core and its command boundary.

The black-box suites elsewhere pin the public byte/exit-code contract. These
tests pin the refactored internals:

* the four load layers (file read / JSON decode / structure / graph
  semantics) each raise their own typed failure with the fixed exit code and
  diagnostic, and never touch stdout/stderr,
* ``ValidGraph`` is read-only, de-duplicated, and Unicode ordered, and is
  equal across equivalent listings,
* ``plan`` and ``schedule`` get *equivalent* error behavior from that one
  core and neither command re-parses JSON or reaches into the other
  command's private loader,
* deep-DAG processing stays iterative.
"""
import io
import inspect
import pathlib
import sys

import pytest

from variants import equivalent_documents, spread
from workflow_orchestrator import graph as graph_module
from workflow_orchestrator.graph import (
    CannotRead,
    GraphError,
    GraphLoadError,
    InvalidDefinition,
    InvalidJson,
    ValidGraph,
    build_graph,
    decode_json,
    load_graph,
    parse_definition,
    read_file,
    select_graph_error,
)
from workflow_orchestrator import plan as plan_module
from workflow_orchestrator import schedule as schedule_module

PACKAGE_DIR = pathlib.Path(graph_module.__file__).parent


def load_graph_from(document) -> ValidGraph:
    return build_graph(parse_definition(document))


# ---------------------------------------------------------------------------
# Layer 1: file reading.
# ---------------------------------------------------------------------------

def test_read_file_returns_bytes(tmp_path):
    path = tmp_path / "d.json"
    path.write_bytes(b'{"tasks":[]}')
    assert read_file(str(path)) == b'{"tasks":[]}'


def test_read_file_missing_raises_cannot_read(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(CannotRead) as excinfo:
        read_file(str(missing))
    error = excinfo.value
    assert isinstance(error, GraphLoadError)
    assert error.code == 1
    assert error.diagnostic == f"cannot read definition: {missing}"


def test_read_file_directory_raises_cannot_read(tmp_path):
    with pytest.raises(CannotRead):
        read_file(str(tmp_path))


# ---------------------------------------------------------------------------
# Layer 2: JSON decoding.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [b"", b"{", b"not json", b"\xff\xfe{"])
def test_decode_json_rejects_bad_bytes(raw):
    with pytest.raises(InvalidJson) as excinfo:
        decode_json(raw)
    assert excinfo.value.code == 2
    assert excinfo.value.diagnostic == "invalid json"


def test_decode_json_returns_value():
    assert decode_json(b'{"tasks": []}') == {"tasks": []}


# ---------------------------------------------------------------------------
# Layer 3: definition structure.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "document",
    [
        None,
        42,
        "x",
        [],
        {},
        {"other": []},
        {"tasks": [], "x": 1},
        {"tasks": {}},
        {"tasks": [42]},
        {"tasks": [{"id": ""}]},
        {"tasks": [{"id": 1}]},
        {"tasks": [{"id": "a", "depends_on": "x"}]},
        {"tasks": [{"id": "a", "depends_on": [1]}]},
        {"tasks": [{"id": "a", "x": 1}]},
    ],
)
def test_parse_definition_rejects_shape(document):
    with pytest.raises(InvalidDefinition) as excinfo:
        parse_definition(document)
    assert excinfo.value.diagnostic == "invalid definition"


def test_parse_definition_keeps_raw_entries_including_duplicates():
    # Structure validation deliberately keeps duplicate tasks and repeated
    # deps; semantic normalization is the next layer's job.
    entries = parse_definition(
        {
            "tasks": [
                {"id": "b", "depends_on": ["a", "a"]},
                {"id": "a"},
                {"id": "a"},
            ]
        }
    )
    assert entries == [("b", ["a", "a"]), ("a", []), ("a", [])]


# ---------------------------------------------------------------------------
# Layer 4: graph semantics and deterministic diagnostic selection.
# ---------------------------------------------------------------------------

def test_select_graph_error_none_when_valid():
    assert select_graph_error(parse_definition({"tasks": [{"id": "a"}]})) is None


@pytest.mark.parametrize(
    "document,message",
    [
        ({"tasks": [{"id": "a"}, {"id": "a"}]}, "duplicate task id: a"),
        ({"tasks": [{"id": "a", "depends_on": ["a"]}]},
         "self dependency: a"),
        ({"tasks": [{"id": "a", "depends_on": ["q"]}]},
         "unknown dependency: a -> q"),
        (
            {
                "tasks": [
                    {"id": "a", "depends_on": ["b"]},
                    {"id": "b", "depends_on": ["a"]},
                ]
            },
            "cycle detected",
        ),
    ],
)
def test_build_graph_raises_graph_error(document, message):
    with pytest.raises(GraphError) as excinfo:
        build_graph(parse_definition(document))
    error = excinfo.value
    assert error.code == 2
    assert error.diagnostic == message


def test_graph_error_selection_unicode_order():
    # dup b; self a->a; unknown a->x; cycle c<->d -> self (a,a) wins.
    document = {
        "tasks": [
            {"id": "b"},
            {"id": "b"},
            {"id": "a", "depends_on": ["x", "a"]},
            {"id": "c", "depends_on": ["d"]},
            {"id": "d", "depends_on": ["c"]},
        ]
    }
    assert select_graph_error(parse_definition(document)) == "self dependency: a"


# ---------------------------------------------------------------------------
# ValidGraph: normalized, de-duplicated, Unicode ordered, read-only.
# ---------------------------------------------------------------------------

def test_valid_graph_normalizes_tasks_and_relations():
    graph = load_graph_from(
        {
            "tasks": [
                {"id": "c", "depends_on": ["a", "a"]},
                {"id": "中"},
                {"id": "a"},
            ]
        }
    )
    # Tasks sorted by Unicode code point (a=97 < c=99 < 中=20013).
    assert graph.tasks == ("a", "c", "中")
    # Repeated dependency counted once; relations are tuples.
    assert isinstance(graph.dependencies["c"], tuple)
    assert graph.dependencies["c"] == ("a",)
    assert graph.dependencies["a"] == ()
    # Reverse dependency populated and Unicode ordered.
    assert graph.dependents["a"] == ("c",)
    assert graph.dependents["c"] == ()


def test_valid_graph_relations_are_immutable():
    graph = load_graph_from(
        {"tasks": [{"id": "a", "depends_on": ["b"]}, {"id": "b"}]}
    )
    assert isinstance(graph, ValidGraph)
    assert isinstance(graph.tasks, tuple)
    with pytest.raises(TypeError):
        graph.dependencies["a"] = ("z",)
    with pytest.raises(TypeError):
        graph.dependents["b"] = ("z",)
    with pytest.raises(TypeError):
        graph.dependencies["new"] = ()
    with pytest.raises(TypeError):
        del graph.dependents["b"]


def test_valid_graph_identical_across_equivalent_listings(write_def):
    tasks = [
        {"id": "a"},
        {"id": "b", "depends_on": ["a"]},
        {"id": "c", "depends_on": ["a", "b"]},
    ]
    docs = spread(equivalent_documents(tasks, duplicate_deps=True), 24)
    paths = [write_def(doc) for doc in docs]
    reference = load_graph(str(paths[0]))
    for path in paths[1:]:
        assert load_graph(str(path)) == reference


# ---------------------------------------------------------------------------
# The core never performs stream I/O.
# ---------------------------------------------------------------------------

def test_core_success_writes_nothing(capsys, write_def):
    path = write_def({"tasks": [{"id": "a"}]})
    graph = load_graph(str(path))
    assert graph.tasks == ("a",)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_core_failure_writes_nothing(capsys, write_def):
    path = write_def({"tasks": [{"id": "a"}, {"id": "a"}]})
    with pytest.raises(GraphError):
        load_graph(str(path))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_core_layer_functions_have_no_stream_parameters():
    # The boundary contract: core functions take data/paths, never streams.
    for fn in (read_file, decode_json, parse_definition, build_graph,
               select_graph_error, load_graph):
        params = set(inspect.signature(fn).parameters)
        assert "stdout" not in params and "stderr" not in params, fn.__name__


# ---------------------------------------------------------------------------
# plan and schedule consume the same core and report equivalent failures.
# ---------------------------------------------------------------------------

ERROR_DOCUMENTS = [
    ("invalid json", "raw", b"{not json", b"invalid json\n"),
    ("invalid definition", "doc",
     {"tasks": [{"id": "a", "extra": 1}]}, b"invalid definition\n"),
    ("duplicate", "doc",
     {"tasks": [{"id": "a"}, {"id": "a"}]}, b"duplicate task id: a\n"),
    ("self", "doc",
     {"tasks": [{"id": "a", "depends_on": ["a"]}]},
     b"self dependency: a\n"),
    ("unknown", "doc",
     {"tasks": [{"id": "a", "depends_on": ["q"]}]},
     b"unknown dependency: a -> q\n"),
    ("cycle", "doc",
     {"tasks": [
         {"id": "a", "depends_on": ["b"]},
         {"id": "b", "depends_on": ["a"]},
     ]}, b"cycle detected\n"),
]


@pytest.mark.parametrize("label,kind,payload,expected", ERROR_DOCUMENTS)
def test_plan_and_schedule_errors_are_equivalent(
    cli, write_def, label, kind, payload, expected
):
    path = write_def(raw=payload) if kind == "raw" else write_def(payload)
    plan_result = cli.plan(path)
    schedule_result = cli.schedule(path, 4)
    for result in (plan_result, schedule_result):
        assert result.returncode == 2, label
        assert result.stdout == b"", label
        assert result.stderr == expected, label
    # The two commands agree byte for byte on the shared diagnostic.
    assert plan_result.stderr == schedule_result.stderr


def test_plan_and_schedule_cannot_read_are_equivalent(cli, tmp_path):
    missing = tmp_path / "missing.json"
    expected = f"cannot read definition: {missing}\n".encode("utf-8")
    plan_result = cli.plan(missing)
    schedule_result = cli.schedule(missing, 4)
    for result in (plan_result, schedule_result):
        assert result.returncode == 1
        assert result.stdout == b""
        assert result.stderr == expected
    assert plan_result.stderr == schedule_result.stderr


def test_both_commands_consume_one_normalized_graph(write_def):
    # Build the shared graph once; both command analyses derive from it and
    # never see raw entries.
    path = write_def(
        {
            "tasks": [
                {"id": "c", "depends_on": ["a", "b", "a"]},
                {"id": "b", "depends_on": ["a"]},
                {"id": "a"},
            ]
        }
    )
    graph = load_graph(str(path))
    assert graph.dependencies["c"] == ("a", "b")
    assert plan_module.build_levels(graph) == [["a"], ["b"], ["c"]]
    assert schedule_module.build_batches(graph, 10) == [["a"], ["b"], ["c"]]


# ---------------------------------------------------------------------------
# Structural coupling guards: one loader, no cross-command private coupling.
# ---------------------------------------------------------------------------

def test_only_graph_core_decodes_json():
    # json.loads must appear only in the shared core; the commands render
    # JSON but must not decode the definition themselves.
    for module_path in sorted(PACKAGE_DIR.glob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        if module_path.name == "graph.py":
            assert "json.loads" in source
        else:
            assert "json.loads" not in source, module_path.name


def test_schedule_does_not_couple_to_plan_private_loader():
    schedule_source = (PACKAGE_DIR / "schedule.py").read_text(encoding="utf-8")
    assert "_load_entries" not in schedule_source
    assert "import plan" not in schedule_source
    # The old private coupling point is gone from plan as well.
    assert not hasattr(plan_module, "_load_entries")
    # Both commands now load through the one shared core function.
    assert schedule_module.load_graph is graph_module.load_graph
    assert plan_module.load_graph is graph_module.load_graph


def test_command_runs_render_failure_to_stream_and_code(write_def):
    path = write_def({"tasks": [{"id": "a"}, {"id": "a"}]})
    out, err = io.StringIO(), io.StringIO()
    assert plan_module.run(str(path), out, err) == 2
    assert out.getvalue() == ""
    assert err.getvalue() == "duplicate task id: a\n"

    out2, err2 = io.StringIO(), io.StringIO()
    assert schedule_module.run(str(path), 3, out2, err2) == 2
    assert out2.getvalue() == ""
    assert err2.getvalue() == "duplicate task id: a\n"


def test_command_runs_success_end_with_single_newline(write_def):
    path = write_def({"tasks": [{"id": "a"}]})
    out, err = io.StringIO(), io.StringIO()
    assert plan_module.run(str(path), out, err) == 0
    assert out.getvalue() == '{"levels":[["a"]]}\n'
    assert err.getvalue() == ""

    out2, err2 = io.StringIO(), io.StringIO()
    assert schedule_module.run(str(path), 2, out2, err2) == 0
    text = out2.getvalue()
    assert text == '{"batches":[["a"]]}\n'
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert err2.getvalue() == ""


# ---------------------------------------------------------------------------
# Deep DAGs stay iterative through the shared core.
# ---------------------------------------------------------------------------

def test_deep_chain_is_handled_without_recursion():
    # Well beyond Python's default recursion limit of 1000: a recursive
    # traversal in validation, level building, or batching would overflow.
    n = 4000
    entries = [("n0", [])]
    entries += [(f"n{i}", [f"n{i - 1}"]) for i in range(1, n)]

    old_limit = sys.getrecursionlimit()
    try:
        # A deliberately tight limit proves the graph-depth processing does
        # not recurse; the shallow pytest call frames stay comfortably above
        # it, while any depth-recursive walk would raise RecursionError.
        sys.setrecursionlimit(500)
        graph = build_graph(entries)
        levels = plan_module.build_levels(graph)
        batches = schedule_module.build_batches(graph, 1)
    finally:
        sys.setrecursionlimit(old_limit)

    assert len(levels) == n
    assert levels[0] == ["n0"]
    assert levels[n - 1] == [f"n{n - 1}"]
    assert len(batches) == n
    assert batches[0] == ["n0"]
    assert batches[n - 1] == [f"n{n - 1}"]
