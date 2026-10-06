"""Generative black-box regression tests for ``plan`` and ``schedule``.

A fixed seed deterministically produces a batch of valid DAGs (empty
graph, single task, multiple roots, wide graph, long chain, diamond join,
independent subgraphs, seeded random DAGs) and invalid definitions
(duplicate task ids, unknown dependencies, self dependencies, cycles,
mixed problems). Every case is exercised end-to-end through
``python -m workflow_orchestrator`` and checked against the independent
reference oracle in ``generative.py`` -- nothing here imports the product
code, so the current planning semantics become a stable baseline for
future executor work.

For every valid graph, several equivalent JSON documents (permuted task
order, permuted ``depends_on`` order, duplicated dependency entries) must
produce byte-identical stdout, with empty stderr and exit code 0. Every
invalid document must keep the documented exit code and diagnostic text
chosen by Unicode code point order, with stdout left empty. Invalid
``max-parallel`` values are rejected before the definition file is read.

Every assertion message carries the seed and the minimal input document
so a random failure can be reproduced directly. Re-running the suite with
the same seed yields the same case sequence (pinned by the determinism
tests below); nothing depends on task enumeration order, hash seed, or
locale of the machine.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import generative
from conftest import render_batches, render_levels

SEED = 20261007

VALID_CASES = generative.valid_cases(SEED)
INVALID_CASES = generative.invalid_cases(SEED)

INVALID_PARALLELISM = b"invalid parallelism\n"


def _run_all(cli, argvs):
    """Run the CLI for each argv concurrently, preserving argv order."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(cli.run, argvs))


def _context(case_name, doc, extra=""):
    """Seed + minimal input document, for reproducible failure messages."""
    minimal = json.dumps(doc, ensure_ascii=False, sort_keys=True)
    head = "case=%r seed=%s" % (case_name, SEED)
    if extra:
        head += " " + extra
    return "%s\nminimal document: %s" % (head, minimal)


# ---------------------------------------------------------------------------
# Generator self-checks: the case sequence is a pure function of the seed.
# ---------------------------------------------------------------------------

def test_case_sequence_is_reproducible():
    assert generative.valid_cases(SEED) == VALID_CASES
    assert generative.invalid_cases(SEED) == INVALID_CASES


def test_case_sequence_depends_on_seed():
    other = SEED + 1
    assert generative.valid_cases(other) != VALID_CASES
    assert generative.invalid_cases(other) != INVALID_CASES


def test_valid_cases_are_valid_dags():
    # Guards against generator bugs: every valid case must pass the
    # reference semantics check and cover the documented shape templates.
    for case in VALID_CASES:
        entries = [(task, list(own)) for task, own in case.deps.items()]
        assert generative.reference_error(entries) is None, case.name
    names = {case.name for case in VALID_CASES}
    assert {
        "empty",
        "multi-root",
        "wide",
        "long-chain",
        "diamond",
        "independent-subgraphs",
    } <= names


# ---------------------------------------------------------------------------
# Valid DAGs: plan output matches the reference, byte for byte.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", VALID_CASES, ids=[c.name for c in VALID_CASES])
def test_plan_matches_reference(cli, write_def, case):
    expected = render_levels(generative.reference_levels(case.deps))
    paths = [write_def(doc) for doc in case.documents]
    results = _run_all(cli, [["plan", str(path)] for path in paths])

    stdouts = set()
    for doc, result in zip(case.documents, results):
        ctx = _context(case.name, doc)
        assert result.returncode == 0, (
            "exit %d, stderr=%r\n%s" % (result.returncode, result.stderr, ctx)
        )
        assert result.stderr == b"", "stderr=%r\n%s" % (result.stderr, ctx)
        assert result.stdout == expected, (
            "got %r, want %r\n%s" % (result.stdout, expected, ctx)
        )
        stdouts.add(result.stdout)
        levels = json.loads(result.stdout.decode("utf-8"))["levels"]
        problems = generative.check_levels(case.deps, levels)
        assert not problems, "\n".join(problems) + "\n" + ctx

    assert len(stdouts) == 1, (
        "equivalent documents produced different plan stdout: %r\ncase=%r seed=%s"
        % (sorted(stdouts), case.name, SEED)
    )


# ---------------------------------------------------------------------------
# Valid DAGs: schedule output matches the reference at boundary parallelisms.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", VALID_CASES, ids=[c.name for c in VALID_CASES])
def test_schedule_matches_reference(cli, write_def, case):
    paths = [write_def(doc) for doc in case.documents]

    # Every boundary parallelism on the first document, plus the remaining
    # documents at the first parallelism value to prove equivalence.
    jobs = []
    for k in case.parallelisms:
        jobs.append((["schedule", str(paths[0]), str(k)], k, case.documents[0]))
    for doc, path in zip(case.documents[1:], paths[1:]):
        k = case.parallelisms[0]
        jobs.append((["schedule", str(path), str(k)], k, doc))
    results = _run_all(cli, [argv for argv, _, _ in jobs])

    stdouts_per_k = {}
    for (_, k, doc), result in zip(jobs, results):
        ctx = _context(case.name, doc, extra="max-parallel=%d" % k)
        expected = render_batches(generative.reference_batches(case.deps, k))
        assert result.returncode == 0, (
            "exit %d, stderr=%r\n%s" % (result.returncode, result.stderr, ctx)
        )
        assert result.stderr == b"", "stderr=%r\n%s" % (result.stderr, ctx)
        assert result.stdout == expected, (
            "got %r, want %r\n%s" % (result.stdout, expected, ctx)
        )
        stdouts_per_k.setdefault(k, set()).add(result.stdout)
        batches = json.loads(result.stdout.decode("utf-8"))["batches"]
        problems = generative.check_batches(case.deps, batches, k)
        assert not problems, "\n".join(problems) + "\n" + ctx

    for k, stdouts in stdouts_per_k.items():
        assert len(stdouts) == 1, (
            "equivalent documents produced different schedule stdout at "
            "max-parallel=%d: %r\ncase=%r seed=%s"
            % (k, sorted(stdouts), case.name, SEED)
        )


def test_schedule_parallelism_boundaries_are_covered():
    # The suite as a whole must exercise all three documented boundaries:
    # 1, exactly the ready-task count, and more than the task total.
    for case in VALID_CASES:
        ks = case.parallelisms
        assert 1 in ks, case.name
        assert max(ks) > len(case.deps), case.name
    wide = next(c for c in VALID_CASES if c.name == "wide")
    assert generative._frontier_width(wide.deps) in wide.parallelisms


# ---------------------------------------------------------------------------
# Invalid definitions: deterministic diagnostic, exit code, empty stdout.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", INVALID_CASES, ids=[c.name for c in INVALID_CASES])
def test_invalid_definitions_match_reference(cli, write_def, case):
    expected = (case.message + "\n").encode("utf-8")
    paths = [write_def(doc) for doc in case.documents]

    argvs = []
    for path in paths:
        argvs.append(["plan", str(path)])
        argvs.append(["schedule", str(path), "3"])
    results = _run_all(cli, argvs)

    for doc, pair in zip(case.documents, zip(results[::2], results[1::2])):
        ctx = _context(case.name, doc)
        for command, result in zip(("plan", "schedule"), pair):
            assert result.returncode == 2, (
                "%s: exit %d\n%s" % (command, result.returncode, ctx)
            )
            assert result.stdout == b"", (
                "%s: stdout=%r\n%s" % (command, result.stdout, ctx)
            )
            assert result.stderr == expected, (
                "%s: stderr=%r, want %r\n%s"
                % (command, result.stderr, expected, ctx)
            )


# ---------------------------------------------------------------------------
# max-parallel validation happens before the definition file is read.
# ---------------------------------------------------------------------------

BAD_PARALLELISMS = [
    "",
    "0",
    "00",
    "01",
    "-1",
    "+1",
    "1.0",
    "1e3",
    "abc",
    " 1",
    "1 ",
    "2147483648",
    "99999999999999999999",
    "１２",  # full-width digits are not ASCII decimal
]


@pytest.mark.parametrize("value", BAD_PARALLELISMS)
def test_invalid_parallelism_rejected_before_reading_file(cli, tmp_path, value):
    # The path does not exist: any attempt to read it first would report
    # "cannot read definition" with exit code 1 instead.
    missing = tmp_path / "no-such-definition.json"
    result = cli.run(["schedule", str(missing), value])
    assert result.returncode == 2, "max-parallel=%r" % value
    assert result.stdout == b"", "max-parallel=%r" % value
    assert result.stderr == INVALID_PARALLELISM, "max-parallel=%r" % value


@pytest.mark.parametrize("value", BAD_PARALLELISMS)
def test_invalid_parallelism_rejected_with_valid_definition(cli, write_def, value):
    path = write_def({"tasks": [{"id": "a"}]})
    result = cli.run(["schedule", str(path), value])
    assert result.returncode == 2, "max-parallel=%r" % value
    assert result.stdout == b"", "max-parallel=%r" % value
    assert result.stderr == INVALID_PARALLELISM, "max-parallel=%r" % value


def test_largest_valid_parallelism_accepted(cli, write_def):
    path = write_def({"tasks": [{"id": "a"}, {"id": "b"}]})
    result = cli.run(["schedule", str(path), "2147483647"])
    assert result.returncode == 0
    assert result.stdout == b'{"batches":[["a","b"]]}\n'
    assert result.stderr == b""
