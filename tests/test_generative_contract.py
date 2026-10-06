"""Generative black-box regression tests for the public ``plan``/``schedule``.

These tests treat the two commands strictly through their documented entry
point (``python -m workflow_orchestrator``) and pin the current planning and
scheduling semantics as a stable byte-level baseline:

* a fixed seed reproduces the same sequence of test cases run after run,
* valid cases cover empty graphs, multiple roots, wide layers, long chains,
  diamond joins, and pairwise-independent subgraphs,
* each graph is presented through several *equivalent* JSON spellings
  (shuffled task order, shuffled ``depends_on`` order, duplicated
  dependency items); equivalent input under identical parameters must
  produce byte-identical stdout, exit code 0, and empty stderr,
* expected levels/batches come from independent reference oracles in
  ``generative`` (which never imports the package) and are additionally
  checked against the semantic rules directly,
* ``schedule`` is exercised at parallelism 1, exactly the number of tasks
  initially ready, and above the total task count,
* seeded invalid cases (duplicate ids, unknown deps, self deps, cycles,
  and mixtures) report the oracle-selected diagnostic from both commands
  with exit code 2, empty stdout, and invariant stderr across listings,
* malformed ``max-parallel`` is rejected as ``invalid parallelism`` before
  the definition file is even read,
* nothing depends on hash seed, locale, or machine iteration order (the
  subprocess environment is pinned in ``conftest``; selected cases are
  additionally run under varied seeds/locales).

Every failure message includes the integer seed, the case label, the
command/parameters, and the minimal failing document so a randomized
failure is directly reproducible.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import make_env, render_batches, render_levels
import generative as g

# Built once at collection time: the fixed, seed-derived case sequence.
VALID_CASES = g.build_valid_cases()
INVALID_CASES = g.build_invalid_cases()
PARALLELISM_SPELLINGS = g.invalid_parallelism_spellings()

# Golden fingerprints pin the exact generated sequence (documents, oracle
# outputs, and labels) across machines and Python hash seeds.
VALID_SEQUENCE_SHA256 = "415b03ac0cfbc5a6"
INVALID_SEQUENCE_SHA256 = "6afd76a7110d1f68"
PARALLELISM_SEQUENCE_SHA256 = "aa766726c3b6bb1c"

def has_non_ascii(text):
    return any(ord(ch) > 127 for ch in text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_documents(tmp_path, docs):
    paths = []
    for i, doc in enumerate(docs):
        path = tmp_path / f"doc-{i:02d}.json"
        path.write_text(g.dump(doc), encoding="utf-8")
        paths.append(path)
    return paths


def run_many(cli, jobs, *, workers=8):
    """Run many CLI invocations concurrently, preserving submission order.

    Each job is ``(argv, env)``; a bare argv list is accepted with the
    pinned default environment.
    """
    normalized = [
        job if isinstance(job, tuple) else (job, None) for job in jobs
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda job: cli.run(job[0], env=job[1]), list(normalized)
            )
        )


def _context(case, doc, command, parallelism=None):
    return (
        f"seed={case['seed']} case={case['label']} command={command} "
        f"max-parallel={parallelism} document={g.dump(doc)}"
    )


def assert_failure_context(result, expected_stderr, context):
    """Like conftest.assert_failure, with seed/document in the message."""
    assert result.returncode == 2, context
    assert result.stdout == b"", (
        "failure path must write nothing to stdout; " + context
    )
    assert result.stderr == expected_stderr, (
        f"stderr={result.stderr!r}; " + context
    )
    assert expected_stderr.endswith(b"\n") and expected_stderr.count(b"\n") == 1


def _parallelism_boundaries(case):
    """1, the initial-ready count (when positive), and n+1; de-duplicated."""
    n = case["n_tasks"]
    roots = len(case["levels"][0]) if n else 0
    values = {1}
    if roots >= 1:
        values.add(roots)
    values.add(n + 1)  # > total task count; equals 1 for the empty graph
    return sorted(values)


def _check_level_properties(entries, levels, context):
    """Structural re-check of plan semantics on the *product's* output."""
    deps = g.sorted_unique_deps(entries)
    ids = set(deps)
    flattened = [task_id for level in levels for task_id in level]
    assert sorted(flattened) == sorted(ids), context
    assert len(flattened) == len(ids), context  # every task exactly once
    position = {
        task_id: depth
        for depth, level in enumerate(levels)
        for task_id in level
    }
    for level in levels:
        assert level == sorted(level), context  # code point order in level
    for task_id, need in deps.items():
        if need:
            assert position[task_id] == 1 + max(position[d] for d in need), (
                task_id, context
            )
        else:
            assert position[task_id] == 0, (task_id, context)


def _check_batch_properties(entries, max_parallel, batches, context):
    """Structural + greedy re-check of schedule semantics on product output."""
    deps = g.sorted_unique_deps(entries)
    ids = sorted(deps)
    dependents = {task_id: [] for task_id in ids}
    for task_id in ids:
        for dep in deps[task_id]:
            dependents[dep].append(task_id)

    flattened = [task_id for batch in batches for task_id in batch]
    assert sorted(flattened) == ids, context
    assert len(flattened) == len(ids), context  # every task exactly once

    batch_of = {}
    for index, batch in enumerate(batches):
        assert len(batch) <= max_parallel, context
        assert batch == sorted(batch), context
        for task_id in batch:
            batch_of[task_id] = index
    for task_id, need in deps.items():
        for dep in need:
            assert batch_of[dep] < batch_of[task_id], context

    # Greedy readiness: each batch must be exactly the first max_parallel
    # tasks that were ready when the batch started.
    remaining = {task_id: len(deps[task_id]) for task_id in ids}
    ready = sorted(task_id for task_id in ids if remaining[task_id] == 0)
    for batch in batches:
        assert batch == ready[:max_parallel], context
        ready = ready[max_parallel:]
        for task_id in batch:
            for nxt in dependents[task_id]:
                remaining[nxt] -= 1
                if remaining[nxt] == 0:
                    ready.append(nxt)
        ready.sort()
    assert ready == [], context


# ---------------------------------------------------------------------------
# The fixed seed must reproduce the same case sequence, run after run.
# ---------------------------------------------------------------------------

def _fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_valid_case_sequence_is_fixed():
    first = g.canonical_case_text(g.build_valid_cases())
    second = g.canonical_case_text(g.build_valid_cases())
    assert first == second
    assert _fingerprint(first) == VALID_SEQUENCE_SHA256


def test_invalid_case_sequence_is_fixed():
    first = g.canonical_case_text(g.build_invalid_cases())
    second = g.canonical_case_text(g.build_invalid_cases())
    assert first == second
    assert _fingerprint(first) == INVALID_SEQUENCE_SHA256


def test_parallelism_spelling_sequence_is_fixed():
    assert (
        _fingerprint(json.dumps(PARALLELISM_SPELLINGS, ensure_ascii=False))
        == PARALLELISM_SEQUENCE_SHA256
    )
    # Guard the seed-derived labels: collection order is archetype sorted.
    assert [c["label"] for c in VALID_CASES] == sorted(
        c["label"] for c in VALID_CASES
    )
    assert [c["label"] for c in INVALID_CASES] == sorted(
        c["label"] for c in INVALID_CASES
    )


def test_archetype_coverage_guards():
    by_name = {}
    for case in VALID_CASES:
        by_name.setdefault(case["archetype"], []).append(case)

    assert {c["n_tasks"] for c in by_name["empty"]} == {0}
    assert all(len(c["levels"][0]) >= 2 for c in by_name["multi-root"])
    assert all(len(c["levels"]) >= 4 for c in by_name["chain"])
    # A wide layer somewhere carries several code-point-sorted peers.
    assert max(
        len(level) for c in by_name["wide"] for level in c["levels"]
    ) >= 4
    # Every generated diamond has a single join over >=2 branches.
    for case in by_name["diamond"]:
        deps = g.sorted_unique_deps(case["entries"])
        assert any(len(need) >= 2 for need in deps.values())
        assert len(case["levels"][0]) == 1
    # Independent subgraphs: >=2 roots with no edges joining the components.
    for case in by_name["subgraphs"]:
        assert len(case["levels"][0]) >= 2
        deps = g.sorted_unique_deps(case["entries"])
        is_leaf = lambda t: not any(t in need for need in deps.values())
        leaves = [t for t, need in deps.items() if is_leaf(t)]
        assert len(leaves) >= 2


# ---------------------------------------------------------------------------
# Valid graphs: plan semantics and equivalence across spellings.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", VALID_CASES, ids=lambda c: c["label"]
)
def test_plan_matches_oracle_and_is_spelling_invariant(cli, tmp_path, case):
    docs = case["documents"]
    paths = _write_documents(tmp_path, docs)
    results = cli.plan_many(paths)
    expected = render_levels(case["levels"])
    for doc, result in zip(docs, results):
        assert result.returncode == 0, _context(case, doc, "plan")
        assert result.stderr == b"", _context(case, doc, "plan")
        assert result.stdout == expected, _context(case, doc, "plan")
    _check_level_properties(
        case["entries"], case["levels"], _context(case, docs[0], "plan")
    )


@pytest.mark.parametrize(
    "case", VALID_CASES, ids=lambda c: c["label"]
)
def test_schedule_matches_oracle_across_equivalent_inputs(cli, tmp_path, case):
    # All equivalent spellings at a fixed parallelism must agree byte-for-byte.
    docs = case["documents"]
    paths = _write_documents(tmp_path, docs)
    results = run_many(cli, [["schedule", str(path), "2"] for path in paths])
    expected = render_batches(g.reference_batches(case["entries"], 2))
    for doc, result in zip(docs, results):
        assert result.returncode == 0, _context(case, doc, "schedule", 2)
        assert result.stderr == b"", _context(case, doc, "schedule", 2)
        assert result.stdout == expected, _context(case, doc, "schedule", 2)


@pytest.mark.parametrize(
    "case", VALID_CASES, ids=lambda c: c["label"]
)
def test_schedule_parallelism_boundaries(cli, tmp_path, case):
    # 1, exactly the number initially ready, and greater than task count.
    doc = case["documents"][0]
    path = _write_documents(tmp_path, [doc])[0]
    parallelisms = _parallelism_boundaries(case)
    results = run_many(
        cli, [["schedule", str(path), str(p)] for p in parallelisms]
    )
    for parallelism, result in zip(parallelisms, results):
        expected_batches = g.reference_batches(case["entries"], parallelism)
        expected = render_batches(expected_batches)
        assert result.returncode == 0, _context(
            case, doc, "schedule", parallelism
        )
        assert result.stderr == b"", _context(case, doc, "schedule", parallelism)
        assert result.stdout == expected, _context(
            case, doc, "schedule", parallelism
        )
        _check_batch_properties(
            case["entries"],
            parallelism,
            expected_batches,
            _context(case, doc, "schedule", parallelism),
        )


def test_plan_and_schedule_share_output_across_commands(cli, tmp_path):
    # For a single-root chain the level layers and the serial batches
    # coincide; both commands must emit the same ids in the same grouping.
    case = next(c for c in VALID_CASES if c["archetype"] == "chain")
    doc = case["documents"][0]
    path = _write_documents(tmp_path, [doc])[0]
    plan_result = cli.plan(path)
    schedule_result = cli.schedule(path, 1)
    plan_payload = json.loads(plan_result.stdout)
    schedule_payload = json.loads(schedule_result.stdout)
    assert schedule_payload["batches"] == [
        [task_id] for level in plan_payload["levels"] for task_id in level
    ]


# ---------------------------------------------------------------------------
# Invalid graphs: deterministic diagnostic, both commands, all listings.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", INVALID_CASES, ids=lambda c: c["label"]
)
def test_graph_errors_are_deterministic_in_both_commands(cli, tmp_path, case):
    expected_stderr = (case["diagnostic"] + "\n").encode("utf-8")
    docs = case["documents"]
    paths = _write_documents(tmp_path, docs)

    argv_list = []
    for path in paths:
        argv_list.append(["plan", str(path)])
        argv_list.append(["schedule", str(path), "4"])
    results = run_many(cli, argv_list)

    for index, doc in enumerate(docs):
        plan_result = results[2 * index]
        schedule_result = results[2 * index + 1]
        assert_failure_context(
            plan_result, expected_stderr, _context(case, doc, "plan")
        )
        assert_failure_context(
            schedule_result,
            expected_stderr,
            _context(case, doc, "schedule", 4),
        )
        # The shared core must make the two commands agree byte-for-byte.
        assert plan_result.stderr == schedule_result.stderr, _context(
            case, doc, "both"
        )


def test_invalid_cases_actually_exercise_each_defect_kind():
    # Each generator's own cases must be won by its defect kind, proving the
    # generators really inject the advertised fault (mixed cases aside:
    # there the oracle is free to pick any winner).
    by_kind = {}
    for case in INVALID_CASES:
        by_kind.setdefault(case["kind"], set()).add(case["diagnostic"])
    assert all(d.startswith("duplicate task id:") for d in by_kind["duplicate"])
    assert all(d.startswith("unknown dependency:") for d in by_kind["unknown"])
    assert all(d.startswith("self dependency:") for d in by_kind["self"])
    assert by_kind["cycle"] == {"cycle detected"}
    assert len(by_kind["mixed"]) >= 2  # seed-driven winners, not one constant


def test_invalid_graph_diagnostics_stable_under_hash_seeds(cli, tmp_path):
    # Mixed cases are re-run under a fixed and an interpreter-chosen seed;
    # diagnostic selection and empty stdout must not move with hash order.
    cases = [c for c in INVALID_CASES if c["kind"] == "mixed"][:2]
    argv_list = []
    targets = []
    for i, case in enumerate(cases):
        path = tmp_path / f"mixed-{i}.json"
        path.write_text(g.dump(case["documents"][0]), encoding="utf-8")
        targets.append((case, path))
    for case, path in targets:
        argv_list.append(["plan", str(path)])
        argv_list.append(["schedule", str(path), "3"])
    for seed_env in (make_env(PYTHONHASHSEED="31"),
                     make_env(PYTHONHASHSEED=None)):
        results = run_many(cli, [(argv, seed_env) for argv in argv_list])
        for index, (case, path) in enumerate(targets):
            expected = (case["diagnostic"] + "\n").encode("utf-8")
            for offset in (0, 1):
                result = results[2 * index + offset]
                assert result.returncode == 2
                assert result.stdout == b""
                assert result.stderr == expected, _context(
                    case, case["documents"][0], "seeded"
                )


# ---------------------------------------------------------------------------
# max-parallel validation precedes the definition read.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", PARALLELISM_SPELLINGS)
def test_invalid_parallelism_before_reading_definition(cli, tmp_path, text):
    # The path deliberately does not exist: invalid parallelism must win.
    missing = tmp_path / "never-read.json"
    result = cli.run(["schedule", str(missing), text])
    assert result.returncode == 2, f"spelling={text!r}"
    assert result.stdout == b"", f"spelling={text!r}"
    assert result.stderr == b"invalid parallelism\n", f"spelling={text!r}"


def test_invalid_parallelism_beats_an_invalid_graph(cli, tmp_path):
    path = tmp_path / "cyclic.json"
    path.write_text(
        g.dump(
            {
                "tasks": [
                    {"id": "a", "depends_on": ["b"]},
                    {"id": "b", "depends_on": ["a"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = cli.run(["schedule", str(path), "0x1"])
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"invalid parallelism\n"


# ---------------------------------------------------------------------------
# Locale / hash-seed independence for seed-generated Unicode content.
# ---------------------------------------------------------------------------

def _unicode_richest_case():
    def non_ascii_count(case):
        return sum(has_non_ascii(task["id"]) for task in case["entries"])

    return max(
        (c for c in VALID_CASES if c["n_tasks"]),
        key=lambda c: (non_ascii_count(c), -len(c["label"]), c["label"]),
    )


@pytest.mark.parametrize(
    "env",
    [
        make_env(),
        make_env(PYTHONHASHSEED="77"),
        make_env(PYTHONHASHSEED=None),
        make_env(LC_ALL="C.UTF-8", LANG="C.UTF-8"),
        make_env(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8"),
    ],
)
def test_output_independent_of_seed_and_locale(cli, tmp_path, env):
    case = _unicode_richest_case()
    doc = case["documents"][0]
    path = tmp_path / "unicode-case.json"
    path.write_text(g.dump(doc), encoding="utf-8")
    parallelism = len(case["levels"][0])

    plan_result = cli.run(["plan", str(path)], env=env)
    schedule_result = cli.run(
        ["schedule", str(path), str(parallelism)], env=env
    )
    for result, expected in (
        (plan_result, render_levels(case["levels"])),
        (
            schedule_result,
            render_batches(g.reference_batches(case["entries"], parallelism)),
        ),
    ):
        assert result.returncode == 0, _context(case, doc, "env")
        assert result.stderr == b"", _context(case, doc, "env")
        assert result.stdout == expected, _context(case, doc, "env")
