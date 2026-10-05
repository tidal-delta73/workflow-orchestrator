"""Regression tests for invalid definitions.

Every failure path documented in README is checked through the public
command entry: exit status, the unique stderr diagnostic, and an empty
stdout. Definitions containing several graph problems at once verify that
the selected diagnostic is determined by Unicode code point order of task
id and dependency id, independent of input ordering.
"""
import itertools
import json
import os
import tempfile
import unittest

from support import (
    dependency_variants,
    main_plan,
    write_json,
    write_raw,
)


class InvalidPlanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def assertFailure(self, document, message, *, raw=None, path=None):
        if raw is not None:
            path = write_raw(self.tmpdir, raw)
        elif path is None:
            path = write_json(self.tmpdir, document)
        returncode, stdout, stderr = main_plan(path)
        self.assertEqual(returncode, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, message + "\n")
        self.assertEqual(stderr.count("\n"), 1)

    def test_file_unreadable_exit_1(self):
        missing = os.path.join(self.tmpdir, "does-not-exist.json")
        returncode, stdout, stderr = main_plan(missing)
        self.assertEqual(returncode, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, f"cannot read definition: {missing}\n")

    def test_directory_is_unreadable_exit_1(self):
        directory = os.path.join(self.tmpdir, "a-directory")
        os.mkdir(directory)
        returncode, stdout, stderr = main_plan(directory)
        self.assertEqual(returncode, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, f"cannot read definition: {directory}\n")

    def test_non_json(self):
        self.assertFailure(None, "invalid json", raw=b"{not json")

    def test_empty_file_is_invalid_json(self):
        self.assertFailure(None, "invalid json", raw=b"")

    def test_invalid_utf8_is_invalid_json(self):
        self.assertFailure(None, "invalid json", raw=b"\xff\xfe{\"tasks\":[]}")

    def test_top_level_must_be_object(self):
        for document in ([], 42, "text", None, True):
            with self.subTest(document=document):
                self.assertFailure(document, "invalid definition")

    def test_top_level_tasks_key_rules(self):
        for document in (
            {},
            {"other": []},
            {"tasks": [], "other": 1},
            {"tasks": None},
            {"tasks": {}},
            {"tasks": "x"},
        ):
            with self.subTest(document=document):
                self.assertFailure(document, "invalid definition")

    def test_task_structure_rules(self):
        for document in (
            {"tasks": [[]]},
            {"tasks": ["x"]},
            {"tasks": [42]},
            {"tasks": [None]},
            {"tasks": [{}]},
            {"tasks": [{"id": ""}]},
            {"tasks": [{"id": 7}]},
            {"tasks": [{"id": None}]},
            {"tasks": [{"id": "a", "depends_on": []}, {"id": "b", "extra": 1}]},
            {"tasks": [{"id": "a", "weird": []}]},
            {"tasks": [{"id": "a", "depends_on": ["b"], "id2": "x"}]},
            {"tasks": [{"id": "a", "depends_on": "b"}]},
            {"tasks": [{"id": "a", "depends_on": [1]}]},
            {"tasks": [{"id": "a", "depends_on": [None]}]},
            {"tasks": [{"id": "a", "depends_on": [["b"]]}]},
        ):
            with self.subTest(document=document):
                self.assertFailure(document, "invalid definition")

    def test_duplicate_task_id(self):
        self.assertFailure(
            {"tasks": [{"id": "a"}, {"id": "a"}]},
            "duplicate task id: a",
        )

    def test_duplicate_unicode_task_id(self):
        self.assertFailure(
            {"tasks": [{"id": "中"}, {"id": "中"}, {"id": "a"}]},
            "duplicate task id: 中",
        )

    def test_unknown_dependency(self):
        self.assertFailure(
            {"tasks": [{"id": "a", "depends_on": ["zzz"]}]},
            "unknown dependency: a -> zzz",
        )

    def test_self_dependency(self):
        self.assertFailure(
            {"tasks": [{"id": "a", "depends_on": ["a"]}]},
            "self dependency: a",
        )

    def test_simple_cycle(self):
        self.assertFailure(
            {"tasks": [
                {"id": "a", "depends_on": ["b"]},
                {"id": "b", "depends_on": ["a"]},
            ]},
            "cycle detected",
        )

    def test_three_node_cycle(self):
        self.assertFailure(
            {"tasks": [
                {"id": "a"},
                {"id": "x", "depends_on": ["z"]},
                {"id": "z", "depends_on": ["y"]},
                {"id": "y", "depends_on": ["x"]},
            ]},
            "cycle detected",
        )


# Each entry: a document containing multiple graph problems and the single
# diagnostic that must win under the documented ordering rule
# (task id Unicode order, then dependency id Unicode order).
MULTI_PROBLEM_CASES = {
    "duplicate_beats_larger_task_problems": (
        {"tasks": [
            {"id": "a-dup"}, {"id": "a-dup"},
            {"id": "t-mid", "depends_on": ["zz-missing"]},
            {"id": "z1", "depends_on": ["z2"]},
            {"id": "z2", "depends_on": ["z1"]},
        ]},
        "duplicate task id: a-dup",
    ),
    "unknown_on_smallest_task_beats_self_and_cycle": (
        {"tasks": [
            {"id": "aaa", "depends_on": ["mmm"]},
            {"id": "self-s", "depends_on": ["self-s"]},
            {"id": "z1", "depends_on": ["z2"]},
            {"id": "z2", "depends_on": ["z1"]},
        ]},
        "unknown dependency: aaa -> mmm",
    ),
    "self_dependency_on_smallest_task": (
        {"tasks": [
            {"id": "aaa", "depends_on": ["aaa"]},
            {"id": "bbb", "depends_on": ["ccc"]},
        ]},
        "self dependency: aaa",
    ),
    "self_vs_unknown_same_pair_keyed_by_dependency": (
        # Same task lists itself and an unknown id; unknown dependency id
        # "q" sorts before "s", so unknown wins.
        {"tasks": [{"id": "s", "depends_on": ["s", "q"]}]},
        "unknown dependency: s -> q",
    ),
    "self_vs_unknown_self_id_smaller": (
        # Self id "m" sorts before the unknown id "n": self wins.
        {"tasks": [
            {"id": "m", "depends_on": ["m", "n"]}, {"id": "n"},
        ]},
        "self dependency: m",
    ),
    "two_unknown_dependencies_same_task": (
        {"tasks": [{"id": "t", "depends_on": ["zzz", "aaa"]}]},
        "unknown dependency: t -> aaa",
    ),
    "two_unknown_dependencies_different_tasks": (
        {"tasks": [
            {"id": "u2", "depends_on": ["ua"]},
            {"id": "u1", "depends_on": ["xa"]},
        ]},
        "unknown dependency: u1 -> xa",
    ),
    "two_disjoint_cycles": (
        {"tasks": [
            {"id": "m1", "depends_on": ["m2"]}, {"id": "m2", "depends_on": ["m1"]},
            {"id": "d1", "depends_on": ["d2"]}, {"id": "d2", "depends_on": ["d1"]},
        ]},
        "cycle detected",
    ),
    "cycle_and_duplicate_cycle_key_smallest": (
        {"tasks": [
            {"id": "dup-a"}, {"id": "dup-a"},
            {"id": "t-mid", "depends_on": ["zz-missing"]},
            {"id": "self-x", "depends_on": ["self-x"]},
            {"id": "c1", "depends_on": ["c2"]}, {"id": "c2", "depends_on": ["c1"]},
        ]},
        "cycle detected",
    ),
    "unknown_smaller_task_beats_cycle_larger_task": (
        {"tasks": [
            {"id": "a1", "depends_on": ["a2"]}, {"id": "a2", "depends_on": ["a1"]},
            {"id": "a0", "depends_on": ["zz"]},
        ]},
        "unknown dependency: a0 -> zz",
    ),
    "unicode_ordering_of_problem_pairs": (
        # "a" -> "missing" sorts before "中" -> "あ" by first task id.
        {"tasks": [
            {"id": "中", "depends_on": ["あ"]},
            {"id": "a", "depends_on": ["missing"]},
        ]},
        "unknown dependency: a -> missing",
    ),
}


class MultiProblemSelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def run_plan(self, document):
        path = write_json(self.tmpdir, document)
        return main_plan(path)

    def test_documented_winner_for_each_case(self):
        for name, (document, message) in MULTI_PROBLEM_CASES.items():
            with self.subTest(case=name):
                returncode, stdout, stderr = self.run_plan(document)
                self.assertEqual(returncode, 2)
                self.assertEqual(stdout, "")
                self.assertEqual(stderr, message + "\n")

    def test_selection_independent_of_task_ordering(self):
        for name, (document, message) in MULTI_PROBLEM_CASES.items():
            tasks = document["tasks"]
            results = set()
            for order in itertools.permutations(range(len(tasks))):
                reordered = {"tasks": [tasks[i] for i in order]}
                rc, stdout, stderr = self.run_plan(reordered)
                results.add((rc, stdout, stderr))
            with self.subTest(case=name):
                self.assertEqual(results, {(2, "", message + "\n")})

    def test_selection_independent_of_dependency_ordering(self):
        for name, (document, message) in MULTI_PROBLEM_CASES.items():
            results = set()
            options_per_task = []
            for task in document["tasks"]:
                deps = task.get("depends_on", [])
                options_per_task.append(
                    dependency_variants(deps) if deps else [deps]
                )
            for combo in itertools.product(*options_per_task):
                reordered = {"tasks": []}
                for task, deps in zip(document["tasks"], combo):
                    entry = {"id": task["id"]}
                    if "depends_on" in task or deps:
                        entry["depends_on"] = list(deps)
                    reordered["tasks"].append(entry)
                results.add(self.run_plan(reordered))
            with self.subTest(case=name):
                self.assertEqual(results, {(2, "", message + "\n")})

    def test_only_one_diagnostic_line_ever(self):
        for name, (document, _) in MULTI_PROBLEM_CASES.items():
            _, stdout, stderr = self.run_plan(document)
            self.assertEqual(stdout, "", name)
            self.assertEqual(len(stderr.splitlines()), 1, (name, stderr))


if __name__ == "__main__":
    unittest.main()
