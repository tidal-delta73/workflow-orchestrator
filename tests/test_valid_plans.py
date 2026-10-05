"""Regression tests for valid plans.

Covers empty task sets, single nodes, multi-layer dependencies, shared
dependencies, Unicode identifiers in the same level, the rule that a node
sits one level below its deepest dependency, and byte-for-byte invariance
under task ordering, dependency ordering, and duplicated dependency entries.
"""
import json
import os
import tempfile
import unittest

from support import (
    VALID_GRAPHS,
    dependency_variants,
    main_plan,
    task_permutations,
    with_dependencies,
    write_json,
)


class ValidPlanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def plan_text(self, document):
        path = write_json(self.tmpdir, document)
        returncode, stdout, stderr = main_plan(path)
        return returncode, stdout, stderr

    def assertPlan(self, document, expected):
        returncode, stdout, stderr = self.plan_text(document)
        self.assertEqual(returncode, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, expected)
        self.assertTrue(stdout.endswith("\n"))
        self.assertEqual(stdout.count("\n"), 1)

    def test_documented_graphs(self):
        for name, (document, expected) in VALID_GRAPHS.items():
            with self.subTest(graph=name):
                self.assertPlan(document, expected)

    def test_empty_tasks_levels_is_empty_array(self):
        self.assertPlan({"tasks": []}, '{"levels":[]}\n')

    def test_output_is_compact_json_single_line(self):
        document = VALID_GRAPHS["multi_layer_diamond"][0]
        _, stdout, _ = self.plan_text(document)
        self.assertNotIn(" ", stdout)
        self.assertNotIn("\n", stdout[:-1])
        decoded = json.loads(stdout)
        self.assertEqual(list(decoded), ["levels"])

    def test_node_one_level_below_deepest_dependency(self):
        # "d" depends on "c" (level 2) and "a" (level 0): it must land at
        # level 3, not level 1 as the shallowest dependency would suggest.
        document = VALID_GRAPHS["deepest_dependency_rule"][0]
        path = write_json(self.tmpdir, document)
        _, stdout, _ = main_plan(path)
        levels = json.loads(stdout)["levels"]
        by_task = {task: depth for depth, layer in enumerate(levels)
                   for task in layer}
        self.assertEqual(by_task, {"a": 0, "b": 1, "c": 1, "d": 2, "e": 3})

    def test_shared_dependency_not_duplicated(self):
        path = write_json(self.tmpdir, VALID_GRAPHS["shared_dependency"][0])
        _, stdout, _ = main_plan(path)
        levels = json.loads(stdout)["levels"]
        flat = [task for layer in levels for task in layer]
        self.assertEqual(flat.count("r1"), 1)
        self.assertEqual(flat.count("r2"), 1)
        self.assertEqual(flat, ["r1", "r2", "x"])

    def test_level_sorted_by_unicode_code_point(self):
        path = write_json(self.tmpdir, VALID_GRAPHS["unicode_same_level"][0])
        _, stdout, _ = main_plan(path)
        (layer,) = json.loads(stdout)["levels"]
        self.assertEqual(layer, sorted(layer))
        self.assertEqual(
            [ord(ch) for ch in ("a", "é", "あ", "中")],
            [0x61, 0xE9, 0x3042, 0x4E2D],
        )
        self.assertEqual(layer, ["a", "aa", "é", "あ", "中"])

    def test_multiple_unicode_identifiers_across_levels(self):
        document = {"tasks": [
            {"id": "δ", "depends_on": ["γ"]},
            {"id": "β", "depends_on": ["α"]},
            {"id": "α"},
            {"id": "γ", "depends_on": ["α"]},
        ]}
        self.assertPlan(
            document,
            '{"levels":[["α"],["β","γ"],["δ"]]}\n',
        )

    def test_task_ordering_does_not_change_plan_bytes(self):
        for name, (document, expected) in VALID_GRAPHS.items():
            if len(document["tasks"]) < 2:
                continue
            seen = set()
            for reordered in task_permutations(document):
                returncode, stdout, stderr = self.plan_text(reordered)
                self.assertEqual((returncode, stderr), (0, ""), name)
                seen.add(stdout)
            with self.subTest(graph=name):
                self.assertEqual(seen, {expected})

    def test_dependency_ordering_does_not_change_plan_bytes(self):
        for name, (document, expected) in VALID_GRAPHS.items():
            tasks_with_deps = [
                index for index, task in enumerate(document["tasks"])
                if len(task.get("depends_on", [])) >= 2
            ]
            for index in tasks_with_deps:
                seen = set()
                for variant in dependency_variants(document["tasks"][index]["depends_on"]):
                    reordered = with_dependencies(document, index, variant)
                    returncode, stdout, stderr = self.plan_text(reordered)
                    self.assertEqual((returncode, stderr), (0, ""), (name, variant))
                    seen.add(stdout)
                with self.subTest(graph=name, task=document["tasks"][index]["id"]):
                    self.assertEqual(seen, {expected})

    def test_duplicate_dependency_entries_are_idempotent(self):
        base = VALID_GRAPHS["complex_six_nodes"][0]
        variants = [
            ["a", "a", "b"], ["a", "b", "a"], ["b", "a", "a"],
            ["a", "b", "b", "a"],
        ]
        seen = set()
        for variant in variants:
            document = with_dependencies(base, 2, variant)  # task c
            returncode, stdout, stderr = self.plan_text(document)
            self.assertEqual((returncode, stderr), (0, ""))
            seen.add(stdout)
        self.assertEqual(seen, {VALID_GRAPHS["complex_six_nodes"][1]})

    def test_json_bytes_on_disk_match_documented_output(self):
        document = VALID_GRAPHS["multi_layer_diamond"][0]
        path = write_json(self.tmpdir, document)
        with open(path, "rb") as handle:
            handle.read()
        returncode, stdout, _ = main_plan(path)
        self.assertIsInstance(stdout, str)
        self.assertEqual(
            stdout.encode("utf-8"),
            b'{"levels":[["build"],["lint","test"],["deploy"]]}\n',
        )


if __name__ == "__main__":
    unittest.main()
