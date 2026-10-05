"""Tests on a fixed large graph covering a deep chain and wide levels.

The graph data below is fully fixed (no random data, no timestamps):

* a chain ``c_0000 ... c_1499`` deeper than Python's default recursion
  limit of 1000, proving planning is iterative;
* 200 independent roots ``r_000 ... r_199`` sharing level 0 with the chain
  root (a wide level);
* 200 sinks ``s_000 ... s_199``, each depending on its own root and on the
  chain tip, forming a wide level 1500.

The same plan bytes must appear regardless of how tasks and dependencies are
ordered on disk and regardless of Python's hash seed (set iteration order).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from support import REPO_ROOT, cli, main_plan

CHAIN_DEPTH = 1500
WIDTH = 200
CHAIN_TIP = f"c_{CHAIN_DEPTH - 1:04d}"


def chain_tasks():
    tasks = [{"id": "c_0000"}]
    tasks += [
        {"id": f"c_{i:04d}", "depends_on": [f"c_{i - 1:04d}"]}
        for i in range(1, CHAIN_DEPTH)
    ]
    return tasks


def root_tasks():
    return [{"id": f"r_{i:03d}"} for i in range(WIDTH)]


def sink_tasks():
    return [
        {"id": f"s_{i:03d}", "depends_on": [f"r_{i:03d}", CHAIN_TIP]}
        for i in range(WIDTH)
    ]


def ordered_documents():
    """Fixed on-disk orderings of the identical graph."""
    chain, roots, sinks = chain_tasks(), root_tasks(), sink_tasks()

    forward = {"tasks": roots + chain + sinks}

    reversed_order = {"tasks": list(reversed(sinks + chain + roots))}

    # Sinks with reversed dependency lists; roots interleaved by a fixed
    # pattern; chain split into two swapped halves.
    reversed_dep_sinks = [
        {"id": task["id"], "depends_on": list(reversed(task["depends_on"]))}
        for task in sinks
    ]
    interleaved_roots = roots[::2] + roots[1::2]
    split_chain = chain[CHAIN_DEPTH // 2:] + chain[: CHAIN_DEPTH // 2]
    shuffled = {"tasks": reversed_dep_sinks[:7] + interleaved_roots
                + split_chain + reversed_dep_sinks[7:]}

    return [("roots_chain_sinks", forward),
            ("fully_reversed", reversed_order),
            ("fixed_shuffle", shuffled)]


def expected_levels():
    """Spec-derived expectation: depth of deepest dependency + 1."""
    levels = [["c_0000"] + [f"r_{i:03d}" for i in range(WIDTH)]]
    for i in range(1, CHAIN_DEPTH):
        levels.append([f"c_{i:04d}"])
    levels.append([f"s_{i:03d}" for i in range(WIDTH)])
    return levels


class LargeGraphTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, document, name="large.json"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as handle:
            handle.write(json.dumps(document, ensure_ascii=False).encode("utf-8"))
        return path

    def expected_bytes(self):
        return (json.dumps({"levels": expected_levels()},
                           ensure_ascii=False, separators=(",", ":")) + "\n")

    def test_plan_matches_spec_levels(self):
        path = self.write(ordered_documents()[0][1])
        returncode, stdout, stderr = main_plan(path)
        self.assertEqual((returncode, stderr), (0, ""))
        self.assertEqual(stdout, self.expected_bytes())

    def test_structure_spot_checks(self):
        path = self.write(ordered_documents()[0][1])
        _, stdout, _ = main_plan(path)
        levels = json.loads(stdout)["levels"]
        self.assertEqual(len(levels), CHAIN_DEPTH + 1)
        self.assertEqual(len(levels[0]), WIDTH + 1)
        self.assertEqual(levels[0][0], "c_0000")
        self.assertEqual(levels[0][1], "r_000")
        self.assertEqual(levels[0][-1], "r_199")
        self.assertEqual(levels[CHAIN_DEPTH // 2], [f"c_{CHAIN_DEPTH // 2:04d}"])
        self.assertEqual(levels[-1][0], "s_000")
        self.assertEqual(levels[-1][-1], f"s_{WIDTH - 1:03d}")
        self.assertEqual(len(levels[-1]), WIDTH)

    def test_deeper_than_recursion_limit_does_not_fail(self):
        # The default recursion limit (1000) is below the chain depth; run
        # with an even smaller limit to make the iterative contract explicit.
        path = self.write(ordered_documents()[0][1])
        code = (
            "import sys; sys.setrecursionlimit(200);\n"
            "from workflow_orchestrator.plan import run;\n"
            f"rc = run({path!r}, sys.stdout, sys.stderr);\n"
            "sys.exit(rc)\n"
        )
        env = dict(os.environ, PYTHONPATH=REPO_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT, env=env, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout.decode("utf-8"), self.expected_bytes())

    def test_orderings_produce_identical_bytes(self):
        expected = self.expected_bytes()
        for name, document in ordered_documents():
            with self.subTest(ordering=name):
                path = self.write(document)
                returncode, stdout, stderr = main_plan(path)
                self.assertEqual((returncode, stderr), (0, ""))
                self.assertEqual(stdout, expected)

    def test_hash_seed_does_not_change_bytes(self):
        path = self.write(ordered_documents()[2][1])
        outputs = set()
        for seed in ("0", "1", "12345", "999999"):
            result = cli("plan", path, env={"PYTHONHASHSEED": seed})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, b"")
            outputs.add(result.stdout)
        self.assertEqual(outputs, {self.expected_bytes().encode("utf-8")})


if __name__ == "__main__":
    unittest.main()
