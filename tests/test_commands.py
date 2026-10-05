"""Tests for command dispatch, output channels, locale, and packaging.

These go through a real interpreter via ``python -m workflow_orchestrator``
so the exit status and the stdout/stderr channel of every documented
command are observed at the process boundary.
"""
import json
import os
import sys
import tempfile
import unittest

from workflow_orchestrator import __version__

from support import USAGE_BYTES, VERSION_BYTES, cli, write_json


class CommandTests(unittest.TestCase):
    def test_version_stdout_exit_0(self):
        result = cli("version")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout, __version__.encode("ascii") + b"\n")
        self.assertEqual(result.stdout, VERSION_BYTES)

    def test_help_variants_go_to_stdout_exit_0(self):
        for argv in (["help"], ["-h"], ["--help"], []):
            with self.subTest(argv=argv):
                result = cli(*argv)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stderr, b"")
                self.assertEqual(result.stdout, USAGE_BYTES)

    def test_unknown_command_goes_to_stderr_exit_2(self):
        result = cli("bogus")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            b"unknown command: bogus\n" + USAGE_BYTES,
        )

    def test_plan_wrong_argc_goes_to_stderr_exit_2(self):
        for argv in (["plan"], ["plan", "a", "b"]):
            with self.subTest(argv=argv):
                result = cli(*argv)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, USAGE_BYTES)

    def test_help_text_does_not_change(self):
        # Pin the documented wording/bytes explicitly.
        self.assertTrue(USAGE_BYTES.startswith(
            b"usage: python3 -m workflow_orchestrator <command>\n"))
        self.assertIn(b"plan <definition.json>", USAGE_BYTES)


class ProcessBoundaryPlanTests(unittest.TestCase):
    """Exact bytes/channels for representative plan paths via real process."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def plan(self, document):
        path = write_json(self.tmpdir, document)
        return cli("plan", path)

    def test_success_bytes_and_channels(self):
        result = self.plan({"tasks": [
            {"id": "task-a"},
            {"id": "task-b", "depends_on": ["task-a"]},
        ]})
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(
            result.stdout,
            b'{"levels":[["task-a"],["task-b"]]}\n',
        )

    def test_every_documented_failure_has_unique_stderr_and_empty_stdout(self):
        cases = [
            ({"tasks": [{"id": "a"}, {"id": "a"}]},
             b"duplicate task id: a\n"),
            ({"tasks": [{"id": "a", "depends_on": ["zzz"]}]},
             b"unknown dependency: a -> zzz\n"),
            ({"tasks": [{"id": "a", "depends_on": ["a"]}]},
             b"self dependency: a\n"),
            ({"tasks": [
                {"id": "a", "depends_on": ["b"]},
                {"id": "b", "depends_on": ["a"]},
            ]}, b"cycle detected\n"),
            ([], b"invalid definition\n"),
        ]
        for document, message in cases:
            with self.subTest(message=message):
                result = self.plan(document)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, message)
                self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_invalid_json_channels(self):
        path = os.path.join(self.tmpdir, "broken.json")
        with open(path, "wb") as handle:
            handle.write(b"{nope")
        result = cli("plan", path)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"invalid json\n")

    def test_unreadable_channels_exit_1(self):
        path = os.path.join(self.tmpdir, "missing.json")
        result = cli("plan", path)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            f"cannot read definition: {path}\n".encode("utf-8"),
        )


class LocaleIndependenceTests(unittest.TestCase):
    """Python 3.10+ must plan Unicode identifiers under a plain C locale."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_unicode_plan_under_c_locale(self):
        path = write_json(self.tmpdir, {"tasks": [
            {"id": "中"}, {"id": "a"}, {"id": "あ"},
        ]})
        env = {
            "LC_ALL": "C",
            "LANG": "C",
            "LANGUAGE": "",
            "PYTHONUTF8": "",   # not explicitly forced either way
            "PYTHONCOERCECLOCALE": "",
        }
        result = cli("plan", path, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(
            result.stdout,
            '{"levels":[["a","あ","中"]]}\n'.encode("utf-8"),
        )
        decoded = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(decoded["levels"][0], ["a", "あ", "中"])


class NoRuntimeDependenciesTests(unittest.TestCase):
    """The installed library stays stdlib-only; pytest stays dev-only."""

    def test_project_declares_no_runtime_dependencies(self):
        pyproject = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pyproject.toml",
        )
        with open(pyproject, "r", encoding="utf-8") as handle:
            text = handle.read()

        if sys.version_info >= (3, 11):
            import tomllib
            data = tomllib.loads(text)
            self.assertEqual(data["project"]["dependencies"], [])
            requires_python = data["project"]["requires-python"]
        else:  # Python 3.10: inspect the fixed declaration lines
            self.assertIn("dependencies = []", text)
            requires_python = next(
                line.split("=", 1)[1].strip().strip('"')
                for line in text.splitlines()
                if line.startswith("requires-python")
            )
        self.assertEqual(requires_python, ">=3.10")

    def test_product_imports_only_stdlib_modules(self):
        package_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "workflow_orchestrator",
        )
        imported = set()
        for name in os.listdir(package_dir):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(package_dir, name), encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped.startswith("import "):
                        imported.add(stripped.split()[1].split(".")[0])
                    elif stripped.startswith("from "):
                        token = stripped.split()[1]
                        if token == "__future__" or token.startswith("."):
                            continue
                        imported.add(token.split(".")[0])
        stdlib = getattr(sys, "stdlib_module_names", set())
        non_stdlib = {module for module in imported if module not in stdlib}
        self.assertEqual(
            non_stdlib, set(),
            f"non-stdlib runtime imports: {sorted(non_stdlib)}",
        )

    def test_runs_without_third_party_packages_on_path(self):
        # -S skips the site module entirely, so neither user nor system
        # site-packages can supply workflow_orchestrator or anything else;
        # the cwd (repo root) is what makes the package importable.
        import subprocess
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys\n"
            "assert not any('site-packages' in p for p in sys.path), sys.path\n"
            "from workflow_orchestrator.__main__ import main\n"
            "raise SystemExit(main(['version']))\n"
        )
        full_env = dict(os.environ, PYTHONNOUSERSITE="1")
        proc = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=repo_root, env=full_env, capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, VERSION_BYTES)


if __name__ == "__main__":
    unittest.main()
