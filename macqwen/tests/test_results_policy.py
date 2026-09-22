"""Enforce the test layout and the single results folder.

Layout: every model keeps checkpoint-free tests in ``tests/unit``, harness
scripts in ``tests/bench`` and terminal cards in ``tests/cases``. Results:
every run writes under ``results/<model>/`` through ``macqwen.results``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from macqwen import results

ROOT = Path(__file__).resolve().parents[2]
MODELS = sorted(
    path for path in (ROOT / "models").iterdir()
    if path.is_dir() and (path / "__init__.py").is_file()
)
OUTPUT_OPTIONS = {"--json", "--jsonl", "--out", "--output", "--log", "--record"}
# Machine state that later runs read back. These are caches, not results.
STATE_FILES = (
    "pins.json", "physical-misses.json", "capacity-sweep-manifest.json",
    "capacity-sweep-", "slab-pack", "sessions", "native",
)
FORBIDDEN = (
    (re.compile(r"""["']/tmp\b"""), "writes to /tmp"),
    (re.compile(r"tempfile\.gettempdir"), "writes to the temporary directory"),
    (re.compile(r"tests/results"), "uses the retired tests/results folder"),
    (re.compile(r"docs/\w+/measurements"), "uses the retired measurements folder"),
    (re.compile(r"\.results_dir\b"), "uses results_dir; use context.output() or context.latest()"),
)


def _test_files(model: Path, sub: str) -> list[Path]:
    folder = model / "tests" / sub
    return sorted(folder.glob("*.py")) if folder.is_dir() else []


class LayoutTests(unittest.TestCase):
    def test_model_roots_hold_no_tests_or_benchmarks(self):
        for model in MODELS:
            stray = [
                path.name for path in model.glob("*.py")
                if path.name.startswith(("test_", "bench", "diagnose_", "case_"))
            ]
            self.assertEqual(stray, [], f"{model.name}: move these into tests/")

    def test_tests_folders_use_the_three_subfolders(self):
        for model in MODELS:
            tests = model / "tests"
            if not tests.is_dir():
                continue
            stray = [
                path.name for path in tests.glob("*.py") if path.name != "__init__.py"
            ]
            self.assertEqual(stray, [], f"{model.name}/tests: use unit/, bench/ or cases/")
            for path in _test_files(model, "unit"):
                if path.name != "__init__.py":
                    self.assertTrue(path.name.startswith("test_"), path)
            for path in _test_files(model, "cases"):
                if path.name != "__init__.py":
                    self.assertTrue(path.name.startswith("case_"), path)

    def test_no_per_model_catalog_or_terminal(self):
        for model in MODELS:
            for name in ("catalog.py", "runner.py", "terminal.py", "api.py"):
                self.assertFalse(
                    (model / "tests" / name).exists(),
                    f"{model.name}/tests/{name}: the shared terminal owns this",
                )

    def test_shared_unit_tests_live_in_macqwen_tests(self):
        stray = [path.name for path in (ROOT / "macqwen").glob("test_*.py")]
        self.assertEqual(stray, [])


class ResultsPolicyTests(unittest.TestCase):
    def test_benchmarks_and_cases_write_only_to_results(self):
        problems = []
        for model in MODELS:
            for sub in ("bench", "cases"):
                for path in _test_files(model, sub):
                    text = path.read_text()
                    for pattern, reason in FORBIDDEN:
                        if pattern.search(text):
                            problems.append(f"{path.relative_to(ROOT)}: {reason}")
                    for match in re.finditer(r"\.cache/flashnext/([\w.-]+)", text):
                        if not match.group(1).startswith(STATE_FILES):
                            problems.append(
                                f"{path.relative_to(ROOT)}: writes {match.group(0)}; "
                                "results belong in results/"
                            )
        self.assertEqual(problems, [])

    def test_output_options_have_no_fixed_destination(self):
        problems = []
        for model in MODELS:
            for path in _test_files(model, "bench"):
                text = path.read_text()
                tree = ast.parse(text)
                declares_output = False
                for node in ast.walk(tree):
                    if not (isinstance(node, ast.Call)
                            and getattr(node.func, "attr", "") == "add_argument"):
                        continue
                    flags = {
                        arg.value for arg in node.args
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    }
                    if not flags & OUTPUT_OPTIONS:
                        continue
                    declares_output = True
                    for keyword in node.keywords:
                        if (
                            keyword.arg == "default"
                            and isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, str)
                            and keyword.value.value
                        ):
                            problems.append(
                                f"{path.relative_to(ROOT)}: {sorted(flags)} defaults "
                                f"to {keyword.value.value!r}"
                            )
                if declares_output and not re.search(
                    r"\boutput_path\(|\bvalidate_path\(", text
                ):
                    problems.append(
                        f"{path.relative_to(ROOT)}: output option not routed "
                        "through macqwen.results.output_path"
                    )
        self.assertEqual(problems, [])


class ResultsModuleTests(unittest.TestCase):
    def test_run_directory_and_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            with mock.patch.object(results, "RESULTS_ROOT", root), \
                    mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(results.ENVIRONMENT_KEY, None)
                path = results.output_path("flashnext", "bench_x", "summary.json")
                self.assertEqual(path.name, "summary.json")
                self.assertEqual(path.parent.parent, (root / "flashnext").resolve())
                self.assertTrue(path.parent.name.endswith("-bench-x-manual"))
                # The same process keeps writing into the same run folder.
                again = results.output_path("flashnext", "bench_x", "other.json")
                self.assertEqual(again.parent, path.parent)

    def test_terminal_folder_is_used_when_handed_over(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            run = root / "flashnext" / "20260101-000000-case"
            with mock.patch.object(results, "RESULTS_ROOT", root), \
                    mock.patch.dict(os.environ, {results.ENVIRONMENT_KEY: str(run)}):
                self.assertEqual(
                    results.output_path("flashnext", "bench_x", "a.json"),
                    (run / "a.json").resolve(),
                )

    def test_destinations_outside_results_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            with mock.patch.object(results, "RESULTS_ROOT", root):
                with self.assertRaises(ValueError):
                    results.output_path("flashnext", "x", "a.json", "/tmp/a.json")
                with self.assertRaises(ValueError):
                    results.ensure_inside(Path(directory) / "elsewhere.json")

    def test_latest_finds_the_newest_earlier_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "flashnext" / "20260101-000000-a" / "sweep.json"
            new = root / "flashnext" / "20260102-000000-b" / "sweep.json"
            for path, stamp in ((old, 1_000), (new, 2_000)):
                path.parent.mkdir(parents=True)
                path.write_text("{}")
                os.utime(path, (stamp, stamp))
            self.assertEqual(results.latest("flashnext", "sweep.json", root), new)
            self.assertIsNone(results.latest("flashnext", "missing.json", root))


if __name__ == "__main__":
    unittest.main()
