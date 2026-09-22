"""The repository tools, including the containment rule they exist to enforce."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from macqwen.tools.repo import Repo


class RepoTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("def main():\n    return 42\n")
        (self.root / "notes.md").write_text("alpha\nbeta\n")
        self.repo = Repo(str(self.root))

    def tearDown(self):
        self.dir.cleanup()

    def test_read_file(self):
        out = self.repo.read_file("src/main.py")
        self.assertIn("return 42", json.dumps(out) if isinstance(out, dict) else str(out))

    def test_find_files(self):
        out = str(self.repo.find_files("*.py"))
        self.assertIn("main.py", out)

    def test_search(self):
        out = str(self.repo.search("return 42"))
        self.assertIn("main.py", out)

    def test_write_refuses_to_overwrite(self):
        self.repo.write_file("new.txt", "one")
        with self.assertRaises(Exception):
            self.repo.write_file("new.txt", "two")
        self.assertEqual((self.root / "new.txt").read_text(), "one")

    def test_replace_text_is_exact(self):
        self.repo.replace_text("notes.md", "alpha", "gamma", 1)
        self.assertIn("gamma", (self.root / "notes.md").read_text())

    def test_replace_text_refuses_wrong_count(self):
        # an edit that does not match expectations must not half-apply
        with self.assertRaises(Exception):
            self.repo.replace_text("notes.md", "alpha", "x", 5)
        self.assertIn("alpha", (self.root / "notes.md").read_text())

    def test_paths_cannot_escape_the_workspace(self):
        # containment is enforced by raising, so a caller cannot ignore it
        for escape in ("../outside.txt", "/etc/passwd", "src/../../gone.txt"):
            with self.subTest(path=escape):
                with self.assertRaises(ValueError):
                    self.repo.read_file(escape)

    def test_write_cannot_escape_the_workspace(self):
        with self.assertRaises(ValueError):
            self.repo.write_file("../escaped.txt", "no")
        self.assertFalse((self.root.parent / "escaped.txt").exists())

    def test_call_dispatches_by_name(self):
        out = str(self.repo.call("find_files", {"pattern": "*.md"}))
        self.assertIn("notes.md", out)


class RunCommandTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.repo = Repo(str(self.root))

    def tearDown(self):
        self.dir.cleanup()

    def test_fast_command_reports_not_timed_out(self):
        out = self.repo.run_command("echo hi", timeout_seconds=10)
        self.assertEqual(out["exit_code"], 0)
        self.assertFalse(out["timed_out"])
        self.assertIn("hi", out["stdout"])
        self.assertIn("cwd", out)

    def test_timeout_keeps_exit_124_shape(self):
        out = self.repo.run_command("sleep 5", timeout_seconds=1)
        self.assertEqual(out["exit_code"], 124)
        self.assertTrue(out["timed_out"])
        self.assertIn("stdout", out)
        self.assertIn("stderr", out)
        self.assertIn("truncated", out)
        self.assertIn("cwd", out)
        self.assertIn("Timed out", out["stderr"])

    def test_timeout_kills_descendant_child(self):
        import os
        import time
        pidfile = self.root / "child.pid"
        if pidfile.exists():
            pidfile.unlink()
        out = self.repo.run_command(
            f'sleep 30 & echo $! > "{pidfile}"; wait', timeout_seconds=1)
        self.assertEqual(out["exit_code"], 124)
        self.assertTrue(out["timed_out"])
        self.assertTrue(pidfile.exists(), "child never started; test proves nothing")
        child = int(pidfile.read_text().strip())
        gone = False
        for _ in range(20):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                gone = True
                break
            except OSError:
                gone = True
                break
            time.sleep(0.1)
        if not gone:
            try:
                os.kill(child, 9)
            except OSError:
                pass
        self.assertTrue(gone, f"descendant {child} still alive after timeout")


class CodeCheckTriStateTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.repo = Repo(str(self.root))

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_checker_is_unavailable_not_clean_pass(self):
        import subprocess
        from unittest.mock import patch
        from macqwen.tools import code_check
        with patch.object(code_check.subprocess, "run",
                          side_effect=FileNotFoundError(2, "no such checker")):
            code, out = code_check._run(["node", "--check"], self.root)
            self.assertIsNone(code)
            self.assertIn("unavailable", out)
            errors, warnings = code_check.check("x.js", "var x = 1;\n")
            self.assertEqual(errors, [])
            self.assertTrue(warnings, "missing checker must not clean-pass")
            self.assertIn("unavailable", " ".join(warnings).lower())

    def test_checker_timeout_is_unavailable(self):
        import subprocess
        from unittest.mock import patch
        from macqwen.tools import code_check
        def _boom(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0] if args else "cmd",
                                            timeout=code_check.TIMEOUT)
        with patch.object(code_check.subprocess, "run", side_effect=_boom):
            code, out = code_check._run(["bash", "-n"], self.root)
            self.assertIsNone(code)
            self.assertIn("timed out", out)
            errors, warnings = code_check.check("x.sh", "echo hi\n")
            self.assertEqual(errors, [])
            self.assertTrue(warnings)
            self.assertIn("unavailable", " ".join(warnings).lower())

    def test_genuine_pass_and_fail(self):
        from macqwen.tools import code_check
        errors, warnings = code_check.check("ok.json", '{"a": 1}\n')
        self.assertEqual((errors, warnings), ([], []))
        errors, _ = code_check.check("bad.json", '{"a": 1,}\n')
        self.assertTrue(errors, "invalid JSON must fail")
        errors, _ = code_check.check("bad.py", "def f(:\n  pass\n")
        self.assertTrue(errors, "invalid Python must fail")

    def test_write_warns_but_writes_when_verification_unavailable(self):
        from unittest.mock import patch
        from macqwen.tools import code_check
        with patch.object(code_check.subprocess, "run",
                          side_effect=FileNotFoundError(2, "no checker")):
            out = self.repo.write_file("warn.js", "var x = 1;\n")
            self.assertTrue((self.root / "warn.js").exists())
            self.assertIn("warnings", out)
            self.assertIn("unavailable", " ".join(out["warnings"]).lower())
            (self.root / "plain.txt").write_text("hello")
            out2 = self.repo.replace_text("plain.txt", "hello", "bye", 1)
            self.assertEqual((self.root / "plain.txt").read_text(), "bye")
            # .txt has no checker, so no warning is required there
            self.assertNotIn("warnings", out2)

    def test_replace_warns_when_checker_unavailable(self):
        from unittest.mock import patch
        from macqwen.tools import code_check
        (self.root / "app.js").write_text("var x = 1;\n")
        # clear the second-attempt cache so each subtest starts fresh
        if hasattr(self.repo, "_refused"):
            self.repo._refused.clear()
        with patch.object(code_check.subprocess, "run",
                          side_effect=FileNotFoundError(2, "no checker")):
            out = self.repo.replace_text("app.js", "var x = 1;", "var x = 2;", 1)
            self.assertIn("2", (self.root / "app.js").read_text())
            self.assertIn("warnings", out)
            self.assertIn("unavailable", " ".join(out["warnings"]).lower())

    def test_genuine_failure_still_blocks_write(self):
        with self.assertRaises(ValueError):
            self.repo.write_file("bad.json", '{"a": 1,}\n')
        self.assertFalse((self.root / "bad.json").exists())


import json  # noqa: E402  (used in the first assertion)

if __name__ == "__main__":
    unittest.main()
