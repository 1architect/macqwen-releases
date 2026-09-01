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


import json  # noqa: E402  (used in the first assertion)

if __name__ == "__main__":
    unittest.main()
