from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from macqwen.tools.repo import Repo
from macqwen.tools.toolbox import Toolbox


class FakeDocs:
    def docs(self, library, topic=None):
        return {"library": library, "topic": topic, "signature": "f(x)"}


class FakeWeb:
    def search(self, query):
        return {"answer": f"about {query}", "sources": []}


class ToolboxTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        Path(self.dir.name, "a.py").write_text("x = 1\n")
        self.repo = Repo(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_filesystem_tools_reach_the_repo(self):
        box = Toolbox(self.repo)
        self.assertIn("a.py", str(box.call("find_files", {"pattern": "*.py"})))

    def test_api_docs_reaches_its_provider(self):
        box = Toolbox(self.repo, docs=FakeDocs())
        out = box.call("api_docs", {"library": "mlx", "topic": "array"})
        self.assertEqual(out["signature"], "f(x)")

    def test_web_search_reaches_its_provider(self):
        box = Toolbox(self.repo, web=FakeWeb())
        self.assertIn("kv cache", str(box.call("web_search", {"query": "kv cache"})))

    def test_missing_provider_returns_an_error_not_a_crash(self):
        box = Toolbox(self.repo)
        self.assertEqual(box.missing, ("api_docs", "web_search"))
        for name, args in (("api_docs", {"library": "mlx"}),
                           ("web_search", {"query": "anything"})):
            with self.subTest(tool=name):
                out = box.call(name, args)
                self.assertIn("error", out)
                self.assertIn("unavailable", out["error"])

    def test_the_advertised_tools_are_all_servable(self):
        # every tool in the schema must route somewhere
        from macqwen.tools import TOOLS

        box = Toolbox(self.repo, docs=FakeDocs(), web=FakeWeb())
        for entry in TOOLS:
            name = entry["function"]["name"]
            with self.subTest(tool=name):
                if name in ("api_docs", "web_search"):
                    continue
                self.assertTrue(hasattr(box.repo, name), f"{name} has no implementation")


if __name__ == "__main__":
    unittest.main()
