from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from macqwen.tools.context7 import Context7, param_types, prefetch
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


class Context7Tests(unittest.TestCase):
    def test_agent_docs_use_compact_default_and_fields(self):
        client = Context7.__new__(Context7)
        seen = []
        client.resolve = lambda _library: "/websites/test"
        client._cached = lambda url: (seen.append(url) or ("x" * 7000, True))

        result = client.docs("sketchup", "pushpull")

        self.assertIn("tokens=1000", seen[0])
        self.assertEqual(len(result["documentation"]), 6000)
        self.assertEqual(result["library"], "/websites/test")
        self.assertEqual(result["topic"], "pushpull")
        self.assertNotIn("cached", result)
        self.assertNotIn("instruction", result)

    def test_internal_signature_checks_keep_their_budgets(self):
        client = Context7.__new__(Context7)
        seen = []
        client.docs = lambda _library, _topic, tokens=0: (
            seen.append(tokens), {"documentation": ""}
        )[1]

        client.signatures("sketchup", "pushpull")
        param_types(client, "sketchup", "pushpull")

        self.assertEqual(seen, [1500, 2000])

    def test_prefetch_keeps_its_explicit_budget(self):
        class FakeClient:
            def __init__(self):
                self.seen = []

            def docs(self, _library, _topic, tokens=0):
                self.seen.append(tokens)
                return {"library": "/websites/ruby_sketchup",
                        "documentation": "pushpull(distance)"}

        client = FakeClient()
        self.assertIn("pushpull(distance)", prefetch(
            "Use SketchUp pushpull", client=client
        ))
        self.assertEqual(client.seen, [1200])

    def test_representative_signatures_and_examples_survive(self):
        fixtures = {
            "pushpull": (
                "## pushpull(distance, copy = false)\n"
                "* distance (Length)\n"
                "* copy (Boolean, optional) - false\n"
                "### Returns\n* nil\n"
                "face.pushpull(100, true)"
            ),
            "UI.inputbox": (
                "## UI.inputbox\n"
                "`inputbox(prompts, defaults, title)`\n"
                "`inputbox(prompts, defaults, list, title)`\n"
                "### Returns\n* Array<String>\n* false\n"
                "input = UI.inputbox(prompts, defaults, list, title)"
            ),
        }
        for topic, documentation in fixtures.items():
            with self.subTest(topic=topic):
                client = Context7.__new__(Context7)
                client.resolve = lambda _library: "/websites/test"
                client._cached = lambda _url, text=documentation: (text, False)

                result = client.docs("sketchup", topic)

                for fragment in documentation.splitlines():
                    if fragment:
                        self.assertIn(fragment, result["documentation"])


if __name__ == "__main__":
    unittest.main()
