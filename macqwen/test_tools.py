"""Tool-call parsing, including a differential test against the original.

macqwen.tools was extracted from frankenstein_engine.py. Extraction is only
safe if the behaviour is identical, so where the original can still be
imported these tests compare the two directly on the awkward inputs that
motivated the parser's leniency.
"""
from __future__ import annotations

import unittest

from macqwen import tools

WELL_FORMED = """<tool_call>
<function=read_file>
<parameter=path>
src/main.py
</parameter>
<parameter=start_line>
1
</parameter>
</function>
</tool_call>"""

UNTERMINATED = """<tool_call>
<function=list_dir>
<parameter=path>
.
</parameter>
</function>"""

TAGS_OUT_OF_ORDER = """<tool_call><function=list_dir></tool_call></function></tool_call>"""

TWO_CALLS = WELL_FORMED + "\n" + """<tool_call>
<function=search>
<parameter=query>
def main
</parameter>
</function>
</tool_call>"""

MULTILINE_VALUE = """<tool_call>
<function=write_file>
<parameter=path>
notes.txt
</parameter>
<parameter=content>
first line
second line
</parameter>
</function>
</tool_call>"""

CASES = [
    "", "just a normal answer", WELL_FORMED, UNTERMINATED,
    TAGS_OUT_OF_ORDER, TWO_CALLS, MULTILINE_VALUE,
    "<think>reasoning</think>\n" + WELL_FORMED,
]


class ParseTests(unittest.TestCase):
    def test_well_formed(self):
        self.assertEqual(
            tools.parse_tool_calls(WELL_FORMED),
            [("read_file", {"path": "src/main.py", "start_line": 1})],
        )

    def test_unterminated_call_is_still_read(self):
        # Qwen often closes </function> and stops before </tool_call>
        self.assertEqual(
            tools.parse_tool_calls(UNTERMINATED), [("list_dir", {"path": "."})]
        )

    def test_plain_text_yields_nothing(self):
        self.assertEqual(tools.parse_tool_calls("just a normal answer"), [])

    def test_two_calls_in_order(self):
        names = [name for name, _ in tools.parse_tool_calls(TWO_CALLS)]
        self.assertEqual(names, ["read_file", "search"])

    def test_multiline_parameter_survives(self):
        calls = tools.parse_tool_calls(MULTILINE_VALUE)
        self.assertEqual(calls[0][1]["content"], "first line\nsecond line")

    def test_mutating_tools_are_named(self):
        self.assertEqual(
            tools.MUTATING_TOOLS, {"write_file", "replace_text", "run_command"}
        )
        for name in tools.MUTATING_TOOLS:
            self.assertIn(name, tools.PARAM_TYPES)

    def test_schema_is_self_consistent(self):
        for entry in tools.TOOLS:
            name = entry["function"]["name"]
            self.assertIn(name, tools.PARAM_TYPES)
            for required in tools.REQUIRED_PARAMS[name]:
                self.assertIn(required, tools.PARAM_TYPES[name])


class DifferentialTests(unittest.TestCase):
    """The extracted parser must agree with the one it came from."""

    @classmethod
    def setUpClass(cls):
        try:
            from models.qwen27b import frankenstein_engine
        except Exception as exc:  # the 27B venv may not be present
            raise unittest.SkipTest(f"original engine unavailable: {exc}")
        cls.original = frankenstein_engine

    def test_parse_matches_the_original(self):
        for case in CASES:
            with self.subTest(case=case[:40]):
                self.assertEqual(
                    tools.parse_tool_calls(case),
                    self.original.parse_tool_calls(case),
                )

    def test_split_think_matches(self):
        for case in ("<think>a</think>b", "no tags", "<think>unclosed"):
            with self.subTest(case=case):
                self.assertEqual(
                    tools.split_think(case), self.original.split_think(case)
                )

    def test_schema_matches(self):
        self.assertEqual(tools.TOOLS, self.original.TOOLS)
        self.assertEqual(tools.MUTATING_TOOLS, self.original.MUTATING_TOOLS)


if __name__ == "__main__":
    unittest.main()
