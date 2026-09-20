"""Tool-call parsing, including a differential test against the original.

macqwen.tools was extracted from frankenstein_engine.py. Extraction is only
safe if the behaviour is identical, so where the original can still be
imported these tests compare the two directly on the awkward inputs that
motivated the parser's leniency.
"""
from __future__ import annotations

import json
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

    def test_whitespace_sensitive_arguments_survive_verbatim(self):
        # Only the single framing newline is protocol; indentation and
        # extra trailing newlines are payload and must survive for
        # write_file/replace_text to act on exact bytes.
        raw = (
            "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n"
            "</parameter>\n<parameter=content>\n  indented\nline2\n\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        (name, args), = tools.parse_tool_calls(raw)
        self.assertEqual(name, "write_file")
        self.assertEqual(args["content"], "  indented\nline2\n")

    def test_typed_scalars_still_convert_around_whitespace(self):
        raw = (
            "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n"
            "</parameter>\n<parameter=start_line>\n  3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        (name, args), = tools.parse_tool_calls(raw)
        self.assertEqual(args["start_line"], 3)
        self.assertEqual(args["path"], "notes.txt")

    def test_quoted_and_short_parameters_preserve_payload_bytes(self):
        quoted = (
            '<tool_call><function=replace_text>'
            '<parameter=path="  a.txt  "></parameter>'
            '<parameter=old_text="old  \n\n"></parameter>'
            '</function></tool_call>'
        )
        (name, args), = tools.parse_tool_calls(quoted)
        self.assertEqual(name, "replace_text")
        self.assertEqual(args["path"], "  a.txt  ")
        self.assertEqual(args["old_text"], "old  \n\n")

        short = (
            "<tool_call><write_file>"
            "<path>  a.txt  </path>"
            "<content>  indented\nline2\n\n</content>"
            "</write_file></tool_call>"
        )
        (name, args), = tools.parse_tool_calls(short)
        self.assertEqual(name, "write_file")
        self.assertEqual(args["path"], "  a.txt  ")
        self.assertEqual(args["content"], "  indented\nline2\n")

    def test_literal_protocol_tags_in_payload_are_not_deleted(self):
        raw = (
            "<tool_call><function=write_file>"
            "<parameter=path>a.txt</parameter>"
            "<parameter=content>literal </function> tag</parameter>"
            "</function></tool_call>"
        )
        (name, args), = tools.parse_tool_calls(raw)
        self.assertEqual(name, "write_file")
        self.assertEqual(args["content"], "literal </function> tag")

    def test_api_docs_pretty_render_keeps_documentation_unescaped(self):
        result = {
            "library": "/websites/ruby_sketchup",
            "topic": "pushpull",
            "documentation": "## pushpull(distance, copy = false)\n\nExample",
        }
        rendered = tools.render_tool_result("api_docs", result)
        self.assertEqual(
            rendered,
            '{"library": "/websites/ruby_sketchup", "topic": "pushpull"}'
            "\n## pushpull(distance, copy = false)\n\nExample",
        )
        self.assertNotIn("\\n## pushpull", rendered)
        self.assertEqual(json.loads(tools.render_tool_result(
            "api_docs", result, fmt="json"
        )), result)

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


class CoerceScalarTests(unittest.TestCase):
    """One shared rule for both parsers: strict int, explicit bool."""

    INTEGER_OK = [("3", 3), (" 3 ", 3), ("+3", 3), ("-2", -2), ("007", 7)]
    INTEGER_BAD = ["3.0", "3.7", "banana", "", "nan", "inf", "0x10", "1_0"]
    NUMBER_OK = [("3", 3.0), ("3.0", 3.0), ("3.7", 3.7), (" 2.5 ", 2.5)]
    NUMBER_BAD = ["banana", "", "nan", "inf", "-inf"]
    BOOLEAN_OK = [
        ("true", True), ("True", True), (" TRUE ", True), ("1", True),
        ("yes", True), ("YES", True), ("on", True),
        ("false", False), ("False", False), (" FALSE ", False), ("0", False),
        ("no", False), ("off", False),
    ]
    BOOLEAN_BAD = ["banana", "", "2", "maybe", "truthy"]

    def test_integer_table(self):
        for raw, expected in self.INTEGER_OK:
            with self.subTest(raw=raw):
                self.assertEqual(tools.coerce_scalar(raw, "integer"), expected)
        for raw in self.INTEGER_BAD:
            with self.subTest(raw=raw):
                with self.assertRaises(tools.ToolCallValidationError):
                    tools.coerce_scalar(raw, "integer")

    def test_number_table(self):
        for raw, expected in self.NUMBER_OK:
            with self.subTest(raw=raw):
                self.assertEqual(tools.coerce_scalar(raw, "number"), expected)
        for raw in self.NUMBER_BAD:
            with self.subTest(raw=raw):
                with self.assertRaises(tools.ToolCallValidationError):
                    tools.coerce_scalar(raw, "number")

    def test_boolean_table(self):
        for raw, expected in self.BOOLEAN_OK:
            with self.subTest(raw=raw):
                self.assertIs(tools.coerce_scalar(raw, "boolean"), expected)
        for raw in self.BOOLEAN_BAD:
            with self.subTest(raw=raw):
                with self.assertRaises(tools.ToolCallValidationError):
                    tools.coerce_scalar(raw, "boolean")

    def test_array_and_object_table(self):
        self.assertEqual(tools.coerce_scalar("[1, 2]", "array"), [1, 2])
        self.assertEqual(tools.coerce_scalar("[]", "array"), [])
        self.assertEqual(
            tools.coerce_scalar('{"a": 1}', "object"), {"a": 1}
        )
        self.assertEqual(tools.coerce_scalar("{}", "object"), {})
        for kind, raw in (
            ("array", '{"a": 1}'), ("array", "3"), ("array", "banana"),
            ("array", ""), ("object", "[1]"), ("object", "3"),
            ("object", "banana"), ("object", ""),
        ):
            with self.subTest(kind=kind, raw=raw):
                with self.assertRaises(tools.ToolCallValidationError):
                    tools.coerce_scalar(raw, kind)

    def test_plain_strings_pass_through(self):
        self.assertEqual(tools.coerce_scalar("banana", "string"), "banana")
        self.assertEqual(tools.coerce_scalar("3.7", "string"), "3.7")
        self.assertEqual(
            tools.coerce_scalar("  padded  ", "string"), "  padded  "
        )
        self.assertEqual(tools.coerce_scalar("banana", None), "banana")

    def test_error_names_tool_and_parameter(self):
        with self.assertRaisesRegex(
            tools.ToolCallValidationError, "read_file.*start_line"
        ):
            tools.coerce_scalar("3.7", "integer", tool="read_file",
                                key="start_line")


def _int_call_xml(value, quoted=False):
    param = (
        f'<parameter=start_line="{value}"></parameter>' if quoted
        else f"<parameter=start_line>\n{value}\n</parameter>"
    )
    return (
        "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n"
        f"</parameter>\n{param}\n</function>\n</tool_call>"
    )


class ParseCoercionTests(unittest.TestCase):
    def test_integer_table_through_parser(self):
        for raw, expected in CoerceScalarTests.INTEGER_OK:
            with self.subTest(raw=raw):
                (name, args), = tools.parse_tool_calls(_int_call_xml(raw))
                self.assertEqual(args["start_line"], expected)

    def test_bad_integers_raise_through_parser(self):
        for raw in CoerceScalarTests.INTEGER_BAD:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    tools.ToolCallValidationError, "start_line"
                ):
                    tools.parse_tool_calls(_int_call_xml(raw))

    def test_quoted_params_coerce_equally(self):
        (name, args), = tools.parse_tool_calls(_int_call_xml("3", quoted=True))
        self.assertEqual(args["start_line"], 3)
        with self.assertRaises(tools.ToolCallValidationError):
            tools.parse_tool_calls(_int_call_xml("3.7", quoted=True))

    def test_short_params_coerce_equally(self):
        ok = (
            "<tool_call><function=read_file><parameter=path>notes.txt"
            "</parameter><start_line>3</start_line></function></tool_call>"
        )
        (name, args), = tools.parse_tool_calls(ok)
        self.assertEqual(args["start_line"], 3)
        bad = ok.replace("<start_line>3</start_line>",
                         "<start_line>banana</start_line>")
        with self.assertRaisesRegex(
            tools.ToolCallValidationError, "start_line"
        ):
            tools.parse_tool_calls(bad)


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
