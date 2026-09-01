"""Tool calling, shared by every model.

Both Qwen models emit the same call format, verified against each
tokenizer's chat template:

    <tool_call>
    <function=name>
    <parameter=key>
    value
    </parameter>
    </function>
    </tool_call>

So the schema and the parser live here rather than beside one model's
engine. A model that emits a different format needs its own parser, not a
change to this one.

The parser is deliberately forgiving. Qwen closes </function> and stops
before </tool_call> often enough that strict parsing loses real calls.
"""
from __future__ import annotations

import json
import re

TOOLS = [
 {"type": "function", "function": {"name": "api_docs", "description": "Look up the exact signature of a library or framework method: argument names, order, defaults, return type and a real example. Use this instead of recalling a signature, and before writing any call you have not read in this session. Covers SketchUp, Ruby, Python, SwiftUI, MLX, React and thousands more. Faster and far more reliable than web_search for API questions.", "parameters": {"type": "object", "properties": {"library": {"type": "string"}, "topic": {"type": "string"}}, "required": ["library", "topic"], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "web_search", "description": "Search the public internet for current external facts. Returns a short answer and up to three source snippets. Use for facts outside the repository. Cite source URLs in the final answer.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "find_files", "description": "Find repository files by case-insensitive glob.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "list_dir", "description": "List a repository directory. Omit path to list the workspace root.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": [], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "read_file", "description": "Read a text file. Use search first, then request a specific line range. Omit the range only for short files.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "search", "description": "Literal case-insensitive search in repository text files.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "write_file", "description": "Create a new UTF-8 text file inside the repository. Parent directories are created automatically. Refuses to overwrite an existing file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "create_parents": {"type": "boolean"}}, "required": ["path", "content"], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "replace_text", "description": "Edit an existing UTF-8 file by replacing exact text. The operation is atomic and fails if the occurrence count differs from expected_occurrences.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}, "expected_occurrences": {"type": "integer"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}}},
 {"type": "function", "function": {"name": "run_command", "description": "Run a zsh command from the repository root. Use this for builds, tests, git inspection, package tools, and approved system changes. Returns exit code, stdout, and stderr.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, "required": ["command"], "additionalProperties": False}}},
]

MUTATING_TOOLS = {"write_file", "replace_text", "run_command"}

PARAM_TYPES = {f["function"]["name"]: {k: v["type"] for k, v in f["function"]["parameters"]["properties"].items()} for f in TOOLS}

REQUIRED_PARAMS = {
    f["function"]["name"]: tuple(f["function"]["parameters"].get("required", ()))
    for f in TOOLS
}

CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

FUNCTION_RE = re.compile(
    r"\s*<function=([^>\s]+)>(.*?)(?:</function>|</>)?\s*$", re.S
)

SHORT_FUNCTION_RE = re.compile(
    r"\s*<([A-Za-z_][A-Za-z0-9_]*)>(.*?)(?:</function>|</>)?\s*$", re.S)

STRAY_TAG_RE = re.compile(r"</?(?:tool_call|function)\s*>")

PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)

PARAM_VALUE_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)=([\"'])(.*?)\2\s*>?\s*</parameter>",
    re.S,
)

SHORT_PARAM_RE = re.compile(
    r"<([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</(?:\1|parameter)>", re.S
)

def parse_tool_calls(text):
    """Accept both Qwen XML tool formats and return [(name, args)]."""
    calls = []
    blocks = CALL_BLOCK_RE.findall(text)
    # Qwen frequently closes </function> and stops before </tool_call>.
    if not blocks and "<tool_call>" in text:
        blocks = [text.rsplit("<tool_call>", 1)[1]]
    for block in blocks:
        match = FUNCTION_RE.match(block) or SHORT_FUNCTION_RE.match(block)
        if not match:
            continue
        name, body = match.groups()
        body = STRAY_TAG_RE.sub("", body)
        allowed = PARAM_TYPES.get(name, {})
        if not allowed:
            continue
        args = {}
        for key, _, raw in PARAM_VALUE_RE.findall(body):
            if key in allowed:
                args[key] = raw.strip()
        for key, raw in PARAM_RE.findall(body):
            if key not in allowed:
                continue
            t = allowed[key]
            v = raw.strip()
            if t == "integer":
                try:
                    v = int(float(v))
                except ValueError:
                    pass
            elif t == "number":
                try:
                    v = float(v)
                except ValueError:
                    pass
            elif t == "boolean":
                v = v.lower() in ("true", "1", "yes")
            args[key] = v
        # Qwen sometimes writes <path>value</parameter> instead of
        # <parameter=path>value</parameter>. Keep the standard form first.
        for key, raw in SHORT_PARAM_RE.findall(body):
            if key in args or key not in allowed:
                continue
            t = allowed[key]
            v = raw.strip()
            if t == "integer":
                try:
                    v = int(float(v))
                except ValueError:
                    pass
            elif t == "number":
                try:
                    v = float(v)
                except ValueError:
                    pass
            elif t == "boolean":
                v = v.lower() in ("true", "1", "yes")
            args[key] = v
        calls.append((name, args))
    return calls

def render_tool_result(name, result, fmt="pretty"):
    if fmt == "json" or not isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    if "content" in result:
        head = {k: v for k, v in result.items() if k != "content"}
        return json.dumps(head, ensure_ascii=False) + "\n" + result["content"]
    return json.dumps(result, ensure_ascii=False)

INTENT_RE = re.compile(
    r"""(?xi)
    # naming the tool at all, while never calling it
    \bweb[_\ -]?search\b
  | # another / one more / a quick lookup
    \b(?:another|one\ more|a\ quick)\ (?:web[_\ -]?search|search|lookup|look-?up)\b
  | # "let me ... search|verify|confirm|check|look up", with filler in between,
    # but not "let me search my memory", which is the opposite of acting
    \b(?:let\ me|i(?:'ll|\ will|\ should|\ must|\ need\ to|\ have\ to)?)
    \s+(?:\w+\s+){0,4}?
    (?:search|verify|confirm|look\s+(?:this|it|that|them)?\s*up|check\s+(?:\w+[.\w]*\s+){0,3}?(?:docs?|api|signature|reference|documentation))
    \b(?!\s+my\s+(?:memory|knowledge|recollection|notes))
    """)

def detect_loop(text, window=220, lookback=1600, min_reps=3):
    """Return a reason string for a stall, or "" when generation is healthy."""
    if len(text) >= window * min_reps:
        tail = text[-window:]
        if text[-lookback:].count(tail) >= min_reps:
            return "loop"

    # Declared an intent to verify and still has not called anything. Three
    # declarations is a stall at any length; two only once it has been talking
    # for a while, so a plan that mentions searching twice is not cut short.
    if "<tool_call>" not in text:
        n = len(INTENT_RE.findall(text))
        if n >= 3 or (n >= 2 and len(text) > 1200):
            return "loop:intent"

    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered[-8000:])
    if len(words) < 160:
        return ""

    # Quantized models often restate a conclusion with small wording changes.
    # Repeated discourse markers catch that stall before thousands of tokens.
    markers = (
        lowered.count("wait, but")
        + lowered.count("but wait")
        + lowered.count("let me re")
        + lowered.count("let me think")
        + lowered.count("reconsider")
        + lowered.count("hmm,")
        + lowered.count("not sure")
    )
    if markers >= 6:
        return "loop"

    # Catch repeated sentence openings while ignoring ordinary function words.
    openings = {}
    for sentence in re.split(r"[.!?\n]+", lowered[-8000:]):
        tokens = re.findall(r"[a-z0-9]+", sentence)
        if len(tokens) < 8:
            continue
        opening = tuple(tokens[:5])
        openings[opening] = openings.get(opening, 0) + 1
        if openings[opening] >= 4:
            return "loop"
    return ""

def split_think(text):
    i = text.find("</think>")
    if i < 0:
        return text, ""
    return text[:i], text[i + len("</think>"):]