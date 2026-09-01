#!/usr/bin/env python3
"""frankenstein_engine.py - ContextVM V0.

A stateful, single-process MLX engine for Frankenstein E2.

Invariant proved by this stage:

    model, tokenizer and prompt cache are created once.
    Every turn appends new tokens only.
    Prior conversation tokens are never processed twice.

Modes:
    --mode selftest   tokenizer-only check of the append-only segment builder
    --mode demo       three-turn tool cycle with a synthetic tool result
    --mode agent      real read-only MacBat coding-agent benchmark
"""

import argparse, fnmatch, json, os, re, subprocess, sys, tempfile, time
from dataclasses import dataclass, asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load_tokenizer

def patch_lm_head_last_token():
    """Apply lm_head only to the final position.

    MEASURED: no benefit. MLX is lazy, and generate_step discards the prefill
    logits without evaluating them, so the lm_head matmul for those chunks
    never runs. Peak memory and prefill speed are unchanged. Kept only for
    reference; off by default because it changes the output shape for nothing.

    Every mlx_lm generation path uses only the last position, so returning
    [B, 1, vocab] is safe here. It would not be safe for training or for
    scoring whole sequences.
    """
    import mlx_lm.models.qwen3_5 as q5
    if getattr(q5, "_lm_head_patched", False):
        return
    def call(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        out = out[:, -1:, :]
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)
    q5.TextModel.__call__ = call
    q5._lm_head_patched = True


def patch_wired_limit(limit_gb):
    """Cap how much MLX pins as wired memory.

    mlx_lm wires up to the whole recommended working set (15.73 GB here).
    Wired pages cannot be reclaimed by macOS, so the rest of the system is
    starved while the model is loaded. A lower cap lets macOS page parts of
    the model out when other apps need memory: the Mac stays usable, the
    model slows down under contention.
    """
    import contextlib
    import mlx_lm.generate as g

    @contextlib.contextmanager
    def limited(model, streams=None):
        old = mx.set_wired_limit(int(limit_gb * 1e9))
        try:
            yield
        finally:
            mx.synchronize()
            mx.set_wired_limit(old)
    g.wired_limit = limited


# Paged KV keeps physical memory flat while logical context grows. Tuned for a
# 32-64K working target on a 16 GB machine: fp16 pages so attention uses the
# fused kernel, cold pages on SSD, only a small resident set.
try:
    from models.qwen27b.paged_kv import (
        PagedKVCache,
        install as install_paged,
        make_paged_cache,
    )
except ImportError:
    PagedKVCache = None

E2 = str(Path.home() / "models/Qwen3.8-27B-Apple-MLX-V3.1-Compact")
MACBAT = os.environ.get("MACQWEN_WORKSPACE", str(Path.cwd()))

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# ----------------------------------------------------------------------------
# agent contract (identical wording to macbat_readonly_agent_v3.py)
# ----------------------------------------------------------------------------

SKIP = {".git", ".build", "DerivedData", ".swiftpm", "node_modules", "Pods", ".idea", ".vscode"}
TEXT = {".swift", ".m", ".mm", ".h", ".hpp", ".c", ".cc", ".cpp", ".plist", ".json", ".yaml",
        ".yml", ".toml", ".md", ".txt", ".entitlements", ".xcconfig", ".pbxproj", ".sh", ".py"}

SYSTEM = """You are a senior Swift/macOS engineer working in a real production repository.
Inspect relevant code before making claims or changes. Use precise edits and verify them.

Work efficiently: locate main.swift first; use search and targeted reads; inspect related types/functions as needed; do not re-read material unnecessarily; once you have enough evidence, stop exploring and write the review. Keep reasoning focused and choose the next tool promptly.

Final review:
1. Explain what main.swift is responsible for.
2. Identify concrete bugs, correctness risks, concurrency issues, lifecycle problems, or architectural weaknesses.
3. Suggest only justified performance/memory improvements.
4. Give Swift/macOS-specific improvements.
5. Rank refactoring opportunities by impact.
6. Call out unnecessary, fragile, duplicated, or overly complex code.
7. Finish with the five changes you would make first.

For every important finding cite the relative file path and relevant type/function/code region. Distinguish definite bugs from risks, tradeoffs, and style preferences."""

TASK = """Review main.swift in the selected workspace. Inspect related files as needed, then give the complete review requested by the system instructions."""

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

# The model reaches for reasonable synonyms. Accepting them saves a whole
# round trip per mistake instead of bouncing an error back.
PARAM_ALIASES = {
    "web_search": {"q": "query", "search": "query", "question": "query"},
    "find_files": {"query": "pattern", "name": "pattern", "glob": "pattern",
                   "filename": "pattern", "file": "pattern"},
    "list_dir":   {"dir": "path", "directory": "path", "folder": "path"},
    "read_file":  {"file": "path", "filename": "path", "filepath": "path",
                   "start": "start_line", "end": "end_line",
                   "from_line": "start_line", "to_line": "end_line"},
    "search":     {"pattern": "query", "text": "query", "term": "query",
                   "dir": "path", "directory": "path"},
    "write_file": {"file": "path", "filename": "path", "text": "content"},
    "replace_text": {"file": "path", "filename": "path", "old": "old_text",
                     "new": "new_text", "count": "expected_occurrences"},
    "run_command": {"cmd": "command", "timeout": "timeout_seconds"},
}


class Repo:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise SystemExit(f"Missing repo: {self.root}")

    def safe(self, rel="."):
        p = (self.root / (rel or ".")).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise ValueError("path escapes repository")
        return p

    def walk(self, start="."):
        b = self.safe(start)
        for d, ds, fs in os.walk(b):
            ds[:] = [x for x in ds if x not in SKIP and not x.startswith(".git")]
            yield Path(d), fs

    def find_files(self, pattern):
        out = []
        pat = pattern.lower()
        for d, fs in self.walk():
            for n in fs:
                rel = str((d / n).relative_to(self.root))
                if fnmatch.fnmatch(n.lower(), pat) or fnmatch.fnmatch(rel.lower(), pat):
                    out.append(rel)
                    if len(out) >= 200:
                        return {"matches": out, "truncated": True}
        return {"matches": out, "truncated": False}

    def list_dir(self, path="."):
        p = self.safe(path)
        if not p.is_dir():
            raise ValueError("not a directory")
        e = []
        for x in sorted(p.iterdir(), key=lambda q: (not q.is_dir(), q.name.lower())):
            if x.name in SKIP:
                continue
            e.append({"name": x.name, "type": "dir" if x.is_dir() else "file"})
            if len(e) >= 200:
                break
        return {"path": str(p.relative_to(self.root)), "entries": e}

    def read_file(self, path, start_line=None, end_line=None):
        p = self.safe(path)
        if not p.is_file():
            raise ValueError("not a file")
        if p.stat().st_size > 2_000_000:
            raise ValueError("file >2MB; search first")
        lines = p.read_text(errors="replace").splitlines()
        # No range still supports short files. Cap large reads to control the
        # neural prefill cost when the model ignores the targeted-read rule.
        a = 1 if start_line is None else max(1, int(start_line))
        end = len(lines) if end_line is None else max(a, int(end_line))
        cap = 240 if start_line is None and end_line is None else 800
        b = min(len(lines), end, a + cap - 1)
        res = {"path": str(p.relative_to(self.root)), "start_line": a, "end_line": b,
               "total_lines": len(lines),
               "content": "\n".join(f"{i:5d} | {lines[i-1]}" for i in range(a, b + 1))}
        if b >= len(lines):
            res["note"] = (f"END OF FILE. This file has {len(lines)} lines and you "
                           f"have now seen through line {b}. Re-read only a specific "
                           f"range if later reasoning conflicts with exact source syntax.")
        return res

    def search(self, query, path=".", glob="*"):
        out = []
        q = query.lower()
        for d, fs in self.walk(path):
            for n in fs:
                if not fnmatch.fnmatch(n, glob):
                    continue
                p = d / n
                if p.suffix.lower() not in TEXT and n != "project.pbxproj":
                    continue
                try:
                    if p.stat().st_size > 2_000_000:
                        continue
                    lines = p.read_text(errors="replace").splitlines()
                except Exception:
                    continue
                for i, line in enumerate(lines, 1):
                    if q in line.lower():
                        out.append({"path": str(p.relative_to(self.root)), "line": i, "text": line[:500]})
                        if len(out) >= 100:
                            return {"matches": out, "truncated": True}
        return {"matches": out, "truncated": False}

    def _atomic_write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = path.stat().st_mode if path.exists() else None
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
            prefix=f".{path.name}.", suffix=".tmp"
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)

    def _api_check(self, path, content):
        """Veto code that does not verify against a real checker.

        A prompt asking the model to confirm unfamiliar APIs is advice, and it
        already ignored it. This is a gate: the write fails and the model gets
        the checker's own words back. Languages without a checker installed
        pass through untouched, so this never blocks on ignorance.
        """
        try:
            from macqwen.tools.code_check import check, report
        except ImportError:
            return []
        errors, warnings = check(path, content, project_root=self.root)
        if errors:
            # An index cannot see every gem, mixin or metaprogrammed method.
            # Refuse once with the specific names; if the model comes back with
            # the same complaint on the same file, believe the model and let it
            # through carrying the warning. A guess must never be a locked door.
            key = (str(path), tuple(sorted(errors)))
            seen = getattr(self, "_refused", None)
            if seen is None:
                seen = self._refused = {}
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 1:
                raise ValueError(report(path, errors))
            return [e + " (index may be wrong; written on your second attempt)"
                    for e in errors]
        return warnings

    def write_file(self, path, content, create_parents=True):
        """Create a file, making its parent directories by default.

        This used to default to false, so the first write into a new folder
        always failed and cost a whole generation round trip to learn that the
        folder was missing. `safe()` already confines the path to the
        workspace, so creating a directory inside it destroys nothing.
        """
        p = self.safe(path)
        if p.exists():
            raise ValueError("file already exists; use replace_text")
        if not p.parent.is_dir() and not create_parents:
            raise ValueError("parent directory does not exist; set create_parents=true")
        if len(content.encode("utf-8")) > 2_000_000:
            raise ValueError("content exceeds 2MB")
        warnings = self._api_check(p, content)
        self._atomic_write(p, content)
        result = {"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size,
                  "created": True}
        if warnings:
            result["warnings"] = warnings
        return result

    def replace_text(self, path, old_text, new_text, expected_occurrences=1):
        p = self.safe(path)
        if not p.is_file():
            raise ValueError("not a file")
        if p.stat().st_size > 2_000_000:
            raise ValueError("file >2MB")
        if not old_text:
            raise ValueError("old_text cannot be empty")
        expected = max(1, int(expected_occurrences))
        content = p.read_text(encoding="utf-8")
        actual = content.count(old_text)
        if actual != expected:
            raise ValueError(f"expected {expected} occurrence(s), found {actual}; read the file again")
        updated = content.replace(old_text, new_text, expected)
        if len(updated.encode("utf-8")) > 2_000_000:
            raise ValueError("updated file exceeds 2MB")
        # Check the whole updated file, not the fragment: a helper defined
        # elsewhere in the file must not read as an invented method.
        warnings = self._api_check(p, updated)
        self._atomic_write(p, updated)
        result = {"path": str(p.relative_to(self.root)), "replacements": expected,
                  "bytes": p.stat().st_size}
        if warnings:
            result["warnings"] = warnings
        return result

    def run_command(self, command, timeout_seconds=120):
        timeout = min(max(1, int(timeout_seconds)), 300)
        environment = {
            key: value for key, value in os.environ.items()
            if not re.search(r"TOKEN|SECRET|PASSWORD|API_KEY", key, re.I)
        }
        try:
            result = subprocess.run(
                ["/bin/zsh", "-c", command], cwd=self.root,
                capture_output=True, text=True, timeout=timeout,
                env=environment, errors="replace",
            )
            stdout, stderr = result.stdout, result.stderr
            limit = 30_000
            truncated = len(stdout) > limit or len(stderr) > limit
            return {
                "exit_code": result.returncode,
                "stdout": stdout[-limit:],
                "stderr": stderr[-limit:],
                "truncated": truncated,
                "cwd": str(self.root),
            }
        except subprocess.TimeoutExpired as error:
            return {
                "exit_code": 124,
                "stdout": (error.stdout or "")[-30_000:],
                "stderr": ((error.stderr or "") + f"\nTimed out after {timeout}s")[-30_000:],
                "truncated": False,
                "cwd": str(self.root),
            }

    def call(self, name, args):
        if name not in PARAM_TYPES:
            raise ValueError(f"unknown tool {name}; available: {sorted(PARAM_TYPES)}")
        allowed = PARAM_TYPES[name]
        alias = PARAM_ALIASES.get(name, {})
        for wrong, right in alias.items():
            if wrong in args and right not in args:
                args[right] = args.pop(wrong)
        bad = sorted(k for k in args if k not in allowed)
        if bad:
            raise ValueError(f"unknown parameter(s) {bad} for {name}; "
                             f"expected parameters: {sorted(allowed)}")
        return getattr(self, name)(**args)


# ----------------------------------------------------------------------------
# Qwen3.5 tool-call format
# ----------------------------------------------------------------------------

CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
# The closing </function> is optional. Qwen routinely closes the tags out of
# order, e.g. <tool_call><function=list_dir></tool_call></function></tool_call>,
# which puts </function> outside the block. Requiring it dropped the call
# silently, so the model saw no result and reissued the same call forever.
FUNCTION_RE = re.compile(
    r"\s*<function=([^>\s]+)>(.*?)(?:</function>|</>)?\s*$", re.S
)
SHORT_FUNCTION_RE = re.compile(
    r"\s*<([A-Za-z_][A-Za-z0-9_]*)>(.*?)(?:</function>|</>)?\s*$", re.S)
# Tags that leak into a body when the model closes things out of order.
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


# ----------------------------------------------------------------------------
# host memory telemetry
# ----------------------------------------------------------------------------

# The model announces an intention to look something up, then keeps reasoning
# instead. Observed: four declarations across 2272 tokens and no call, ending in
# an invented method name. Thinking cannot recover a signature it never knew, so
# the only useful recovery is to make it issue the call.
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


def host_mem():
    """Return (free_gb, swap_used_gb) for the whole machine."""
    free = swap = 0.0
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 16384
        m = re.search(r"page size of (\d+)", vm)
        if m:
            page = int(m.group(1))
        def pages(label):
            mm = re.search(rf"{label}:\s+(\d+)", vm)
            return int(mm.group(1)) if mm else 0
        free = (pages("Pages free") + pages("Pages inactive") + pages("Pages purgeable")) * page / 1e9
    except Exception:
        pass
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"used\s*=\s*([\d.]+)([MG])", sw)
        if m:
            swap = float(m.group(1)) / (1000.0 if m.group(2) == "M" else 1.0)
    except Exception:
        pass
    return free, swap


# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------

@dataclass
class TurnStats:
    turn: int
    new_prompt_tokens: int
    prompt_tps: float
    gen_tokens: int
    gen_tps: float
    logical_tokens: int
    cache_tokens: int
    cache_gb: float
    attn_cache_gb: float
    active_gb: float
    pool_gb: float
    peak_gb: float
    host_free_gb: float
    swap_gb: float
    seconds: float
    finish: str


class FrankensteinEngine:
    def __init__(self, model_path=E2, prefill_step_size=512, kv_bits=None,
                 kv_group_size=64, quantized_kv_start=1024, temperature=0.0,
                 repetition_penalty=None, repetition_context_size=64,
                 backtrack_bias=0.0,
                 loop_guard=True, tokenizer_only=False,
                 paged=False, page_size=256, top_k_pages=16,
                 resident_pages=24, spill_dir=None, min_context=16384,
                 lm_head_last=False, wired_limit_gb=None, layer_indices=None,
                 bf16_ends=False, shortlist_k=1024):
        self.path = Path(model_path)
        # The cap has to be set before the weights arrive: loading is what runs
        # out of memory, and patching the generation-time context manager was
        # applying it far too late.
        #
        # Lazy, memory-mapped weights were tried here and are NOT safe. Under
        # memory pressure the forward pass stalls waiting on page faults, a
        # single Metal command buffer runs past the GPU watchdog, and the whole
        # generation dies with kIOGPUCommandBufferCallbackErrorTimeout. A clean
        # failure at load is better than a crash mid-answer, so weights load
        # eagerly and a model that does not fit simply does not run.
        self.lazy_weights = False
        if wired_limit_gb:
            mx.set_wired_limit(int(float(wired_limit_gb) * 1e9))
        if tokenizer_only:
            self.model = None
            self.tokenizer = load_tokenizer(self.path)
        else:
            if bf16_ends:
                # The shortlist head ranks a single position, so the last-token
                # patch is required here, not optional.
                patch_lm_head_last_token()
            if bf16_ends:
                # A loader that never materialises the 2-bit embedding this
                # build ships and immediately discards. Worth 0.40 GB of peak,
                # and peak is what decides whether a build fits at all.
                from models.qwen27b.bf16_ends import load_v4_lean
                self.model, self.tokenizer = load_v4_lean(model_path,
                                                          k=shortlist_k)
            else:
                self.model, self.tokenizer = load(model_path,
                                                  lazy=self.lazy_weights)
        self.layer_indices = None
        if layer_indices and self.model is not None:
            text_model = self.model.language_model.model
            original = list(text_model.layers)
            indices = [int(value) for value in layer_indices.split(",") if value.strip()]
            if not indices or indices != sorted(set(indices)):
                raise ValueError("layer indices must be unique and sorted")
            if indices[0] < 0 or indices[-1] >= len(original):
                raise ValueError("layer index outside model range")
            text_model.layers = [original[index] for index in indices]
            text_model.ssm_idx = next(
                (index for index, layer in enumerate(text_model.layers) if layer.is_linear),
                None,
            )
            text_model.fa_idx = next(
                (index for index, layer in enumerate(text_model.layers) if not layer.is_linear),
                None,
            )
            self.layer_indices = indices
        self._ensure_chat_template()
        self.sampler = make_sampler(temp=temperature)
        # Backtracking words are what circular reasoning is made of: the model
        # writes "Wait", "Actually", "Hmm" and then re-litigates a conclusion it
        # already reached. Biasing them down shortens reasoning without
        # forbidding a genuine correction, since the bias is a nudge and not a ban.
        bias = None
        if backtrack_bias:
            bias = {}
            for word in ("Wait", " Wait", "Actually", " Actually", "Hmm", " Hmm",
                         " However", " reconsider", " Alternatively", " But wait"):
                try:
                    ids = self.tokenizer.encode(word, add_special_tokens=False)
                except Exception:
                    continue
                if len(ids) == 1:
                    bias[ids[0]] = -abs(backtrack_bias)
        self.logits_processors = make_logits_processors(
            logit_bias=bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size) \
            if (repetition_penalty or bias) else None
        self.loop_guard = loop_guard
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        if lm_head_last:
            patch_lm_head_last_token()
        if wired_limit_gb:
            patch_wired_limit(wired_limit_gb)
        self.paged = paged and PagedKVCache is not None and self.model is not None
        if self.paged:
            install_paged()
            self.cache = make_paged_cache(
                self.model, page_size, top_k_pages=top_k_pages,
                pinned_pages=1, recent_pages=2, refresh_every=16,
                min_context=min_context, spill_dir=spill_dir,
                resident_pages=resident_pages)
        else:
            self.cache = make_prompt_cache(self.model) if self.model is not None else []
        self.tape = []       # token ids already inside the cache
        self.pending = []    # token ids appended but not processed yet
        self.turn_closed = True   # last assistant turn ended with <|im_end|>
        self.turn = 0
        self.stats = []

    # -- chat template ------------------------------------------------------

    def _ensure_chat_template(self):
        if getattr(self.tokenizer, "chat_template", None):
            return
        f = self.path / "chat_template.jinja"
        if f.is_file():
            self.tokenizer.chat_template = f.read_text()
        else:
            raise SystemExit("model has no chat template")

    # -- append-only segment builders --------------------------------------

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def append_text(self, text):
        ids = self.encode(text)
        self.pending.extend(ids)
        return len(ids)

    def append_tokens(self, ids):
        """Append token IDs produced by this exact tokenizer."""
        self.pending.extend(int(token) for token in ids)
        return len(ids)

    def open_conversation(self, system, user, tools=None, enable_thinking=True,
                          reasoning_effort="xhigh"):
        """First segment: system (+ tool contract), first user turn, generation prompt."""
        if self.tape or self.pending:
            raise RuntimeError("conversation already open")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            msgs, tools=tools, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort)
        return self.append_text(text)

    def _close(self):
        """Close a truncated assistant turn. Generation stopped before <|im_end|>."""
        return "" if self.turn_closed else IM_END

    @staticmethod
    def _assistant_prefix(enable_thinking=True):
        if enable_thinking:
            return f"{IM_START}assistant\n<think>\n"
        return f"{IM_START}assistant\n<think>\n\n</think>\n\n"

    def append_user(self, text, enable_thinking=True):
        return self.append_text(
            f"{self._close()}\n{IM_START}user\n{text}{IM_END}\n"
            f"{self._assistant_prefix(enable_thinking)}")

    def append_tool_results(self, results, enable_thinking=True):
        body = "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
        return self.append_text(
            f"{self._close()}\n{IM_START}user{body}{IM_END}\n"
            f"{self._assistant_prefix(enable_thinking)}")

    # -- telemetry ----------------------------------------------------------

    @property
    def cache_tokens(self):
        for c in self.cache:
            if hasattr(c, "offset"):
                return int(c.offset)
        return 0

    def cache_bytes(self):
        total = attn = 0
        for c in self.cache:
            n = c.nbytes
            total += n
            if hasattr(c, "offset"):
                attn += n
        return total, attn

    # -- generation ---------------------------------------------------------

    def generate(self, max_tokens=1600, echo=True, out=print, progress=None,
                 on_token=None):
        """Process the pending tokens only, then stream one assistant turn."""
        if not self.pending:
            raise RuntimeError("nothing to process")
        n_new = len(self.pending)
        prompt = mx.array(self.pending)
        t0 = time.perf_counter()
        parts, tokens = [], []
        stats = {"prompt_tps": 0.0, "gen_tps": 0.0, "finish": "?", "peak": 0.0}
        interrupted = False
        # `stream_generate` is lazy, so whatever this loop body does happens
        # before the next token is asked for, and mlx_lm counts it as model
        # time. The terminal fade lives in `on_token` and its cost varies per
        # word, which made the reported rate wander. Measure the body and take
        # it back out.
        gen_began = None
        body_seconds = 0.0
        try:
          for r in stream_generate(
            self.model, self.tokenizer, prompt,
            max_tokens=max_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
            prompt_cache=self.cache,
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            quantized_kv_start=self.quantized_kv_start,
            prompt_progress_callback=progress,
        ):
            body_began = time.perf_counter()
            try:
                if gen_began is None:
                    gen_began = body_began
                tokens.append(r.token)
                parts.append(r.text)
                if echo and r.text:
                    out(r.text, end="")
                stats["prompt_tps"] = r.prompt_tps
                stats["gen_tps"] = r.generation_tps
                stats["peak"] = r.peak_memory
                if r.finish_reason:
                    stats["finish"] = r.finish_reason
                if on_token is not None:
                    if on_token(len(tokens), r) is False:
                        stats["finish"] = "callback"
                        break
                if self.loop_guard and len(tokens) % 32 == 0:
                    reason = detect_loop("".join(parts))
                    if reason:
                        stats["finish"] = reason
                        if echo:
                            out("\n[loop guard: "
                                + ("declared a lookup and never made it"
                                   if reason == "loop:intent" else "repeated tail")
                                + ", stopping this turn]")
                        break
            finally:
                body_seconds += time.perf_counter() - body_began
        except KeyboardInterrupt:
            # Every token yielded is already in the cache, so stopping here
            # leaves tape and cache consistent. The turn stays open and the
            # next segment closes it with <|im_end|>.
            interrupted = True
            stats["finish"] = "interrupted"
        wall = time.perf_counter() - t0
        if gen_began is not None and len(tokens) > 1:
            span = time.perf_counter() - gen_began - body_seconds
            if span > 0:
                stats["gen_tps"] = (len(tokens) - 1) / span

        # every token yielded by the generator is already inside the cache
        self.tape.extend(self.pending)
        self.pending = []
        self.tape.extend(tokens)
        self.turn_closed = stats["finish"] == "stop"
        self.turn += 1

        total, attn = self.cache_bytes()
        free, swap = host_mem()
        st = TurnStats(
            turn=self.turn, new_prompt_tokens=n_new, prompt_tps=stats["prompt_tps"],
            gen_tokens=len(tokens), gen_tps=stats["gen_tps"],
            logical_tokens=len(self.tape), cache_tokens=self.cache_tokens,
            cache_gb=total / 1e9, attn_cache_gb=attn / 1e9,
            active_gb=mx.get_active_memory() / 1e9, peak_gb=stats["peak"],
            pool_gb=mx.get_cache_memory() / 1e9,
            host_free_gb=free, swap_gb=swap, seconds=wall, finish=stats["finish"])
        self.stats.append(st)
        return "".join(parts), st

    def check_invariant(self):
        """Cache length must equal the logical tape length."""
        return self.cache_tokens == len(self.tape)


def split_think(text):
    i = text.find("</think>")
    if i < 0:
        return text, ""
    return text[:i], text[i + len("</think>"):]


def fmt_stats(s):
    return (f"turn {s.turn:>2} | new {s.new_prompt_tokens:>6} tok @ {s.prompt_tps:>6.1f} t/s"
            f" | gen {s.gen_tokens:>5} @ {s.gen_tps:>5.1f} t/s"
            f" | ctx {s.logical_tokens:>7} | kv {s.cache_gb:>5.2f} GB (attn {s.attn_cache_gb:>5.2f})"
            f" | mlx act {s.active_gb:>5.2f} pool {s.pool_gb:>4.2f} peak {s.peak_gb:>5.2f} GB"
            f" | free {s.host_free_gb:>5.2f} swap {s.swap_gb:>5.2f} GB"
            f" | {s.seconds:>6.1f}s | {s.finish}")


# ----------------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------------

def mode_selftest(a):
    """Tokenizer-only: the append-only tape must equal the full template render."""
    eng = FrankensteinEngine(a.model, tokenizer_only=True)
    tok = eng.tokenizer

    answer = "<think>\nI must read the file.\n</think>\n\nLooking now.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nSources/main.swift\n</parameter>\n<parameter=start_line>\n1\n</parameter>\n<parameter=end_line>\n80\n</parameter>\n</function>\n</tool_call>"
    result = '{"path": "Sources/main.swift", "start_line": 1, "end_line": 2, "total_lines": 2}\n    1 | import Foundation\n    2 | print("hi")'

    # incremental tape, exactly what the engine feeds the model
    eng.open_conversation(SYSTEM, TASK, tools=TOOLS)
    incremental = list(eng.pending)
    eng.tape.extend(eng.pending)
    eng.pending = []
    incremental += tok.encode(answer[len("<think>\n"):] + IM_END, add_special_tokens=False)
    eng.tape = list(incremental)
    eng.append_tool_results([result])
    incremental += eng.pending

    # reference render through the chat template
    reason, content = split_think(answer)
    calls = parse_tool_calls(content)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK},
        {"role": "assistant", "content": content.split("<tool_call>")[0].strip(),
         "reasoning_content": reason.replace("<think>\n", "").strip(),
         "tool_calls": [{"type": "function", "function": {"name": n, "arguments": ar}} for n, ar in calls]},
        {"role": "tool", "content": result},
    ]
    ref_text = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=True)
    reference = tok.encode(ref_text, add_special_tokens=False)

    print(f"parsed tool calls : {calls}")
    print(f"incremental tokens: {len(incremental)}")
    print(f"template tokens   : {len(reference)}")
    ok = incremental == reference
    if not ok:
        n = min(len(incremental), len(reference))
        i = next((k for k in range(n) if incremental[k] != reference[k]), n)
        print(f"first difference at index {i}")
        print("incremental:", repr(tok.decode(incremental[max(0, i - 40):i + 60])))
        print("reference  :", repr(tok.decode(reference[max(0, i - 40):i + 60])))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def mode_demo(a):
    out = Printer(a.log)
    eng = build_engine(a, out)
    out(f"\n{'='*100}\nTURN 1 - first user request\n{'='*100}")
    eng.open_conversation("You are a concise assistant.",
                          "Say hello in one short sentence, then stop.", tools=None,
                          reasoning_effort=a.reasoning_effort)
    text, st = eng.generate(max_tokens=a.max_tokens, out=out)
    out("\n" + fmt_stats(st))

    out(f"\n{'='*100}\nTURN 2 - appended user message (only new tokens must be processed)\n{'='*100}")
    eng.append_user("Now repeat my first request back to me in five words.")
    text, st = eng.generate(max_tokens=a.max_tokens, out=out)
    out("\n" + fmt_stats(st))

    out(f"\n{'='*100}\nTURN 3 - synthetic tool result\n{'='*100}")
    eng.append_tool_results(['{"battery": 87, "charging": false}'])
    text, st = eng.generate(max_tokens=a.max_tokens, out=out)
    out("\n" + fmt_stats(st))

    out(f"\ninvariant cache==tape: {eng.check_invariant()} "
        f"(cache {eng.cache_tokens}, tape {len(eng.tape)})")
    summary(eng, out)
    return 0


def mode_agent(a):
    out = Printer(a.log)
    repo = Repo(a.root)
    eng = build_engine(a, out)
    out(f"ROOT : {repo.root}")
    out(f"TASK : MacBat read-only review benchmark")

    eng.open_conversation(SYSTEM, TASK, tools=TOOLS, reasoning_effort=a.reasoning_effort)
    swap0 = host_mem()[1]
    forced = 0
    stop = None
    for turn in range(1, a.max_turns + 1):
        out(f"\n{'#'*30} TURN {turn} {'#'*30}")
        out(f"[appending {len(eng.pending)} new tokens]")
        text, st = eng.generate(max_tokens=a.max_tokens, out=out)
        out("\n" + fmt_stats(st))
        if not eng.check_invariant():
            out(f"!! INVARIANT BROKEN: cache {eng.cache_tokens} tape {len(eng.tape)}")
            stop = "invariant"
            break
        if st.host_free_gb and st.host_free_gb < a.min_free_gb:
            out(f"!! host free memory {st.host_free_gb:.2f} GB below --min-free-gb; stopping")
            stop = "memory"
            break
        if st.swap_gb - swap0 > a.max_swap_growth_gb:
            out(f"!! swap grew {st.swap_gb - swap0:.2f} GB since start; stopping")
            stop = "swap"
            break

        _, content = split_think(text)
        calls = parse_tool_calls(content) or parse_tool_calls(text)
        if not calls:
            if st.finish == "stop":
                out("\n[final answer produced]")
                stop = "answer"
                break
            if "</think>" not in text and forced < a.max_forced_closes:
                forced += 1
                out(f"\n[turn ended as '{st.finish}' inside <think>; forcing closure {forced}]")
                eng.append_text("\n</think>\n\n")
                continue
            out(f"\n[turn ended as '{st.finish}' with no tool call]")
            stop = "truncated"
            break
        forced = 0
        results = []
        for name, args in calls:
            out(f"\n[tool] {name}({json.dumps(args, ensure_ascii=False)[:200]})")
            try:
                res = repo.call(name, args)
                results.append(render_tool_result(name, res, a.tool_format))
            except Exception as e:
                results.append(json.dumps({"error": f"{type(e).__name__}: {e}"}))
                out(f"[tool error] {e}")
        eng.append_tool_results(results)
    else:
        stop = "max-turns"
    out(f"\nSTOP: {stop}")
    summary(eng, out)
    return 0


class Printer:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1) if path else None

    def __call__(self, s="", end="\n"):
        print(s, end=end, flush=True)
        if self.f:
            print(s, end=end, file=self.f, flush=True)


def build_engine(a, out):
    free, swap = host_mem()
    if free < a.min_start_free_gb:
        out(f"ABORT: only {free:.2f} GB free, need {a.min_start_free_gb:.1f} GB.")
        out("Close apps and retry. Loading now would swap and freeze the Mac.")
        sys.exit(2)
    out(f"MODEL: {a.model}")
    out(f"PAGED: {a.paged} page={a.page_size} top_k={a.top_k_pages} "
        f"resident={a.resident_pages} min_ctx={a.min_context}")
    out(f"KV   : bits={a.kv_bits} group={a.kv_group_size} start={a.quantized_kv_start} "
        f"prefill={a.prefill_step_size} temp={a.temperature} "
        f"rep_penalty={a.repetition_penalty} loop_guard={not a.no_loop_guard}")
    out(f"HOST : free {free:.2f} GB, swap {swap:.2f} GB")
    t0 = time.perf_counter()
    eng = FrankensteinEngine(a.model, prefill_step_size=a.prefill_step_size,
                             kv_bits=a.kv_bits, kv_group_size=a.kv_group_size,
                             quantized_kv_start=a.quantized_kv_start,
                             temperature=a.temperature,
                             repetition_penalty=a.repetition_penalty,
                             repetition_context_size=a.repetition_context_size,
                             loop_guard=not a.no_loop_guard,
                             paged=a.paged, page_size=a.page_size,
                             top_k_pages=a.top_k_pages,
                             resident_pages=a.resident_pages,
                             spill_dir=a.spill_dir, min_context=a.min_context)
    out(f"LOAD : {time.perf_counter()-t0:.1f}s, mlx active {mx.get_active_memory()/1e9:.2f} GB")
    return eng


def summary(eng, out):
    out(f"\n{'='*100}\nPER-TURN LEDGER\n{'='*100}")
    for s in eng.stats:
        out(fmt_stats(s))
    total_new = sum(s.new_prompt_tokens for s in eng.stats)
    naive = sum(s.logical_tokens - s.gen_tokens for s in eng.stats)
    out(f"\nnew prompt tokens processed : {total_new}")
    out(f"stateless re-prefill would be: {naive}")
    if total_new:
        out(f"prefill saved               : {naive - total_new} tokens "
            f"({(1 - total_new / max(naive,1))*100:.1f}%)")
    out(f"final logical context       : {len(eng.tape)} tokens")
    out(f"cache/tape invariant        : {eng.check_invariant()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["selftest", "demo", "agent"], default="agent")
    p.add_argument("--model", default=E2)
    p.add_argument("--root", default=MACBAT)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=1600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prefill-step-size", type=int, default=512)
    p.add_argument("--kv-bits", type=int, default=None)
    p.add_argument("--kv-group-size", type=int, default=64)
    p.add_argument("--quantized-kv-start", type=int, default=1024)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--repetition-context-size", type=int, default=64)
    p.add_argument("--no-loop-guard", action="store_true")
    p.add_argument("--max-forced-closes", type=int, default=2)
    p.add_argument("--paged", action="store_true",
                   help="paged KV with SSD spill; physical memory stops tracking context")
    p.add_argument("--page-size", type=int, default=256)
    p.add_argument("--top-k-pages", type=int, default=16)
    p.add_argument("--resident-pages", type=int, default=24)
    p.add_argument("--min-context", type=int, default=16384)
    p.add_argument("--spill-dir", default=None)
    p.add_argument("--min-start-free-gb", type=float, default=8.0,
                   help="refuse to load the model below this much free memory")
    p.add_argument("--min-free-gb", type=float, default=0.35)
    p.add_argument("--max-swap-growth-gb", type=float, default=3.0)
    p.add_argument("--reasoning-effort", choices=["xhigh", "medium", "low"], default="xhigh")
    p.add_argument("--tool-format", choices=["pretty", "json"], default="pretty")
    p.add_argument("--log", default=None)
    a = p.parse_args()
    if a.log is None and a.mode != "selftest":
        a.log = f"contextvm_v0_{a.mode}_{int(time.time())}.log"
    return {"selftest": mode_selftest, "demo": mode_demo, "agent": mode_agent}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
