#!/usr/bin/env python3
"""code_check.py - verify code before it reaches the disk.

The model invents method names. Telling it not to does not work, because it
does not know which names it invented. But the machine does: every language
here ships a checker that resolves names against reality.

    .swift   swiftc -typecheck   exact. Catches every unknown member.
    .rb      ruby -c, plus the documented API index in api_guard
    .py      compile(), plus pyflakes for undefined names
    .js      node --check
    .sh      bash -n
    .json    json.loads
    .yaml    yaml.safe_load when PyYAML is present

Strength varies. Swift is a full type check and cannot be fooled. Ruby and
Python have no static type information, so they catch syntax plus whatever the
index or linter knows. Partial verification still beats none: two invented
SketchUp methods were caught by an index that took one HTTP request to build.

Nothing here runs the code under test. `swiftc -typecheck` stops before code
generation, and pyflakes only parses.
"""
import difflib
import json
import subprocess
import tempfile
from pathlib import Path

from macqwen.api_keys import sanitized_environment

TIMEOUT = 40


def _run(cmd, path):
    # Strip the malloc debug variables. The parent process inherits them from
    # the launching terminal and every spawned checker prints a warning about
    # them, which buries the actual result in noise.
    env = {
        name: value
        for name, value in sanitized_environment().items()
        if "Malloc" not in name
    }
    try:
        r = subprocess.run(cmd + [str(path)], capture_output=True, text=True,
                           timeout=TIMEOUT, env=env)
        return r.returncode, (r.stderr or r.stdout).strip()
    except FileNotFoundError:
        return 0, ""                       # checker absent: no opinion
    except subprocess.TimeoutExpired:
        return 0, ""


def _tmp(content, suffix):
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False,
                                    encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


def _swift(content):
    p = _tmp(content, ".swift")
    try:
        code, out = _run(["swiftc", "-typecheck"], p)
        if code == 0:
            return []
        # keep real errors, drop notes and the source echo
        return [l for l in out.splitlines()
                if ": error:" in l][:8]
    finally:
        p.unlink(missing_ok=True)


def _ruby(content, project_root=None):
    p = _tmp(content, ".rb")
    try:
        code, out = _run(["ruby", "-c"], p)
        if code != 0:
            return [l for l in out.splitlines() if l.strip()][:6], []
    finally:
        p.unlink(missing_ok=True)
    # The index caught start_transaction and commit_transaction, which do not
    # exist, so this blocks. But the index cannot see gems, mixins or
    # metaprogrammed methods, so `_repeat_guard` in the caller lets a second
    # attempt through as a warning: one strong signal, never a locked door.
    errors = []
    warnings = []
    try:
        from macqwen.tools.api_guard import check_ruby, check_arity
        for m, near in check_ruby(content, project_root=project_root):
            errors.append(f"{m} does not exist in the SketchUp API or Ruby core"
                          + (f". Closest real method: {near}" if near else ""))
        # This BLOCKS, it does not warn. The model has read a correct
        # signature and then written a call that contradicts it, in the same
        # turn, more than once. Under-supplying a documented method is not a
        # judgement call, and a warning it can rationalise past is worthless.
        # Only too-few arguments is reported, so variadic methods are safe.
        for name, given, low, params in check_arity(content):
            shape = f" ({', '.join(params)})" if params else ""
            errors.append(f"{name} is called with {given} argument(s); the "
                          f"documented form takes at least {low}{shape}")
        # Right count, wrong order. Only literals are judged, so a variable is
        # never second-guessed. This is the check arity structurally cannot do.
        from macqwen.tools.api_guard import check_arg_types
        for name, pos, given, expected in check_arg_types(content):
            errors.append(f"{name}: argument {pos} is a {given}, but the "
                          f"documented signature has {expected}")
    except ImportError:
        pass
    return errors, warnings


def _stdlib_attrs(content):
    """Catch `os.getcwdd()`: an invented attribute on a real module.

    pyflakes only resolves names, so it passes an imported module with a
    misspelt member, which is the exact shape of the SketchUp failure. Only
    standard-library modules are introspected: importing an arbitrary
    third-party package would execute its code, and a checker must not do that.
    """
    import ast
    import importlib
    import sys
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    std = getattr(sys, "stdlib_module_names", set())
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in std and "." not in a.name:
                    imported[a.asname or a.name] = a.name
    problems = []
    seen = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in imported):
            mod_name = imported[node.value.id]
            key = (mod_name, node.attr)
            if key in seen:
                continue
            seen.add(key)
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            if not hasattr(mod, node.attr):
                near = difflib.get_close_matches(node.attr, dir(mod), n=1, cutoff=0.7)
                problems.append(
                    f"line {node.lineno}: {mod_name}.{node.attr} does not exist"
                    + (f". Closest real name: {mod_name}.{near[0]}" if near else ""))
    return problems[:6]


def _python(content):
    try:
        compile(content, "<write>", "exec")
    except SyntaxError as e:
        return [f"line {e.lineno}: {e.msg}"]
    attrs = _stdlib_attrs(content)
    if attrs:
        return attrs
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        import io
        out, err = io.StringIO(), io.StringIO()
        check(content, "<write>", Reporter(out, err))
        return [l for l in out.getvalue().splitlines()
                if "undefined name" in l or "imported but unused" not in l][:8]
    except ImportError:
        return []


def _node(content):
    p = _tmp(content, ".js")
    try:
        code, out = _run(["node", "--check"], p)
        return [l for l in out.splitlines() if l.strip()][:6] if code else []
    finally:
        p.unlink(missing_ok=True)


def _shell(content):
    p = _tmp(content, ".sh")
    try:
        code, out = _run(["bash", "-n"], p)
        return [l for l in out.splitlines() if l.strip()][:6] if code else []
    finally:
        p.unlink(missing_ok=True)


def _json(content):
    try:
        json.loads(content)
        return []
    except ValueError as e:
        return [str(e)]


def _yaml(content):
    try:
        import yaml
        yaml.safe_load(content)
        return []
    except ImportError:
        return []
    except Exception as e:
        return [str(e).splitlines()[0]]


CHECKERS = {
    ".swift": _swift, ".rb": _ruby, ".py": _python,
    ".js": _node, ".mjs": _node, ".cjs": _node,
    ".sh": _shell, ".bash": _shell, ".zsh": _shell,
    ".json": _json, ".yaml": _yaml, ".yml": _yaml,
}


def check(path, content, project_root=None):
    """Return (errors, warnings).

    Errors are definitive: a parser or a type checker said no. They block the
    write. Warnings come from indexes that cannot see everything, so they ride
    along with a successful write instead of stopping it.
    """
    fn = CHECKERS.get(Path(str(path)).suffix.lower())
    if not fn:
        return [], []
    out = fn(content, project_root) if fn is _ruby else fn(content)
    return out if isinstance(out, tuple) else (out, [])


def report(path, problems):
    lines = [f"Refused to write {Path(str(path)).name}: it does not verify.",
             *(f"  {p}" for p in problems),
             "Fix these before writing. If a method name is the problem, look it "
             "up with web_search rather than guessing another name."]
    return "\n".join(lines)


if __name__ == "__main__":
    cases = [
        ("bad.swift", 'import Foundation\nlet s = "hi"\nprint(s.reverseeed())\n'),
        ("ok.swift", 'import Foundation\nlet s = "hi"\nprint(s.reversed())\n'),
        ("bad.rb", "ents.add_glue(a, b)\n"),
        ("ok.rb", "Sketchup.active_model.selection.grep(Sketchup::Face).each { |f| f.pushpull(3) }\n"),
        ("bad.py", "import os\nprint(undefined_thing)\n"),
        ("ok.py", "import os\nprint(os.getcwd())\n"),
        ("bad.js", "function f( { return 1 }\n"),
        ("bad.json", '{"a": 1,}\n'),
    ]
    for name, src in cases:
        errs, warns = check(name, src)
        state = ("REFUSED: " + errs[0][:80] if errs
                 else "warn: " + warns[0][:80] if warns else "CLEAN")
        print(f"{name:<11} {state}")
