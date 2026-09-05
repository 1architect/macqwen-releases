"""The nine repository tools, shared by every model.

Filesystem work, no model in it. Every path is resolved against the
workspace root and refused if it escapes, so a tool call cannot reach
outside the repository it was pointed at.

Extracted from frankenstein_engine.py, where it sat beside the 27B
generation loop and could not be reached by any other model.
"""
from __future__ import annotations

import fnmatch
import os
import subprocess
import tempfile
from pathlib import Path

from macqwen.api_keys import sanitized_environment
from macqwen.tools import PARAM_TYPES

SKIP = {".git", ".build", "DerivedData", ".swiftpm", "node_modules", "Pods", ".idea", ".vscode"}

TEXT = {".swift", ".m", ".mm", ".h", ".hpp", ".c", ".cc", ".cpp", ".plist", ".json", ".yaml",
        ".yml", ".toml", ".md", ".txt", ".entitlements", ".xcconfig", ".pbxproj", ".sh", ".py"}

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
                    target = p.resolve()
                    target.relative_to(self.root)
                    if target.stat().st_size > 2_000_000:
                        continue
                    lines = target.read_text(errors="replace").splitlines()
                except (OSError, RuntimeError, ValueError):
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
        environment = sanitized_environment()
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
