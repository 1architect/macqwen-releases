#!/usr/bin/env python3
"""Persistent, model-off repository token cache.

The cache lives outside the repository in one SQLite file. The model weights
never load. `watch` uses a lightweight portable scan, then rebuilds only files
whose size or nanosecond modification time changed.
"""

from __future__ import annotations

import argparse
import array
import fnmatch
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from macqwen.api_keys import sanitized_environment

DEFAULT_MODEL = Path(
    "/Users/gioma/.lmstudio/models/gioma/"
    "Qwen3.8-27B-Apple-MLX-V3.1-Compact"
)
DEFAULT_CACHE_ROOT = Path.home() / "Library/Application Support/QwenRepoCache"
MAX_FILE_BYTES = 2_000_000

SKIP_DIRS = {
    ".git", ".build", ".swiftpm", ".idea", ".vscode", ".venv",
    "DerivedData", "node_modules", "Pods", "build", "dist", "venv",
    "__pycache__",
}
KNOWN_TEXT = {
    ".bash", ".c", ".cc", ".conf", ".cpp", ".css", ".csv", ".env",
    ".gitattributes", ".gitignore", ".go", ".gradle", ".h", ".hpp",
    ".html", ".ini", ".java", ".js", ".json", ".jsx", ".kt", ".lock",
    ".m", ".md", ".mm", ".modulemap", ".pbxproj", ".plist", ".podspec",
    ".properties", ".py", ".rb", ".rs", ".rtf", ".sh", ".sql",
    ".strings", ".swift", ".swiftinterface", ".toml", ".ts", ".tsx",
    ".tsv", ".txt", ".xcconfig", ".xml", ".yaml", ".yml", ".zsh",
}
SECRET_NAMES = {
    ".env", ".npmrc", ".pypirc", "credentials", "credentials.json",
    "id_dsa", "id_ed25519", "id_rsa",
}
SECRET_GLOBS = (".env.*", "*.key", "*.p12", "*.pem")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenizer_fingerprint(model: Path) -> str:
    digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = model / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing tokenizer file: {path}")
        digest.update(name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def repo_key(root: Path) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", root.name).strip("-") or "repo"
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return f"{clean}-{suffix}"


def encode_ids(ids: list[int]) -> bytes:
    values = array.array("I", ids)
    if sys.byteorder != "little":
        values.byteswap()
    return zlib.compress(values.tobytes(), level=1)


def decode_ids(blob: bytes) -> list[int]:
    values = array.array("I")
    values.frombytes(zlib.decompress(blob))
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def is_secret(path: Path) -> bool:
    name = path.name.lower()
    return name in SECRET_NAMES or any(fnmatch.fnmatch(name, pat) for pat in SECRET_GLOBS)


def is_likely_text(path: Path) -> bool:
    if path.suffix.lower() in KNOWN_TEXT or path.name in {
        "Dockerfile", "Gemfile", "LICENSE", "Makefile", "NOTICE", "Podfile",
    }:
        return True
    try:
        with path.open("rb") as handle:
            return b"\0" not in handle.read(8000)
    except OSError:
        return False


def git_paths(root: Path) -> list[Path] | None:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env=sanitized_environment(),
    )
    if result.returncode != 0:
        return None
    return [root / os.fsdecode(value) for value in result.stdout.split(b"\0") if value]


def repository_files(
    root: Path,
    max_bytes: int = MAX_FILE_BYTES,
    include_secrets: bool = False,
) -> list[Path]:
    candidates = git_paths(root)
    if candidates is None:
        candidates = []
        for directory, dirs, names in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
            candidates.extend(Path(directory) / name for name in sorted(names))

    files = []
    for path in candidates:
        try:
            relative = path.relative_to(root)
            stat = path.stat()
        except (OSError, ValueError):
            continue
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if not path.is_file() or stat.st_size > max_bytes:
            continue
        if not include_secrets and is_secret(path):
            continue
        if is_likely_text(path):
            files.append(path)
    return sorted(set(files), key=lambda item: item.relative_to(root).as_posix())


def wrapped_text(relative: str, body: str) -> str:
    return f"\n\n// FILE: {relative}\n{body}\n"


@dataclass
class BuildResult:
    files: int
    changed: int
    reused: int
    removed: int
    tokens: int
    seconds: float


class RepoTokenCache:
    def __init__(
        self,
        root: str | Path,
        model: str | Path = DEFAULT_MODEL,
        cache_root: str | Path = DEFAULT_CACHE_ROOT,
    ):
        self.root = Path(root).expanduser().resolve()
        self.model = Path(model).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.fingerprint = tokenizer_fingerprint(self.model)
        self.directory = self.cache_root / "repos" / repo_key(self.root)
        self.db_path = self.directory / f"tokens-{self.fingerprint[:16]}.sqlite3"
        self._tokenizer = None

    def connect(self) -> sqlite3.Connection:
        self.directory.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=30)
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute(
            """CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                token_data BLOB NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        db.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            [
                ("repository", str(self.root)),
                ("model", str(self.model)),
                ("tokenizer_fingerprint", self.fingerprint),
                ("format", "wrapped-file-v1"),
            ],
        )
        return db

    def tokenizer(self):
        if self._tokenizer is None:
            from mlx_lm.utils import load_tokenizer

            self._tokenizer = load_tokenizer(self.model)
        return self._tokenizer

    def build(
        self,
        max_bytes: int = MAX_FILE_BYTES,
        include_secrets: bool = False,
    ) -> BuildResult:
        started = time.perf_counter()
        paths = repository_files(self.root, max_bytes, include_secrets)
        relative_paths = {path.relative_to(self.root).as_posix() for path in paths}
        changed = reused = tokens = 0

        with self.connect() as db:
            existing = {
                row[0]: row[1:]
                for row in db.execute(
                    "SELECT path, size, mtime_ns, content_sha256, token_count FROM files"
                )
            }
            for path in paths:
                relative = path.relative_to(self.root).as_posix()
                try:
                    stat = path.stat()
                except OSError:
                    continue
                old = existing.get(relative)
                if old and old[0] == stat.st_size and old[1] == stat.st_mtime_ns:
                    reused += 1
                    tokens += int(old[3])
                    continue

                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
                content_hash = sha256_bytes(raw)
                if old and old[2] == content_hash:
                    db.execute(
                        "UPDATE files SET size=?, mtime_ns=? WHERE path=?",
                        (stat.st_size, stat.st_mtime_ns, relative),
                    )
                    reused += 1
                    tokens += int(old[3])
                    continue

                body = raw.decode("utf-8", errors="replace")
                ids = self.tokenizer().encode(
                    wrapped_text(relative, body), add_special_tokens=False
                )
                db.execute(
                    """INSERT OR REPLACE INTO files
                    (path, size, mtime_ns, content_sha256, token_count, token_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        relative,
                        stat.st_size,
                        stat.st_mtime_ns,
                        content_hash,
                        len(ids),
                        encode_ids(ids),
                        time.time(),
                    ),
                )
                changed += 1
                tokens += len(ids)

            stale = sorted(set(existing) - relative_paths)
            if stale:
                db.executemany("DELETE FROM files WHERE path=?", [(path,) for path in stale])
            db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_build', ?)",
                (str(time.time()),),
            )

        return BuildResult(
            files=len(paths),
            changed=changed,
            reused=reused,
            removed=len(stale),
            tokens=tokens,
            seconds=time.perf_counter() - started,
        )

    def get(self, path: str | Path) -> list[int] | None:
        path = Path(path).expanduser().resolve()
        try:
            relative = path.relative_to(self.root).as_posix()
            stat = path.stat()
        except (OSError, ValueError):
            return None
        if not self.db_path.is_file():
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT size, mtime_ns, token_data FROM files WHERE path=?", (relative,)
            ).fetchone()
        if not row or row[0] != stat.st_size or row[1] != stat.st_mtime_ns:
            return None
        return decode_ids(row[2])

    def summary(self) -> tuple[int, int, int]:
        if not self.db_path.is_file():
            return 0, 0, 0
        with self.connect() as db:
            files, tokens, compressed = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(token_count), 0), "
                "COALESCE(SUM(LENGTH(token_data)), 0) FROM files"
            ).fetchone()
        return int(files), int(tokens), int(compressed)

    def snapshot(self, max_bytes: int, include_secrets: bool) -> tuple:
        values = []
        for path in repository_files(self.root, max_bytes, include_secrets):
            try:
                stat = path.stat()
                values.append(
                    (path.relative_to(self.root).as_posix(), stat.st_size, stat.st_mtime_ns)
                )
            except OSError:
                pass
        return tuple(values)


def print_result(cache: RepoTokenCache, result: BuildResult) -> None:
    rate = result.tokens / max(result.seconds, 1e-9)
    size = cache.db_path.stat().st_size if cache.db_path.exists() else 0
    print(f"cache:   {cache.db_path}")
    print(f"files:   {result.files} ({result.changed} changed, {result.reused} reused, "
          f"{result.removed} removed)")
    print(f"tokens:  {result.tokens} at {rate:,.0f} tok/s")
    print(f"storage: {size / 1024**2:.2f} MiB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--max-file-mb", type=float, default=2.0)
    parser.add_argument("--include-secrets", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("build", "status", "path"):
        command = sub.add_parser(name)
        command.add_argument("repository")
    watch = sub.add_parser("watch")
    watch.add_argument("repository")
    watch.add_argument("--interval", type=float, default=1.5)
    watch.add_argument("--debounce", type=float, default=0.8)

    args = parser.parse_args()
    cache = RepoTokenCache(args.repository, args.model, args.cache_root)
    max_bytes = max(1, int(args.max_file_mb * 1_000_000))

    if args.command == "path":
        print(cache.db_path)
        return 0
    if args.command == "status":
        files, tokens, compressed = cache.summary()
        print(f"cache:   {cache.db_path}")
        print(f"files:   {files}")
        print(f"tokens:  {tokens}")
        print(f"payload: {compressed / 1024**2:.2f} MiB")
        return 0
    if args.command == "build":
        print_result(cache, cache.build(max_bytes, args.include_secrets))
        return 0

    result = cache.build(max_bytes, args.include_secrets)
    print_result(cache, result)
    print(f"watching {cache.root}; Ctrl+C stops", flush=True)
    previous = cache.snapshot(max_bytes, args.include_secrets)
    pending_since = None
    try:
        while True:
            time.sleep(max(0.2, args.interval))
            current = cache.snapshot(max_bytes, args.include_secrets)
            if current != previous:
                previous = current
                pending_since = time.monotonic()
                continue
            if pending_since is not None and time.monotonic() - pending_since >= args.debounce:
                result = cache.build(max_bytes, args.include_secrets)
                print_result(cache, result)
                pending_since = None
    except KeyboardInterrupt:
        print("stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
