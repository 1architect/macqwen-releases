#!/usr/bin/env python3
"""context7.py - exact API signatures, so the model never has to remember one.

Every failure this agent had on library code was the same shape: it knew the
method existed and invented the arguments. `UI.inputbox` was wrong twice in a
row; `Face#pushpull` took four runs to get right; `length_unit_conversion_factor`
does not exist at all.

Context7 indexes documentation for thousands of libraries and returns it in a
form built for a model to read: the signature, the parameter types, the
defaults, the return type, and a runnable example. Reading beats recalling, so
this replaces recall wherever it can.

    search(query)          -> candidate library ids
    docs(library, topic)   -> documentation text for that topic
    signature(library, m)  -> parsed argument list, for checking a write

Answers are cached on disk. The service allows 200 requests per window and a
coding session asks the same questions repeatedly.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://context7.com/api/v1"
CACHE = Path.home() / ".frankenstein" / "context7-cache"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TTL = 14 * 24 * 3600           # documentation moves slowly

# Names the model is likely to use, mapped to the library id that actually
# answers. Saves a search round trip on the ones this project hits daily.
ALIASES = {
    "sketchup": "/websites/ruby_sketchup",
    "sketchup ruby": "/websites/ruby_sketchup",
    "ruby": "/ruby/ruby",
    "mlx": "/ml-explore/mlx",
    "swiftui": "/websites/developer_apple_swiftui",
    "numpy": "/numpy/numpy",
}


class Context7:
    def __init__(self, timeout=20):
        self.timeout = timeout
        self.calls = 0
        CACHE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- transport

    def _cached(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()[:32]
        p = CACHE / f"{key}.txt"
        if p.exists() and time.time() - p.stat().st_mtime < TTL:
            return p.read_text(), True
        headers = {"User-Agent": UA, "Accept": "*/*"}
        # Optional. The free tier answers without one; a key only raises the
        # request ceiling, and cached answers cost nothing either way.
        key = os.environ.get("CONTEXT7_API_KEY", "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=self.timeout) as r:
            body = r.read().decode("utf-8", "replace")
        self.calls += 1
        p.write_text(body)
        return body, False

    # --------------------------------------------------------------- lookups

    def search(self, query, limit=5):
        url = f"{BASE}/search?query={urllib.parse.quote(query[:120])}"
        try:
            body, _ = self._cached(url)
            hits = json.loads(body).get("results", [])
        except (HTTPError, URLError, ValueError):
            return []
        return [{"id": h.get("id"), "title": h.get("title"),
                 "description": (h.get("description") or "")[:160]}
                for h in hits[:limit]]

    def resolve(self, library):
        """A library name the model typed -> the id Context7 indexes it under."""
        low = library.strip().lower()
        if low in ALIASES:
            return ALIASES[low]
        if library.startswith("/"):
            return library
        hits = self.search(library, limit=1)
        return hits[0]["id"] if hits else None

    def docs(self, library, topic=None, tokens=1500):
        lib = self.resolve(library)
        if not lib:
            return {"error": f"No documentation indexed for {library!r}."}
        url = f"{BASE}{lib}?tokens={int(tokens)}&type=txt"
        if topic:
            url += f"&topic={urllib.parse.quote(topic[:80])}"
        try:
            body, hit = self._cached(url)
        except (HTTPError, URLError) as e:
            return {"error": f"Context7 unavailable: {getattr(e, 'reason', e)}"}
        if not body.strip():
            return {"error": f"Nothing indexed for {topic!r} in {lib}."}
        return {
            "library": lib,
            "topic": topic,
            "cached": hit,
            "documentation": body[: tokens * 6],
            "instruction": (
                "This is the real signature from the library's own reference. "
                "Use it exactly as written. Do not adapt it from memory, and do "
                "not assume an overload that is not shown here."
            ),
        }

    # ------------------------------------------------------------- signatures

    # Headings give the canonical form; examples give the real call shapes.
    # Both matter: the canonical heading for UI.inputbox is `inputbox(*args)`,
    # which says nothing, while the examples show the 3 and 4 argument forms
    # that actually exist. Reading only the heading is how the argument order
    # got guessed wrong twice.
    CALL_RE = re.compile(r"\b([A-Za-z_][\w]*)\s*\(([^()]*)\)")
    STR_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

    @classmethod
    def _split_args(cls, raw):
        raw = cls.STR_RE.sub("S", raw)          # a literal is one argument
        return [a.strip() for a in raw.split(",") if a.strip()]

    def signatures(self, library, topic, tokens=1500):
        """{method: {"params": [...], "arities": sorted set}} from the reference.

        `arities` is what a write check needs: passing two arguments to a call
        documented only at three and four is a bug, whatever the names are.
        """
        d = self.docs(library, topic, tokens)
        if "error" in d:
            return {}
        out = {}
        for name, raw in self.CALL_RE.findall(d["documentation"]):
            if name in ("if", "for", "while", "def", "return", "puts", "print"):
                continue
            args = self._split_args(raw)
            if any(a.startswith("*") for a in args):
                out.setdefault(name, {"params": [], "arities": set()})
                continue
            entry = out.setdefault(name, {"params": [], "arities": set()})
            entry["arities"].add(len(args))
            required = [a for a in args if "=" not in a]
            if len(args) > len(entry["params"]):
                entry["params"] = [a.split("=")[0].strip().lstrip("*&") for a in args]
            entry["arities"].add(len(required))
        return {k: {"params": v["params"], "arities": sorted(v["arities"])}
                for k, v in out.items() if v["arities"]}


# Words that name a library we can look up, checked against the user's own
# request. The model should not have to decide to fetch documentation: it has
# been wrong about that decision all day, and the lookup is cheap and cached.
TRIGGERS = {
    "sketchup": "sketchup", "skp": "sketchup", "ruby": "ruby",
    "swiftui": "swiftui", "swift": "swiftui", "appkit": "swiftui",
    "numpy": "numpy", "mlx": "mlx", "react": "react",
}
# Terms worth asking the reference about, once a library is identified.
TOPIC_STOP = {"crie", "uma", "para", "que", "ao", "mesmo", "tempo", "ate", "até",
              "definida", "pelo", "usuario", "usuário", "extensao", "extensão",
              "create", "make", "write", "build", "extension", "plugin", "the",
              "and", "for", "with", "user", "defined", "several", "multiple",
              "same", "time", "height", "add", "please", "a", "an", "of", "to"}


# What the project is built out of beats what the request happens to mention.
# A file tree is unambiguous where "build me an app" is not.
WORKSPACE_MARKERS = [
    ("package.json", "react"), ("tsconfig.json", "typescript"),
    ("Cargo.toml", "rust"), ("go.mod", "go"), ("Gemfile", "ruby"),
    ("requirements.txt", "python"), ("pyproject.toml", "python"),
    ("Package.swift", "swiftui"), ("CMakeLists.txt", "cmake"),
]
EXT_MARKERS = {".rb": "ruby", ".swift": "swiftui", ".py": "python",
               ".tsx": "react", ".ts": "typescript", ".rs": "rust",
               ".go": "go"}


def from_workspace(root, limit=1):
    """Libraries implied by what is actually in the project."""
    try:
        root = Path(root)
        names = {p.name for p in root.iterdir()} if root.is_dir() else set()
    except OSError:
        return []
    hits = [lib for marker, lib in WORKSPACE_MARKERS if marker in names]
    if not hits:
        counts = {}
        try:
            for p in list(root.rglob("*"))[:2000]:
                lib = EXT_MARKERS.get(p.suffix.lower())
                if lib:
                    counts[lib] = counts.get(lib, 0) + 1
        except OSError:
            pass
        hits = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    return hits[:limit]


# Ordinary words that also happen to be the names of tools. Matching them is
# how "just build me an app" fetched a manual for a tool called `just`, and
# "refactor this function" fetched `refactor-mcp`.
COMMON = set("""
just make build create write add get set run open close start stop show list
find fix change update delete remove refactor improve clean test check help
need want like give take use using used new old good best fast slow simple
this that these those what which where when how why some any all more most
can could would should will shall may might must have has had does did done
about after before between into through during without within along across
app apps application code file files folder project script program tool
crie criar fazer faça arquivo pasta programa aplicativo teste mudar
function method class module package library api server client daemon
endpoint database query table array string number object value data
load save parse render import export config setting option result
""".split())


def discover(text, client=None, limit=1):
    """Ask Context7 whether the request names a library it indexes.

    Searching word by word returned whichever word came first, so "load the csv
    with pandas" fetched `load-esm`. Search the whole phrase instead and let
    the index rank it, then accept the top hit only if its name actually
    appears in the request as a word. Ranking picks the candidate; the word
    check stops it inventing a connection.
    """
    low = text.lower()
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_.+-]{2,}", low))
    words -= COMMON | TOPIC_STOP | {"windows", "linux", "macos", "desktop",
                                    "mobile", "web", "android", "ios"}
    if not words:
        return []
    c = client or Context7()
    for hit in c.search(text, limit=8):
        title = (hit.get("title") or "").lower()
        first = re.split(r"[\s/._-]+", title)[0] if title else ""
        if first and first in words:
            return [hit["id"]][:limit]
    # Phrase ranking can bury an exact name: "create an electron desktop app"
    # returned eight hits and none of them was Electron. Fall back to asking
    # about each distinctive word on its own.
    for w in sorted(words, key=len, reverse=True)[:3]:
        for hit in c.search(w, limit=3):
            title = (hit.get("title") or "").lower()
            first = re.split(r"[\s/._-]+", title)[0] if title else ""
            if first == w:
                return [hit["id"]][:limit]
    return []


def detect(text, limit=2, workspace=None, client=None):
    """(library, topic) pairs worth fetching before the model writes anything."""
    low = text.lower()
    libs = []
    for word, lib in TRIGGERS.items():
        if re.search(rf"\b{re.escape(word)}\b", low) and lib not in libs:
            libs.append(lib)
    if not libs:
        libs = discover(text, client=client)
    if not libs and workspace:
        libs = from_workspace(workspace)
    if not libs:
        return []
    terms = [w for w in re.findall(r"[a-zA-Zà-ú_]{3,}", low)
             if w not in TOPIC_STOP and w not in TRIGGERS]
    topic = " ".join(terms[:6]) or None
    return [(lib, topic) for lib in libs[:limit]]


# Facts a signature index cannot carry. These are beliefs about the platform
# that get applied in arithmetic, and they have been wrong in three runs
# running: "mm is SketchUp's internal unit" (it is inches), and a conversion
# constant of 1/39.37 where 1/25.4 belonged.
FACTS = {
    "sketchup": (
        "SketchUp platform facts, verified. Trust these over your priors:\n"
        "- The internal unit is INCHES, not millimetres and not metres. A bare "
        "number passed to pushpull or any Length argument is inches.\n"
        "- Do NOT hand-roll unit conversion. Use String#to_l, which parses "
        "\"500\", \"2,5m\", \"10'\" in the model's own units and returns "
        "internal inches: distancia = texto.to_l\n"
        "- Numeric#mm, #cm, #m, #feet convert the other way: 500.mm is a "
        "Length in internal units.\n"
        "- String#to_f does NOT return NaN on bad input, it returns 0.0. "
        "Checking .nan? never fires; check the string before converting.\n"
        "- vertex.position is in the LOCAL coordinates of its container. For a "
        "face inside a group or component it is not model space.\n"
    ),
}


def prefetch(text, client=None, tokens=1200, workspace=None):
    """Documentation block to place in front of the model, or ''."""
    c = client or Context7()
    pairs = detect(text, workspace=workspace, client=c)
    if not pairs:
        return ""
    parts = []
    for lib, topic in pairs:
        d = c.docs(lib, topic, tokens=tokens)
        if "error" in d:
            continue
        parts.append(f"### {d['library']}  (topic: {topic})\n"
                     + d["documentation"].strip())
    facts = ""
    for lib, _ in pairs:
        key = lib.lower().strip("/").split("/")[-1]
        for name, block in FACTS.items():
            if name in key or name in lib.lower():
                facts = block + "\n"
                break
    if not parts and not facts:
        return ""
    return (facts + "REFERENCE, fetched for you from the library's own documentation. "
            "These are the real signatures. Use them exactly as written. Do not "
            "adapt them from memory and do not assume an overload that is not "
            "shown. If what you need is missing, call api_docs for it.\n\n"
            + "\n\n".join(parts))


# A documented parameter list with types, positionally. This is what catches
# an argument order that has the right COUNT: UI.inputbox(prompts, defaults,
# title) is (Array, Array, String), and passing (Array, String, Array) is
# wrong in a way arity can never see.
PARAM_RE = re.compile(
    r"\*\s*\*\*(\w+)\*\*\s*\(([^)]+?)(?:,\s*optional)?\)")


def param_types(client, library, topic):
    """{method: [(name, Type), ...]} for the documented signature."""
    d = client.docs(library, topic, tokens=2000)
    if "error" in d:
        return {}
    text = d["documentation"]
    out = {}
    # A heading may or may not carry its argument list. `## pushpull(distance,
    # copy)` does; `## UI.inputbox` does not, and its parameters live only in
    # the block below. Take the order from the parentheses when they are there
    # and from the parameter block otherwise, since the block lists them in
    # signature order.
    for m in re.finditer(r"^#{1,3}\s*([\w.:#]+)\s*(?:\(([^()]*)\))?\s*$",
                         text, re.M):
        method = m.group(1).split("#")[-1].split(".")[-1]
        block = text[m.end(): m.end() + 1600]
        typed = PARAM_RE.findall(block)
        if not typed:
            continue
        if m.group(2) is not None:
            order = [a.split("=")[0].strip().lstrip("*&")
                     for a in m.group(2).split(",") if a.strip()]
            if not order or order == ["*args"]:
                continue
            types = {n: t.strip() for n, t in typed}
            pairs = [(n, types.get(n, "")) for n in order]
        else:
            pairs = [(n, t.strip()) for n, t in typed]
        if pairs and method not in out:
            out[method] = pairs
    return out


if __name__ == "__main__":
    import sys
    c = Context7()
    lib = sys.argv[1] if len(sys.argv) > 1 else "sketchup"
    topic = sys.argv[2] if len(sys.argv) > 2 else "pushpull"
    d = c.docs(lib, topic, tokens=600)
    print(d.get("documentation", d.get("error"))[:600])
    print("\nparsed signatures:", c.signatures(lib, topic))
    print(f"network calls this run: {c.calls}")
