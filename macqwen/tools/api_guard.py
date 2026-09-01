#!/usr/bin/env python3
"""api_guard.py - refuse to write a method that does not exist.

Observed failure: asked for a SketchUp extension, the model invented
`Entities#add_glue` and built its whole design on it, then planned to call
`Entity#delete_me`. Neither exists. The system prompt already told it to verify
unfamiliar methods with web_search. It did not, and a prompt cannot guarantee
that it will.

So check instead of asking. The documentation ships a complete method index, and
the local Ruby interpreter knows its own core. Anything in neither, and not
defined in the file itself, is invented.

    check_ruby(code) -> [(method, suggestion)]

Empty list means every call resolves. The caller blocks the write and hands the
list back to the model, which is a far stronger signal than a prompt bullet.
"""
import difflib
import json
import re
from pathlib import Path

DOCS = Path.home() / ".frankenstein" / "apidocs"

# A receiver we cannot resolve statically, so the check is name-based: a method
# is accepted if any documented class defines it. That under-blocks rather than
# over-blocks, which is the right way round for a guard that can veto a write.
_cache = {}


def _load(name):
    if name not in _cache:
        p = DOCS / f"{name}.json"
        _cache[name] = json.loads(p.read_text()) if p.exists() else None
    return _cache[name]


# Things that look like method calls but are not, or are always defined.
SKIP = {"new", "initialize", "call", "name", "class", "send", "puts", "print",
        "require", "require_relative", "raise", "loop", "lambda", "proc",
        "attr_accessor", "attr_reader", "attr_writer", "include", "extend",
        "module_function", "private", "public", "protected", "define_method"}

CALL_RE = re.compile(r"\.([a-z_][a-zA-Z0-9_]*[?!]?)\s*(?:\(|\s|$)")
DEF_RE = re.compile(r"\bdef\s+(?:self\.)?([a-z_][a-zA-Z0-9_]*[?!=]?)")
STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|#\s*[^\n]*')
# `MultiExtrude.run` is the author's own module, not an API we can check.
CONST_CALL_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\.([a-z_][a-zA-Z0-9_]*[?!]?)")
MODULE_RE = re.compile(r"\b(?:module|class)\s+([A-Z][A-Za-z0-9_]*)")
# Namespaced constants: Sketchup::Length raises NameError, Length is top level.
NAMESPACED_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)::([A-Z][A-Za-z0-9_]*)")
# Legacy globals the modern API dropped, e.g. $sketchup.
GLOBAL_RE = re.compile(r"(\$[a-z_][a-zA-Z0-9_]*)")
LIVE_GLOBALS = {"$stdout", "$stderr", "$stdin", "$0", "$LOAD_PATH", "$:", "$PROGRAM_NAME"}


def check_ruby(code, api="sketchup", project_root=None):
    """Return [(invented_method, closest_real_method_or_None)]."""
    index = _load(api)
    core = _load("ruby_core")
    if not index or not core:
        return []                      # no index, no opinion
    real = set(index["methods"])
    known_classes = set(index.get("classes", []))
    core = set(core)
    body = STRING_RE.sub(" ", code)    # do not read strings or comments
    local = set(DEF_RE.findall(body))
    own = set(MODULE_RE.findall(body))

    # Methods defined anywhere in the project. A loader file calls into the
    # implementation file, and refusing that was making the model inline
    # working code to get past this check.
    if project_root:
        try:
            for f in Path(project_root).rglob("*.rb"):
                if f.stat().st_size < 400_000:
                    other = f.read_text(encoding="utf-8", errors="replace")
                    local |= set(DEF_RE.findall(other))
                    own |= set(MODULE_RE.findall(other))
        except OSError:
            pass

    # Calls on the author's own constants cannot be checked against any index.
    for recv, meth in CONST_CALL_RE.findall(body):
        head = recv.split("::")[0]
        if head in own or (head not in known_classes and recv not in known_classes
                           and head not in ("Sketchup", "UI", "Geom", "Layout")):
            local.add(meth)
    # `x.foo = 1` is an assignment to a writer; accept if `foo=` or `foo` exists
    called = set(CALL_RE.findall(body))
    bad = []
    for m in sorted(called):
        base = m.rstrip("?!")
        if (m in real or base in real or m in core or base in core
                or m in local or base in local or m in SKIP
                or base + "=" in real or base + "?" in real):
            continue
        # Only object when the name is a near miss of a real API method. An
        # invented name is a corruption of one that exists; a name resembling
        # nothing is far more likely to be the author's own helper.
        near = difflib.get_close_matches(base, real, n=1, cutoff=0.72)
        if near:
            bad.append((m, near[0]))

    # Constants. `Sketchup::Length` looks plausible and does not exist: Length
    # is top level. A method index cannot see this, because it is not a call.
    if known_classes:
        tops = {c.split("::")[0] for c in known_classes}
        for ns, name in set(NAMESPACED_RE.findall(body)):
            full = f"{ns}::{name}"
            if full in known_classes or ns in own or ns not in tops:
                continue
            if name in known_classes:          # exists, but at the top level
                bad.append((full, name))
            else:
                near = difflib.get_close_matches(full, known_classes, n=1, cutoff=0.72)
                bad.append((full, near[0] if near else None))

    # Bare constants. MB_ICONWARNING and MB_ICONERROR are Windows API names
    # that SketchUp never defined; using one raises NameError on the first
    # error path, which is the path least likely to be tested. Namespaced
    # constants were already checked above; these have no "::" to spot them by.
    known_consts = set(index.get("constants", []))
    if known_consts:
        prefixes = {c.split("_")[0] for c in known_consts if "_" in c}
        for name in set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", body)):
            if name in known_consts or name in own or name in local:
                continue
            # Only judge names that look like they belong to a documented
            # family, so ordinary Ruby constants are left alone.
            if name.split("_")[0] in prefixes:
                near = difflib.get_close_matches(name, known_consts, n=1, cutoff=0.6)
                bad.append((name, near[0] if near else None))

    # Globals. $sketchup was real in SketchUp 6 and is gone from the modern API.
    for g in set(GLOBAL_RE.findall(body)):
        if g in LIVE_GLOBALS or g.startswith("$" + "_"):
            continue
        if g == "$sketchup":
            bad.append((g, "UI.menu"))
    return bad


def report(bad, api="sketchup"):
    """The message handed back to the model in place of a successful write."""
    lines = [f"Refused: {len(bad)} method(s) in this code do not exist in the "
             f"{api} API or in Ruby core."]
    for m, near in bad:
        if near:
            lines.append(f"  {m} does not exist. Closest real method: {near}")
        else:
            lines.append(f"  {m} does not exist.")
    lines.append("Look up the real method with web_search before writing again. "
                 "Do not guess a replacement name.")
    return "\n".join(lines)


if __name__ == "__main__":
    sample = '''
    module Extruder
      def self.run(height)
        model = Sketchup.active_model
        ents  = model.active_entities
        model.selection.grep(Sketchup::Face).each do |face|
          n = face.normal
          top = face.vertices.map { |v| v.position + n }
          e = ents.add_glue(top[0], top[1])
          face.delete_me
        end
      end
    end
    '''
    bad = check_ruby(sample)
    print(report(bad) if bad else "all methods resolve")
    good = '''
    model.selection.grep(Sketchup::Face).each { |f| f.pushpull(height) }
    Sketchup.active_model.active_entities.add_face(pts)
    '''
    b2 = check_ruby(good)
    print()
    print(report(b2) if b2 else "all methods resolve (correct version)")


# --------------------------------------------------------------- arity check

# A call with too FEW arguments is a bug. A call with too many is usually a
# variadic method whose documented examples happen to show a shorter form, so
# only under-supply is reported. That asymmetry is deliberate: `add_face` takes
# any number of points, and flagging it would be worse than missing a bug.
RUBY_CALL_RE = re.compile(
    r"(?:^|[\s.(=])([A-Za-z_][\w]*)\s*\(([^()]*)\)")
QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def check_arity(code, library="sketchup", client=None):
    """Return [(method, given, documented_minimum)] for under-supplied calls."""
    try:
        from macqwen.tools.context7 import Context7
    except ImportError:
        return []
    c = client or Context7()
    # STRING_RE blanks literals to whitespace, which deletes arguments:
    # `f(["a"], "b")` became `f([ ],  )` and counted as one argument. Keep a
    # placeholder token instead so the comma count survives.
    body = re.sub(r"#[^\n]*", " ", code)
    body = QUOTED_RE.sub("S", body)
    local = set(DEF_RE.findall(body))
    seen, bad = set(), []
    for name, raw in RUBY_CALL_RE.findall(body):
        if name in local or name in SKIP or name in seen:
            continue
        seen.add(name)
        args = QUOTED_RE.sub("S", raw)
        given = len([a for a in args.split(",") if a.strip()])
        try:
            sigs = c.signatures(library, name)
        except Exception:
            continue
        entry = sigs.get(name)
        if not entry or not entry["arities"]:
            continue
        low = min(entry["arities"])
        if given < low:
            # Parameter names scraped from example call sites are often
            # literals rather than names. Only show them when they read as
            # identifiers, otherwise the count alone is the useful part.
            names = entry["params"][:low]
            if not all(re.fullmatch(r"[A-Za-z_]\w*", n or "") for n in names):
                names = []
            bad.append((name, given, low, names))
    return bad


def arity_report(bad):
    lines = [f"{len(bad)} call(s) have too few arguments:"]
    for name, given, low, params in bad:
        shape = f": ({', '.join(params)})" if params else ""
        lines.append(f"  {name} was given {given} argument(s); the documented "
                     f"form takes at least {low}{shape}")
    lines.append("Call api_docs for the exact signature. Do not guess an overload.")
    return "\n".join(lines)


# ----------------------------------------------------- documented arg types

# Context7 flattens some methods to `*args` where the vendor page carries the
# real positional types. UI.inputbox is exactly that case, and it is the call
# this model has now got wrong three times, so fall back to the vendor page.
VENDOR = "https://ruby.sketchup.com/"
TYPED_SIG_RE = re.compile(r"(\w+)\s*\(([^)]{2,120})\)\s*&#x21d2;|(\w+)\s*\(([^)]{2,120})\)\s*⇒")
_types_cache = {}


def _vendor_text(page):
    import urllib.request, html as _html
    cache = DOCS / "vendor"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / (page.replace("/", "_") + ".txt")
    if f.exists():
        return f.read_text()
    req = urllib.request.Request(VENDOR + page,
                                 headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
    t = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", t)))
    f.write_text(t)
    return t


def vendor_param_types(method, api="sketchup"):
    """All documented forms: [[(name, Type), ...], ...], richest first.

    A method is usually listed several times: `inputbox(*args)` for the
    wrapper, `(prompts, defaults, title)` and `(prompts, defaults, list,
    title)` for the real ones. Returning a single form meant a call matching a
    different overload was skipped entirely.
    """
    if method in _types_cache:
        return _types_cache[method]
    index = _load(api) or {}
    owners = [c for c, ms in index.get("by_class", {}).items() if method in ms]
    forms = []
    for owner in owners[:2]:
        try:
            text = _vendor_text(owner.replace("::", "/") + ".html")
        except Exception:
            continue
        for m in re.finditer(rf"\b{re.escape(method)}\s*\(([^)]{{2,140}})\)", text):
            args = [a.strip() for a in m.group(1).split(",") if a.strip()]
            if not args or any(a.startswith("*") for a in args):
                continue
            names = [a.split("=")[0].strip().lstrip("&") for a in args]
            after = text[m.end(): m.end() + 1800]
            pairs = []
            for n in names:
                t = re.search(rf"\b{re.escape(n)}\s*\(\s*([A-Za-z][\w<>, ]*?)\s*\)",
                              after)
                pairs.append((n, re.sub(r"\s+", "", t.group(1)) if t else ""))
            if any(t for _, t in pairs) and pairs not in forms:
                forms.append(pairs)
        if forms:
            break
    forms.sort(key=len, reverse=True)
    _types_cache[method] = forms
    return forms




# Ruby literals whose type is obvious at the call site. Anything else (a
# variable, a method call) is unknown and never judged.
def _literal_type(arg):
    a = arg.strip()
    if not a:
        return None
    if a.startswith("["):
        return "Array"
    if a.startswith(("'", '"')):
        return "String"
    if a in ("true", "false"):
        return "Boolean"
    if re.fullmatch(r"-?\d+", a):
        return "Integer"
    if re.fullmatch(r"-?\d*\.\d+", a):
        return "Float"
    return None


def _compatible(literal, documented):
    d = (documented or "").replace(" ", "")
    if not d or not literal:
        return True                       # unknown on either side: no opinion
    if literal in d:
        return True
    if literal == "Integer" and ("Numeric" in d or "Float" in d or "Length" in d):
        return True
    if literal == "Float" and ("Numeric" in d or "Length" in d):
        return True
    if literal == "String" and "Length" in d:
        return True                       # "2,5m".to_l style
    return False


def check_arg_types(code, api="sketchup"):
    """[(method, position, given, expected)] where a literal contradicts the docs.

    Arity cannot see an argument ORDER that has the right count. This can:
    UI.inputbox is documented (Array, Array, String) and the model wrote
    (Array, String, Array) three separate times.
    """
    # Blank out the INSIDE of string literals but keep the quotes: a prompt
    # like "altura (ex.: 30 cm)" carries a parenthesis that breaks call
    # matching, while the quotes are what identify the argument as a String.
    body = re.sub(r"#[^\n]*", " ", code)
    body = QUOTED_RE.sub('"S"', body)
    local = set(DEF_RE.findall(body))
    bad, seen = [], set()
    for name, raw in RUBY_CALL_RE.findall(body):
        if name in local or name in SKIP or name in seen:
            continue
        seen.add(name)
        forms = vendor_param_types(name, api)
        args = [a for a in re.split(r",(?![^\[]*\])", raw) if a.strip()]
        # Judge against the overload with this argument count, if one exists.
        types = next((f for f in forms if len(f) == len(args)), None)
        if not types:
            continue
        for i, a in enumerate(args):
            lit, (pname, doc) = _literal_type(a), types[i]
            if lit and not _compatible(lit, doc):
                bad.append((name, i + 1, lit, f"{pname}: {doc}"))
    return bad
