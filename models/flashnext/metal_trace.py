#!/usr/bin/env python3
"""Attribute GPU time in a Metal System Trace to named kernels.

The earlier trace work reported one number: the union of GPU spans. That
proved the IOKit counter undercounts by about 3.2x. It did not say what the
GPU runs. This tool exports `metal-gpu-intervals` and groups it, so the 149
ms/token of GPU busy time gets names.

Three export details break a naive parser and each one produced a wrong first
result before:

  1. Rows are positional. A row's children line up with the schema's columns
     in order, and `<sentinel/>` holds an empty column.
  2. A repeated value is written once with `id=` and later as `ref=`. An
     unresolved reference reads as empty, which silently drops channels and
     latencies.
  3. The trace records every process and the intervals nest. Summing raw
     durations across processes double counts. Filter to one process and take
     the union of the spans.

Record a trace with:

    export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
    xcrun xctrace record --template "Metal System Trace" \\
      --output run.trace --time-limit 90s --target-stdout run.log \\
      --launch -- PYTHON BENCH ARGS
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

XPATH = "/trace-toc/run[@number='{run}']/data/table[@schema='{schema}']"


def export(trace: str, schema: str, run: int = 1) -> bytes:
    """Return one table of the trace as XML."""
    env = dict(os.environ)
    env.setdefault("DEVELOPER_DIR", "/Applications/Xcode-beta.app/Contents/Developer")
    out = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace,
         "--xpath", XPATH.format(run=run, schema=schema)],
        capture_output=True, env=env,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode()[-2000:])
    return out.stdout


def rows(xml: bytes):
    """Yield each row as a list of column values, with references resolved.

    A value is `(text, fmt)`. Nested elements keep their own children, so a
    process cell carries its pid as a child; the caller reads `fmt` instead.
    """
    root = ET.fromstring(xml)
    table = root.find(".//schema/..")
    schema = table.find("schema")
    names = [c.find("mnemonic").text for c in schema.findall("col")]
    seen: dict[str, tuple[str, str]] = {}

    def value(node):
        ref = node.get("ref")
        if ref is not None:
            return seen.get(ref, ("", ""))
        cell = (node.text or "", node.get("fmt") or "")
        ident = node.get("id")
        if ident is not None:
            seen[ident] = cell
        return cell

    for row in table.findall("row"):
        cells = []
        for node in row:
            if node.tag == "sentinel":
                cells.append(("", ""))
            else:
                # Register every id in the subtree, not only the top node.
                for child in node.iter():
                    if child.get("id") is not None and child is not node:
                        value(child)
                cells.append(value(node))
        yield names, cells


def union(spans) -> int:
    """Total covered time of possibly overlapping (start, end) pairs."""
    total = 0
    end = -1
    for begin, stop in sorted(spans):
        if begin > end:
            total += stop - begin
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def analyse(trace: str, pid: int | None, tokens: int, top: int, run: int,
            last_ms: float = 0.0):
    xml = export(trace, "metal-gpu-intervals", run)
    groups: dict[tuple[str, str], list] = defaultdict(list)
    pids = defaultdict(int)
    every = []
    every_full = []
    for names, cells in rows(xml):
        cell = dict(zip(names, cells))
        who = cell["process"][1]
        found = re.search(r"\((\d+)\)", who)
        this = int(found.group(1)) if found else -1
        pids[who] += 1
        if pid is not None and this != pid:
            continue
        begin = int(cell["start"][0] or 0)
        length = int(cell["duration"][0] or 0)
        latency = int(cell["start-latency"][0] or 0)
        channel = cell["channel-name"][1] or cell["channel-name"][0]
        label = cell["event-label"][1] or cell["channel-subtitle"][1] or "(unlabelled)"
        depth = cell["event-depth"][0] or "0"
        groups[(channel, label)].append((begin, begin + length, length, latency, depth))
        every.append((begin, begin + length))
        every_full.append((begin, begin + length, length, latency))

    if pid is None:
        print("processes in the trace:")
        for who, count in sorted(pids.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {count:8d}  {who}")
        print("\nRe-run with --pid to attribute one process.")
        return 0

    if last_ms and every:
        # The trace also covers load, warmup and prefill. Decode is the last
        # thing the benchmark does, so keep only the final window. Its width
        # comes from the benchmark's own ms/token times its token count.
        cut = max(e for _, e in every) - int(last_ms * 1e6)
        keep = [i for i, span in enumerate(every) if span[0] >= cut]
        chosen = set(keep)
        every = [every[i] for i in keep]
        every_full = [every_full[i] for i in keep]
        for key in list(groups):
            groups[key] = [g for g in groups[key] if g[0] >= cut]
            if not groups[key]:
                del groups[key]

    covered = union(every)
    span = (max(e for _, e in every) - min(b for b, _ in every)) if every else 0
    print(f"trace          {trace}")
    print(f"pid            {pid}")
    print(f"intervals      {len(every)}")
    print(f"wall span      {span / 1e6:.1f} ms")
    print(f"gpu busy union {covered / 1e6:.1f} ms  ({covered / span * 100:.1f}% of span)")
    total_lat = sum(i[3] for i in every_full) / len(every_full) if every_full else 0
    print(f"mean interval  {sum(i[2] for i in every_full) / len(every_full) / 1000:.1f} us")
    print(f"mean cpu->gpu  {total_lat / 1000:.1f} us")
    if tokens:
        print(f"tokens         {tokens}")
        print(f"gpu busy/token {covered / 1e6 / tokens:.2f} ms")
        print(f"intervals/token {len(every) / tokens:.1f}")

    # Intervals nest: a command buffer encloses its encoders, which enclose
    # the work. The union across every depth is therefore the outermost
    # envelope, and that includes time a buffer sits on the GPU timeline
    # rather than executing. Reporting per depth separates the two. Both
    # sessions' GPU figures were taken across all depths.
    depths = sorted({int(i[4]) for v in groups.values() for i in v})
    print("\nby nesting depth")
    print(f"{'depth':>6} {'count':>9} {'/token':>9} {'union ms':>10} {'/token ms':>11}")
    for level in depths:
        picked = [i for v in groups.values() for i in v if int(i[4]) == level]
        cover = union([(b, e) for b, e, *_ in picked])
        per = f"{cover / 1e6 / tokens:11.2f}" if tokens else " " * 11
        share = f"{len(picked) / tokens:9.1f}" if tokens else " " * 9
        print(f"{level:>6} {len(picked):>9} {share} {cover / 1e6:>10.1f} {per}")

    buckets = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0),
               (1.0, 2.0), (2.0, 5.0), (5.0, 1e9)]
    print("\ninterval duration distribution")
    print(f"{'range ms':>14} {'count':>8} {'/token':>8} {'union ms':>10} {'/token ms':>10} {'lat us':>8}")
    for low, high in buckets:
        picked = [i for i in every_full if low <= i[2] / 1e6 < high]
        if not picked:
            continue
        cover = union([(b, e) for b, e, *_ in picked])
        lat = sum(i[3] for i in picked) / len(picked)
        per_n = f"{len(picked) / tokens:8.1f}" if tokens else " " * 8
        per_ms = f"{cover / 1e6 / tokens:10.2f}" if tokens else " " * 10
        print(f"{low:6.2f}-{high if high < 1e8 else 0:6.2f} {len(picked):>8} {per_n} "
              f"{cover / 1e6:>10.1f} {per_ms} {lat / 1000:>8.1f}")

    print("\nby channel")
    for channel in sorted({k[0] for k in groups}):
        items = [i for k, v in groups.items() if k[0] == channel for i in v]
        cover = union([(b, e) for b, e, *_ in items])
        per = f"{cover / 1e6 / tokens:8.2f}" if tokens else " " * 8
        print(f"  {channel:<12} {len(items):>7} intervals  {cover / 1e6:>9.1f} ms  {per} ms/token")

    print("\nby channel and label, ranked by covered GPU time")
    header = f"{'channel':<10} {'label':<42} {'count':>7} {'union ms':>10} {'/token':>8} {'mean us':>9} {'lat us':>8}"
    print(header)
    print("-" * len(header))
    ranked = sorted(
        groups.items(), key=lambda kv: -union([(b, e) for b, e, *_ in kv[1]])
    )
    for (channel, label), items in ranked[:top]:
        cover = union([(b, e) for b, e, *_ in items])
        mean = sum(i[2] for i in items) / len(items)
        lat = sum(i[3] for i in items) / len(items)
        per = f"{cover / 1e6 / tokens:8.2f}" if tokens else " " * 8
        print(f"{channel:<10} {label[:42]:<42} {len(items):>7} "
              f"{cover / 1e6:>10.1f} {per} {mean / 1000:>9.1f} {lat / 1000:>8.1f}")
    if len(ranked) > top:
        print(f"... {len(ranked) - top} more groups")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--pid", type=int, default=None,
                        help="process to attribute; omit to list the processes")
    parser.add_argument("--tokens", type=int, default=0,
                        help="generated tokens, to report per-token costs")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--last-ms", type=float, default=0.0,
                        help="keep only the final window, in ms; use "
                             "ms/token times tokens to isolate decode")
    args = parser.parse_args()
    return analyse(args.trace, args.pid, args.tokens, args.top, args.run,
                   args.last_ms)


if __name__ == "__main__":
    sys.exit(main())
