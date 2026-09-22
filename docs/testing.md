# Testing

This guide describes where tests live, how to run them, and where results go.
It applies to every model in `models/`. The
[measurement standard](measurement-standard.md) adds the rules for live model
evidence.

## Layout

Each model keeps its runtime code in `models/<model>/` and all its tests in
`models/<model>/tests/`:

```text
models/<model>/
  *.py                 runtime code only
  tests/
    __init__.py        optional hook: canonical_environment(context)
    unit/test_*.py     checkpoint-free unit tests
    bench/*.py         harness scripts that load a checkpoint, and their helpers
    cases/case_*.py    test-terminal cards
    inputs/            operator-supplied inputs, such as a prompt file
  tools/               offline scripts that are neither runtime nor tests
macqwen/
  tests/test_*.py      unit tests for the shared code
  testsuite/           the test terminal
  results.py           the results folder API
results/<model>/       every run's output
```

A module stays in the runtime folder only if runtime code or a backend imports
it. `macqwen/tests/test_results_policy.py` fails when a test, benchmark or case
file sits outside this layout.

## The three kinds of test

**Unit tests** (`tests/unit/`) run without a checkpoint and finish in seconds.
CI runs them on every push. Name files `test_*.py`.

**Benchmarks** (`tests/bench/`) load a real checkpoint and measure it. Run them
as modules from the repository root:

```bash
.venv/bin/python -m models.flashnext.tests.bench.bench_production --compare none
```

One benchmark can serve several cases. Helpers that only benchmarks use, such
as `gpustat` or `metal_trace`, live here too.

**Cases** (`tests/cases/`) are the cards the test terminal lists. One file
holds one runnable test. A case describes the test (purpose, reason, controls,
metrics, source) and builds the command that runs a benchmark. Case files
import only from `macqwen.testsuite.api`.

## Run tests

Unit tests, per tree:

```bash
.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'
.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py'
```

Replace `flashnext` with `bonsai2`, `k2_horizon` or `qwen27b` for the other
models. Always pass `-t .`, so imports resolve from the repository root.

Live tests, through the one test terminal:

```bash
./tests/run.sh
./tests/run.sh --model flashnext --checkpoint PATH
```

The terminal discovers every `models/*/tests/cases/case_*.py` file. A new model
needs no terminal change: add case files and it appears.

## Results

Every run writes to its own folder:

```text
results/<model>/<YYYYMMDD-HHMMSS>-<test-id>/
  record.jsonl    canonical measurement record
  output.log      the complete stdout and stderr of the run
  ...             every file the benchmark writes
```

This is not optional. The terminal creates the folder and passes it to the
benchmark in `MACQWEN_RESULTS_DIR`. A benchmark started by hand creates its own
folder, named `<stamp>-<script>-manual`. `macqwen.results.output_path` refuses a
destination outside `results/`, and the policy test rejects fixed output paths,
`/tmp`, and the retired `docs/<model>/measurements/` and `tests/results/`
folders.

Caches that later runs read back as state are not results. Pin profiles, slab
packs, sessions and compiled native libraries keep their locations under
`~/.cache/flashnext/`.

Large binary artifacts (`*.gputrace`, `*.trace`, `*.npz`, `*.npy`, `*.bin`,
`*.pstats`) stay local. Git keeps the records, logs and JSON summaries.

## Write a benchmark

Send every output file through `output_path`:

```python
from macqwen.results import output_path

parser.add_argument("--json")          # no default path
args = parser.parse_args()
args.json = output_path("flashnext", "bench_example", "summary.json", args.json)
```

Without `--json` the file goes in this run's folder. With `--json`, the path
must be inside `results/`.

## Write a case

```python
from macqwen.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, record_path):
    return [
        str(context.python), "-m", bench_module("flashnext", "bench_example"),
        "--tokens", str(context.tokens),
        "--json", str(context.output("summary.json")),
    ]


TEST = TestSpec(
    id="example", title="Example comparison", category="performance",
    explanation="What the test runs.", why="Why we need the result.",
    script=command, metrics=COMMON_METRICS,
    controls={"decode": "greedy", "digest": "required"},
    source="models/flashnext/tests/bench/bench_example.py",
)
```

The context gives a case three kinds of path:

| Method | Returns |
|---|---|
| `context.output(name)` | a file in this run's folder |
| `context.latest(name)` | the newest `name` from an earlier run of this model, or `None` |
| `context.input(name)` | a file in `models/<model>/tests/inputs/` |

Use `latest` when one test consumes another test's evidence, for example a
worker sweep that gates a later comparison. Raise a `RuntimeError` that names
the missing prerequisite when it returns `None`.

`script_case(runtime=..., ...)` and `production_case(...)` build common cases
from a benchmark and its arguments. A model's `tests/__init__.py` can define
`canonical_environment(context)` to give every case its launch environment.
