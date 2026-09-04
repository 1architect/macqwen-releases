# FlashNext terminal test suite

This terminal uses the same colors, prompts, progress glow, and compact command
language as `chat.sh`. It does not run a test until the user types `/run TEST`
and confirms with `yes`.

Start it with:

```bash
./models/flashnext/tests/run.sh
```

Run trusted performance tests in Apple Terminal. An embedded Codex terminal
activates the Codex renderer, GPU service, and WindowServer while the benchmark
runs. The suite warns when `TERM_PROGRAM` is not `Apple_Terminal`.

The terminal parent never imports MLX. Preflight helpers live in
`models/flashnext/system_state.py`, which keeps the parent process from holding
a second Metal device beside the benchmark child.

Production comparisons keep per-read I/O profiling off. Use a separate
diagnostic case ending in `-attribution` when queue and positioned-read timing
is required.

Primary commands:

```text
/list performance
/show up-swiglu
/run up-swiglu
/controls
/config tokens 32
/config pairs 6
/config workers 16
/config purge off
/research prefetch
/results
```

The terminal discovers every `case_*.py` file in this folder. Adding a file
does not require a central registry change.

Each case file must provide `TEST`, `TESTS`, or `get_tests()`. Every returned
`TestSpec` must include:

- a unique ID and title;
- a plain explanation;
- why the test was proposed;
- metrics and controls;
- a source reference;
- an executable script function for runnable tests.

Case files can also supply environment controls, a custom live metric parser,
and a custom interpreter. This lets a new file test model behavior that the
main terminal does not know yet.
The terminal keeps ownership of commands, confirmation, live display, result
storage, and interruption.

The catalog has three evidence levels:

- Runnable retained benchmarks.
- Verification and manual quality entries.
- Historical entries generated from every research heading, bullet, and data
  row. Removed prototypes remain visible but cannot run.

Every runnable test shows:

- what it does;
- why it was proposed;
- enabled controls;
- expected metrics;
- the exact command;
- live per-arm metrics;
- a final interpretation;
- a JSON result record.

The default control is 60-slot skew plus Frontier 8A. Benchmarks use greedy
decoding and exact digests. The user evaluates final quality with `chat.sh`,
normal sampling, and `xhigh` effort.

The suite never requires a reboot. The VM quiescence gate and file-cache purge
are optional diagnostics and disabled by default.
