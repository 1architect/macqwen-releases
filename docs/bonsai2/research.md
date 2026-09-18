# Bonsai-2 ternary 27B research record

This is our active record of Bonsai-2 measurements, rejected ideas, and
design decisions. Current operation belongs in [`handoff.md`](handoff.md).
Exact commands, the compact results table, and retained JSONL arms will live
in [`measurements/`](measurements/).

## 2026-09-18 — Branch setup

We created branch `bonsai-2-research` and added `models/bonsai2/` following
the K2-Horizon package pattern: settings, checkpoint discovery with
`prism_hadamard_qwen35` compatibility, resident backend, protocol translator,
KV cache helper, fresh-process bench harness, and checkpoint-free tests.

Key loader finding from the downloaded checkpoint: the pack stores
`language_model.*` plus `vision_tower.*` keys in one `model.safetensors`.
The text-only path must use the bundled `runtime/vision_artifact.py`
`load_vl_model` entry point and run its `language_model`. The bare
`runtime/artifact.py` `load_model` path targets text-only packs and does not
match this pack's key namespace. Stock `mlx_lm.load` skips the Hadamard
activation transform and returns wrong output silently, so our backend
refuses to run when `runtime/` files are absent.

We downloaded `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (about 8.6 GB) to
`~/models/Ternary-Bonsai-2-27B-mlx-2bit`. Milestone 1 stays text-only with
the vision tower unloaded.

## 2026-09-18 — Live smoke test

We ran the resident backend against the downloaded checkpoint in `.venv`
(MLX 0.32.2, MLX-LM 0.31.3, mlx-vlm 0.6.17). Both arms used the bundled
`runtime/vision_artifact.py` loader and kept the vision tower unloaded.

- No-thinking hello prompt, 64 prompt tokens, 32-token limit: finish `stop`,
  3 tokens, visible `Hello!`, cache invariant holds.
- Thinking `medium` arithmetic prompt (17 times 23), 72 prompt tokens,
  64-token limit: finish `stop`, 52 tokens at about 8.5 tok/s, correct
  answer 391 with reasoning closed, cache invariant holds.

Three loader facts follow from this run. The text path must use
`load_vl_model` with `load_processor=False` and run its `language_model`;
the bare `artifact.load_model` path does not match this pack's
`language_model.*` key namespace. The language model answers
`LanguageModelOutput`, so our backend unwraps `.logits` before the shared
`generate_step` helper. Its prompt cache mixes 48 `ArraysCache` and 16
`KVCache` layers, so validation and the invariant accept both kinds instead
of requiring every layer to carry an offset.

## 2026-09-18 — Truncated turns after tool use

Interactive agent chat ended turns with no visible answer right after the
model reached for a tool. The cause was our protocol translator, not the
model. Bonsai-2 emits native Qwen XML calls
(`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`),
which already match the shared contract. The first translator tried to
JSON-parse every tool block and replaced non-JSON content with a bare close
tag, swallowing the function name. The tool filter then found no function,
nothing executed, and the turn closed with an empty answer.

The fix passes non-JSON tool blocks through untouched and converts JSON
blocks to the shared XML form as before. A forced `list_dir` probe now
parses to `[('list_dir', {'path': '.'})]`, and a regression test pins the
passthrough across split-marker chunkings.

## Promotion rules

Future comparisons must freeze checkpoint, prompt IDs, template, sampler,
reasoning effort, token limits, interpreter, and source fingerprints. Use one
fresh process per arm, at least three arms per condition, and
forward/reverse/forward ordering. Avoid arbitrary warmups on the fanless Mac.

Report paired effects and the two-standard-error resolution band. A candidate
must clear both a 5% effect and that band, or provide a substantial measured
memory saving at identical context and output lengths. Exact operations
require intermediate equality and whole-run greedy digest checks.
Output-changing work also requires the sampled quality gate in
`CONTRIBUTING.md`.

We keep ternary weights, every context token, all layers, and current model
semantics for milestone 1. Vision input, changed reasoning budgets, disk
offload, KV recomputation, and speculative decoding are outside this scope.
