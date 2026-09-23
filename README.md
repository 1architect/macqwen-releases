# MACQWEN

[![CI](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml/badge.svg)](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml)

MACQWEN streams sparse Mixture-of-Experts language models from SSD on
low-memory Apple Silicon Macs. The primary runtime keeps the dense model
core in unified memory while streaming selected expert rows and n-gram
rows from the checkpoint on demand.

Three resident runtimes (K2-Horizon 7B, Bonsai-2 27B, Qwen3.8-27B, all
smaller than the primary MoE) are
supported for comparison and research, but they are not the project's
focus. Everything below leads with the streamed MoE runtime.

## Primary runtime: Flash-Next (SSD-streamed sparse MoE)

| Item | Value |
|---|---|
| Checkpoint | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (installed; alias `vontra-mtp`) |
| Weights | 22 safetensors shards, 3,747 indexed tensors (incl. 76 MTP), 113.2 GB / 105.4 GiB |
| Quantisation | Uniform 4-bit affine, group 32; multimodal modules and MoE router gates stay BF16 |
| Architecture | `qwen4_exp` sparse MoE: 48 layers, 512 routed experts (top-10) + 1 shared, vocab 248,320, 262,144 context |
| Language stack | 125B total / 6B active, plus 51B n-gram table (20M entries) and a 4B MTP draft block (MTP disabled in production) |
| Launch | `./chat.sh --model flashnext --checkpoint vontra-mtp` |

Streaming means each token reads only the experts and n-gram rows it
needs. `exact-quality` is the normal routing profile; the other profiles
(`standard`, `cache-aware`, `fast-quality`, `fused-quality`) are research
controls — see the [Flash-Next brief](docs/flashnext/brief.md).

Current chat defaults live in `models/flashnext/settings/launch.py`:
Metal runtime, a 6 GB application-owned expert pool read in place by the
Metal kernels (it replaces the static slab pack and the expert pins),
GPU keep-warm on, one host sync per decode layer, compiled GatedDeltaNet
glue, and the exact opt-in bundle (streamed embedding, compiled
glue/norm, cached norm gain, interactive QoS, parallel n-gram reads, both
QSA flags, stream-pack chunk 2). Every default keeps the exact token
digest. The pool needs about 6 GB of free memory; on a busier Mac, roll
it back with an explicit value, for example:

```bash
FLASHNEXT_EXPERT_POOL_GB=0 ./chat.sh --model flashnext --checkpoint vontra-mtp
```

## Reference performance (Flash-Next, installed checkpoint)

| Operation | Result |
|---|---|
| Current defaults (6 GB expert pool), 128 tokens | 3.79–3.93 tok/s; 4.09 tok/s over tokens 65–128; +11.3% over the pool-off defaults, 3/3 pairs, two-SE 1.9%, digest `e19af44d5268e9d1` |
| Pool-off defaults (slab and 8 pins), 128 tokens, same session | 3.42–3.47 tok/s, 294–310 MB/token |
| Pool-off defaults before the host/GPU-glue stack, 128 tokens | 3.05–3.09 tok/s (earlier session); stack +6.6% inside 8.0% |
| 128-token paired arms, 8 pins, keep-warm off / on | 2.33–2.41 / 2.86–2.93 tok/s |
| `bench_production`, 96 tokens, keep-warm off / on | 2.19 / 2.55 tok/s |
| Chat, 8 pins (policy), 72 tokens | about 2.14 tok/s, about 350 MB/token |
| Historical 60-slot protocol, 32 pins, 32 tokens, 4 arms | 2.28 tok/s median (2.02–2.77), 2.36 tok/s tail, 402 MB/token |
| Historical oQ4 (not installed), 60-slot pack, 32 tokens | 3.08 tok/s, 3.00 tok/s tail, 279.7 MB/token |
| Historical REAP-288 (not installed), terminal sanity, 32 tokens | 3.74 tok/s median, 3.45 tok/s tail, 193.3 MB/token |
| Long-prompt prefill near 5,000 tokens (historical oQ4) | About 40–50 tok/s; 62.19 tok/s in a synthetic diagnostic (not production) |

Measured on an M4 Mac with 16 GB unified memory and a 256 GB SSD, on a
quiet machine (no other heavy applications); free memory moves the rate
more than most code changes, because the expert stream reads less when
more RAM holds experts. The oQ4/REAP rows cannot be reproduced here.
Short 32-token arms sit inside a ~40-token warm-up window and read higher
than longer answers. Evidence: [results/flashnext/](results/flashnext/)
(latest: `20260923-153742-expert-pool/`, `20260923-133612-read-replay-native/`),
[Flash-Next handoff](docs/flashnext/handoff.md). The earlier goal of
3 tok/s decode without quantizing or pruning is met; the current target is
4 tok/s (250 ms/token).

## Quick start

Requirements: Apple Silicon Mac, Python 3.12, a fast SSD, ~120 GB free
for the checkpoint, and about 10 GB of free memory for the default
Flash-Next chat (about 3.4 GB model core plus the 6 GB expert pool).

```bash
git clone https://github.com/1architect/macqwen-releases.git
cd macqwen-releases
./chat.sh setup
hf download Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
./chat.sh --model flashnext --checkpoint vontra-mtp
```

With exactly one compatible checkpoint installed, MACQWEN selects it
automatically. Set `MACQWEN_MODEL_ROOT` when checkpoints live outside
`~/models`. A full path also works with `--checkpoint`, and the
launcher validates the checkpoint before loading.

Run the project live-test terminal:

```bash
./tests/run.sh
```

It asks which installed runtime and checkpoint to test, discovers that
runtime's cases, and writes each run to its own folder under
`results/<runtime>/`. See [docs/testing.md](docs/testing.md).

## Secondary runtimes (supported, not the focus)

They run resident checkpoints with no SSD streaming. Prefer them only
for comparison or their own research questions.

### K2-Horizon 7B

Resident 7B MLX model (about 9.6 GB on disk).

```bash
hf download abenzerps/K2-Horizon-7B-MLX-8bit \
  --local-dir "$HOME/models/K2-Horizon-7B-MLX-8bit"
./chat.sh --model k2-horizon --checkpoint k2
```

The checkpoint supplies executable `model.py`; use a source we trust.
See the [brief](docs/k2_horizon/brief.md) and
[handoff](docs/k2_horizon/handoff.md).

### Bonsai-2 27B

Experimental ternary-weight 27B, text-only (Milestone 1), via the
checkpoint's bundled `runtime/` loader.

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
  --local-dir "$HOME/models/Ternary-Bonsai-2-27B-mlx-2bit"
./chat.sh --model bonsai2 --checkpoint b2
```

See the [brief](docs/bonsai2/brief.md) and
[handoff](docs/bonsai2/handoff.md).

### Qwen3.8-27B

Dense research runtime on a compatible local V4 build (must include
`bf16-ends/` beside its weights). `BUILD` is the directory suffix:

```bash
./chat.sh BUILD --profile plain
```

See the [handoff](docs/qwen27b/handoff.md). No V4 checkpoint is
currently installed.

## Daily use

Inside the chat, use:

```text
/help [all]
/new
/session save|load|list|delete [name]
/config [section] ...
/status
/quit
```

Use `/status` for model, profile, routing, context, and memory
information. Flash-Next extras: `/config model gpu-keepwarm off`
disables keep-warm and saves the choice. Use
`/config display animate off` to disable output animation.

## Local API server

Start the local server with:

```bash
./chat.sh --server
```

The compatibility command `/server` is also accepted inside chat. The
default address is `http://127.0.0.1:8080`, and the server processes one
generation at a time.

| Protocol | Endpoint |
|---|---|
| OpenAI Responses | `/v1/responses` |
| OpenAI Chat Completions | `/v1/chat/completions` |
| Anthropic Messages | `/v1/messages` |

Localhost does not require authentication by default. A non-local bind
requires a shared key:

```bash
export MACQWEN_SERVER_API_KEY="PRIVATE_VALUE"
./chat.sh --server --host 0.0.0.0
```

Allow browser requests only for a trusted origin:

```bash
./chat.sh --server --allow-origin http://localhost:3000
```

Read [SECURITY.md](SECURITY.md) before changing the host, allowing browser
origins, or enabling repository tools.

## Keys and local data

Manage optional Tavily and Context7 keys inside the chat:

```text
/config keys
/config keys set tavily
/config keys set context7
/config keys delete tavily
```

Key input does not echo. Session files can contain private prompts and model
state; keep them private.

| Data | Default location |
|---|---|
| Preferences | `~/.macqwen/preferences.json` |
| API keys | `~/Library/Application Support/MACQWEN/api_keys.json` |
| Flash-Next sessions | `~/.cache/flashnext/sessions/` |
| K2-Horizon sessions | `~/.cache/k2-horizon/sessions/` |
| Bonsai-2 sessions | `~/.cache/bonsai2/sessions/` |
| Qwen3.8-27B sessions | `~/.frankenstein/sessions/` |

The launcher uses `.venv` by default. Developers can provide a validated
override with `MACQWEN_PYTHON` or a model-specific `MACQWEN_*_PYTHON` variable.

## Troubleshooting

- No checkpoint appears: pass its full path with `--checkpoint`.
- A download is incomplete: resume the `hf download` command into the same directory.
- The Vontra download is ~105 GiB: confirm free space before retrying.
- Several checkpoints are installed: provide both `--model` and `--checkpoint`.
- Generation slows down: close memory-heavy applications and retry.
- A secondary runtime is not selected: pass its model and checkpoint aliases explicitly.
- A command has changed: run `/help` for the active command list.

## Tests and development

Run the shared unit tests:

```bash
.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'
```

Run a runtime's unit tests with the same pattern, replacing `RUNTIME`:

```bash
.venv/bin/python -m unittest discover -s models/RUNTIME/tests/unit -t . -p 'test_*.py' -q
```

Use `./tests/run.sh` for live-model evidence. Every run writes to
`results/<runtime>/`. Read [docs/testing.md](docs/testing.md),
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[measurement standard](docs/measurement-standard.md) before changing code or
running experiments. Do not run benchmarks without approval of the plan.

## Documentation and layout

The [documentation index](docs/README.md) lists the current component
documents. Each active runtime follows the same structure:

- `brief.md`: purpose, scope, status, and main results;
- `research.md`: dated measurements, decisions, and rejected approaches;
- `handoff.md`: current operation, validation, constraints, and next work.

Historical records are in [docs/archive](docs/archive/README.md). They preserve
provenance and may contain obsolete commands.

```text
macqwen/                    Shared chat, commands, settings, tools, and test engine
tests/                      Project live-test terminal
models/flashnext/           Flash-Next streaming runtime and benchmarks
models/k2_horizon/          K2-Horizon runtime, adapter, settings, and tests
models/bonsai2/             Bonsai-2 runtime, kernels, settings, and tests
models/qwen27b/             Qwen3.8-27B runtime and research utilities
docs/                       Current guides, evidence, and historical records
results/                    Per-run measurement records (one folder per run)
```

## License

MACQWEN source code uses the MIT License. Models and dependencies use their
own licenses. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The Vontra
checkpoint card carries the Qwen Community License 1.0; review it before
use or redistribution.

## Acknowledgements

MACQWEN uses and builds on these third-party projects and platform tools:

- Apple macOS, Apple Silicon, Metal, Objective-C++ Metal APIs, Xcode, and
  Instruments provide the runtime platform, custom kernels, and trace tools.
- MLX, MLX-LM, and MLX-VLM provide tensor execution, model loading, and model
  architecture support.
- Transformers, NumPy, Requests, and Hugging Face Hub provide tokenization,
  array operations, HTTP access, and checkpoint access.
- Hugging Face hosts the public checkpoint files and model discussions used in
  this project and its evidence record.
- The Qwen team provides the Qwen model architecture, tokenizer, sampling
  guidance, and checkpoint family used by MACQWEN.
- Vontra provides the public Flash-Next MLX checkpoints used in this work.
- IFM provides K2-Horizon, and abenzerps provides its public MLX checkpoint.
- Tavily and Context7 provide optional search and documentation tools for the
  repository-tool chat profile.
- Python provides the runtime and standard-library components.
- GitHub provides repository hosting, issue tracking, and GitHub Actions CI.
- The MACQWEN research record credits external reports and source notes where
  they affect a measurement or a decision.
