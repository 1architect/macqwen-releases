# MACQWEN

[![CI](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml/badge.svg)](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml)

MACQWEN runs large language models on low-memory Apple Silicon Macs. Its
primary runtime streams selected Flash-Next model data from SSD while keeping
the dense model core in unified memory.

The repository contains runtime code, tests, documentation, and measurement
records. It does not contain model weights.

## Supported runtimes

| Runtime | Role | Checkpoint | Launch |
|---|---|---|---|
| Flash-Next | Primary SSD-streamed runtime | oQ4 quality baseline; REAP-288 research checkpoint | `./chat.sh --model flashnext --checkpoint oq4` |
| K2-Horizon 7B | Resident MLX alternative | Official 8-bit MLX checkpoint | `./chat.sh --model k2-horizon --checkpoint k2` |
| Bonsai-2 27B | Experimental ternary, text-only runtime | Official 2-bit MLX checkpoint | `./chat.sh --model bonsai2 --checkpoint b2` |
| Qwen3.8-27B | Research runtime | Compatible local V4 build | `./chat.sh BUILD --profile plain` |

## Reference performance (Flash-Next)

| Runtime | Operation | Result |
|---|---|---:|
| Flash-Next | REAP terminal sanity, 32 tokens | 3.74 tok/s median, 3.45 tok/s tail, 193.3 MB/token |
| Flash-Next | Historical Q4/G32 60-slot control | 3.08 tok/s generation, 3.00 tok/s tail, 279.7 MB/token |
| Flash-Next | Long-prompt prefill near 5,000 tokens | About 40–50 tok/s; 62.19 tok/s in a synthetic diagnostic |

These reference observations come from the M4 test system and cover
Flash-Next only. The REAP row is a short terminal sanity check, the Q4/G32
row is historical control data, and the 62.19 tok/s figure is not production
throughput. See the [Flash-Next measurement evidence](results/flashnext/)
for conditions and provenance.

Flash-Next REAP-288 currently uses the G64 Metal executor by default. Its
short exact-digest speed result supports that executor choice under the tested
conditions; long-turn quality remains unverified. See the
[Flash-Next handoff](docs/flashnext/handoff.md) for the operational state.

We test on an M4 Mac with 16 GB of unified memory and a 256 GB SSD. Results
vary with memory pressure, SSD state, and the macOS file cache.

## Quick start

Requirements: an Apple Silicon Mac, Python 3.12, a fast SSD, and enough free
space for at least one checkpoint.

Clone the repository and create its managed environment:

```bash
git clone https://github.com/1architect/macqwen-releases.git
cd macqwen-releases
./chat.sh setup
```

Download the recommended oQ4 checkpoint:

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ4 \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

Start a chat:

```bash
./chat.sh --model flashnext --checkpoint oq4
```

When exactly one compatible checkpoint is installed, MACQWEN selects it
automatically. With multiple checkpoints, select one with `--model` and
`--checkpoint`.

Run the project live-test terminal:

```bash
./tests/run.sh
```

It asks which installed runtime and checkpoint to test, discovers that
runtime's cases, and writes each run to its own folder under
`results/<runtime>/`. See [docs/testing.md](docs/testing.md).

## Flash-Next

Flash-Next streams routed experts and n-gram data from SSD. `exact-quality` is
the normal routing profile. The other profiles are research controls and may
change output; use them only with the quality and measurement rules in the
[Flash-Next documentation](docs/flashnext/brief.md).

The current REAP rollback is generic MLX expert execution:

```bash
FLASHNEXT_METAL_G64=0 ./chat.sh --checkpoint "$HOME/models/Qwen3.8-Flash-Next-REAP-288-MLX-4bit"
```

G64 slabs, stream packing, and QSA optimization flags remain off. The removed
native-runtime prototype is not part of the supported launcher.

### Flash-Next checkpoints

| Alias | Checkpoint | Use |
|---|---|---|
| `oq4` | `Vontra/Qwen3.8-Flash-Next-MLX-oQ4` | Recommended quality baseline |
| `oq3` / `oq3-mtp` | `Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP` | Disk-constrained research only; it failed the recorded code-quality gate |
| — | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` | Current research checkpoint; use its full path |

Download REAP-288 with:

```bash
hf download sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-REAP-288-MLX-4bit"
```

Set `MACQWEN_MODEL_ROOT` when checkpoints live outside `~/models`. The
launcher accepts a full checkpoint path with `--checkpoint` and validates the
checkpoint before loading it.

## Other runtimes

### K2-Horizon 7B

K2-Horizon is a resident 7B MLX model. It does not use Flash-Next streaming or
routing.

```bash
hf download abenzerps/K2-Horizon-7B-MLX-8bit \
  --local-dir "$HOME/models/K2-Horizon-7B-MLX-8bit"
./chat.sh --model k2-horizon --checkpoint k2
```

The checkpoint supplies executable `model.py` code. Use a checkpoint source we
trust. Read the [K2-Horizon brief](docs/k2_horizon/brief.md) and
[handoff](docs/k2_horizon/handoff.md) before changing or benchmarking it.

### Bonsai-2 27B

Bonsai-2 is an experimental ternary-weight 27B runtime. Milestone 1 is
text-only and uses the checkpoint's bundled `runtime/` loader.

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
  --local-dir "$HOME/models/Ternary-Bonsai-2-27B-mlx-2bit"
./chat.sh --model bonsai2 --checkpoint b2
```

The backend validates the bundled loader before selecting the checkpoint. Use
a checkpoint source we trust. See the [Bonsai-2 brief](docs/bonsai2/brief.md),
[research record](docs/bonsai2/research.md), and
[handoff](docs/bonsai2/handoff.md).

### Qwen3.8-27B

The research runtime uses the managed environment and a compatible local V4
checkpoint. A supported build must include `bf16-ends/` beside its weights.
`BUILD` is the directory suffix under the model root:

```bash
./chat.sh BUILD --profile plain
```

See the [Qwen3.8-27B handoff](docs/qwen27b/handoff.md) before using or
benchmarking this runtime.

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

Use `/status` for model, profile, routing, context, and memory information.
Use `/config display animate off` to disable output animation.

## Local API server

Start the local server with:

```bash
./chat.sh --server
```

The compatibility command `/server` is also accepted inside chat. The default
address is `http://127.0.0.1:8080`, and the server processes one generation at
a time.

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
- Several checkpoints are installed: provide both `--model` and `--checkpoint`.
- Generation slows down: close memory-heavy applications and retry.
- K2-Horizon or Bonsai-2 is not selected: provide its model and checkpoint aliases explicitly.
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
running experiments.

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
models/flashnext/           Flash-Next runtime and benchmarks
models/k2_horizon/          K2-Horizon runtime, adapter, settings, and tests
models/bonsai2/             Bonsai-2 runtime, kernels, settings, and tests
models/qwen27b/             Qwen3.8-27B runtime and research utilities
docs/                       Current guides, evidence, and historical records
```

## License

MACQWEN source code uses the MIT License. Models and dependencies use their
own licenses. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

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
