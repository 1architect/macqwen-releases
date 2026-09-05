# MACQWEN

[![CI](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml/badge.svg)](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml)

MACQWEN runs large Qwen models locally on low-memory Apple Silicon Macs.
It keeps the core model in unified memory and streams selected data from SSD.

The tested system is an M4 Mac with 16 GB of unified memory and a 256 GB SSD.
The project includes no model weights.

## Reference performance

| Operation | Result |
|---|---:|
| Short terminal generation | About 2.0 to 3.0 tok/s |
| Current 60-slot Frontier 8A + Up-QMV/SwiGLU | 3.08 tok/s gen, 3.00 tok/s tail, 279.7 MB/token |
| Corrected 16-worker decode-only control | 3.13 tok/s gen, 3.04 tok/s tail, 262.0 MB/token |
| Long-prompt prefill near 5,000 tokens | About 40 to 50 tok/s; 62.19 tok/s synthetic result |

These results come from the reference Mac.
Speed changes with memory pressure, SSD state, and the macOS file cache.
See [measurement evidence](docs/flashnext/measurements/) for test conditions.

The listed controlled results preserve token IDs.

## Current FlashNext runtime

The current FlashNext runtime adds a custom Metal execution path for the
FlashNext MoE layers. The launcher enables these current settings:

```text
FLASHNEXT_METAL_RUNTIME=1
FLASHNEXT_SLAB_GLOBAL=60
FLASHNEXT_SLAB_PACK=1
FLASHNEXT_SLAB_POLICY=skew
FLASHNEXT_FUSED_SHARED=1
FLASHNEXT_FUSED_UP_SWIGLU=1
```

The current controlled 60-slot Frontier 8A profile with Up-QMV/SwiGLU measured
3.08 tok/s generation, 3.00 tok/s tail, and 279.7 MB/token. All arms kept the
same token digest. Use the research record for comparison rules and status.

The corrected decode-only 16-worker control measured 3.13 tok/s generation,
3.04 tok/s tail, 262.0 MB/token, and 3,620.7 MB active memory. The result is a
control measurement, not a new production guarantee.

### Custom Metal kernels

The custom path lives in `models/flashnext/metal_runtime.py` and
`models/flashnext/metal_runtime_native.mm`. It uses SIMD Q4/G32 helpers with
float32 activations and bfloat16 scales and biases. It combines the routed
down-projection with router scores and writes the output without the separate
intermediate routed tensor. It also removes 48 `astype` launches per token.

The isolated production-shape kernel is bit-identical to the MLX reference and
measures about 3.5% to 4.4% faster across controlled miss levels. The current
60-slot Frontier 8A profile with Up-QMV/SwiGLU measures 3.08 tok/s generation,
3.00 tok/s tail, and 279.7 MB/token with the custom path active. The corrected
16-worker control measures 3.13 tok/s generation, 3.04 tok/s tail, and
262.0 MB/token. Both controls preserve token digests.

The direct 16-arm stock-versus-custom comparison measured 2.91 tok/s for the
custom path versus 2.89 tok/s for stock MLX. Its +0.7% median difference stayed
inside a 7.8% resolution band. This comparison does not limit the newer
profile's measured rate.

The slab path uses page-aligned, file-backed storage and direct expert-major
Metal addressing. The 60-slot setting remains the engineering control. Frontier
8B and streamed expert-major records remain disabled.

## Current support

| Runtime | Status | Typical use |
|---|---|---|
| Qwen3.8-Flash-Next | Primary | Local chat and compatible local APIs |
| Qwen3.8-27B | Research | Existing custom V4 installations |

Flash-Next supports regular local testing. The project remains experimental.


## Why Qwen3.8-FlashNext and not others?

Qwen3.8-Flash-Next has unusually fine-grained experts. Its moe_intermediate_size is 640 against a 2560 hidden size, so one expert is 3.07 MB. Considering the architecture of the challenge (running big models on low memory/high bandwidth machines) it is close to a best case scenario.

## Requirements

- An Apple Silicon Mac with Metal support.
- Python 3.12.
- A fast local SSD.
- About 120 GB for oQ4, or 100 GB for oQ3-MTP.

Keep enough free SSD space for macOS and temporary files.
The 256 GB reference Mac normally holds only one Flash-Next checkpoint.

## Choose a Flash-Next checkpoint

| Checkpoint | Size | Guidance |
|---|---:|---|
| REAP-288 | 73.5 GB | Current research checkpoint; quality and speed gates remain open |
| oQ4 | 111.7 GB | Quality baseline; recommended for code and accurate API names |
| oQ3-MTP | 86.2 GiB | Public quick start; faster and smaller |

The current REAP checkpoint is `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`.
It uses Q4/G64 expert weights and Q4/G32 n-gram weights. Its expert path
uses reference streaming. The custom Q4/G32 Metal executor and packed slabs
remain inactive until compatible support is measured.

oQ4 passed a recorded API coding test that oQ3-MTP failed.
Use oQ3-MTP for prose, general chat, and the smallest supported installation.
Read the [Flash-Next research record](docs/flashnext/research.md) for the comparison.

The production runtime does not use the MTP weights included with oQ3-MTP.

An external [oQ4-MTP report](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4/discussions/2) describes degenerate repetition during long tool-use and coding turns. The run reached `max_tokens` and cut off tool calls. The same report did not reproduce the failure with oQ3-MTP under matching settings. The report uses oMLX on an M5 Max, and the checkpoint maintainer is investigating. This warning applies to the MTP variant. MACQWEN keeps MTP disabled in the production path.

## Quick start

Clone the repository:

```bash
git clone https://github.com/1architect/macqwen-releases.git
cd macqwen-releases
```

Create the tested Python environment:

```bash
./chat.sh setup
```

This command creates `.venv` and installs the pinned Flash-Next dependencies.

Download the current REAP research checkpoint:

```bash
hf download sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-REAP-288-MLX-4bit"
```

Download the public oQ3-MTP checkpoint:

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ3-MTP"
```

Verify the download:

```bash
find "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ3-MTP" \
  -name 'model-*-of-*.safetensors' | wc -l
du -sh "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ3-MTP"
```

The first command must report `19`.
The checkpoint contains 86.2 GiB of model weights.

Start the chat:

```bash
./chat.sh
```

MACQWEN selects a checkpoint automatically when one compatible checkpoint exists.
Select one explicitly when the model directory contains multiple checkpoints:

```bash
./chat.sh --checkpoint oq4
./chat.sh --checkpoint oq3
./chat.sh --checkpoint /path/to/a/compatible-checkpoint
```

The explicit selection stays in the preferences file.
Set `MACQWEN_MODEL_ROOT` when checkpoints are outside `~/models`.
Set `MACQWEN_FLASHNEXT_PYTHON` when the Python environment uses another path.
The launcher also accepts an active compatible Python environment.

## What to expect

The default `exact-quality` mode prioritizes output quality.
The model reads selected expert data from SSD during generation.
Baseline resident memory is about 4.19 GB.
Pinned memory is about 4.66 GB.
The performance figures are measurements, not minimum guarantees.

## Daily use

Run `/help` inside the chat for the current command list.

Primary commands:

```text
/help [all]
/new
/session save|load|list|delete [name]
/config [section] ...
/status
/quit
```

`/help` shows these primary commands. `/help all` also shows compatibility
commands. Existing commands such as `/thinking`, `/settings`, `/save`, and
`/reset` remain accepted.

The chat uses Qwen's recommended thinking-mode sampling defaults.
Use `/config sampling greedy` only when deterministic output is necessary.

The default answer limit is 4,096 tokens.
The default thinking capacity is 512 additional tokens.
Use `/status` to inspect the active model, sampling, routing, context, and memory values.

The terminal streams complete words and shows prefill progress.
Use `/config display animate off` to disable the text fade.

## Routing modes

Select a mode at startup:

```bash
./chat.sh --exact-quality
./chat.sh --cache-aware
./chat.sh --standard
./chat.sh --fast
./chat.sh --fast-quality
./chat.sh --fused-quality
```

| Mode | Purpose |
|---|---|
| `exact-quality` | Default mode with selective expert residency |
| `cache-aware` | Faster mode with small routing substitutions |
| `standard` | Threshold routing without selective pinning |
| `fast` | Aggressive approximate routing |
| `fast-quality` | Approximate routing with quality recovery |
| `fused-quality` | Experimental draft verification |

Use `exact-quality` for code, long work, and tasks that require precise facts.
Cache-aware routing changes some expert choices and can change the reply.
The fast modes trade output accuracy for speed.

Use `--threshold 1.0` to keep every expert selected by the shipped router.
The default threshold is `0.85`.

Read the [Flash-Next brief](docs/flashnext/brief.md) for current mode status.
Read the [Flash-Next research record](docs/flashnext/research.md) for full results.

## Local API server

Start the server:

```bash
./chat.sh /server
```

The default address is `http://127.0.0.1:8080`.
The server processes one generation at a time.

| Protocol | Endpoint |
|---|---|
| OpenAI Responses | `/v1/responses` |
| OpenAI Chat Completions | `/v1/chat/completions` |
| Anthropic Messages | `/v1/messages` |

Localhost mode accepts any client key.
The server reuses its cache when the next prompt extends the prior prompt exactly.

Allow a specific browser origin with:

```bash
./chat.sh --server --allow-origin http://localhost:3000
```

A non-local address requires a shared key:

```bash
export MACQWEN_SERVER_API_KEY="PRIVATE_VALUE"
./chat.sh --server --host 0.0.0.0
```

Keep the server on localhost or a trusted local network.
Read [SECURITY.md](SECURITY.md) before changing the host or enabling repository tools.

## Keys and local data

Tavily requires an API key. Context7 accepts an optional key.
Manage these keys inside the chat:

```text
/config keys
/config keys set tavily
/config keys set context7
/config keys delete tavily
```

The compatibility commands `/keys` and `/api-keys` remain accepted.
Key management works in both chat profiles.

Key input does not echo.

| Data | Default location |
|---|---|
| Preferences | `~/.macqwen/preferences.json` |
| API keys | `~/Library/Application Support/MACQWEN/api_keys.json` |
| Flash-Next sessions | `~/.cache/flashnext/sessions/` |
| Qwen3.8-27B sessions | `~/.frankenstein/sessions/` |

Session files can contain private prompts and model state.
Do not publish session files, credentials, or custom system prompts.

## Qwen3.8-27B research runtime

This runtime requires a custom MLX environment and a compatible local V4 checkpoint.
The repository does not provide a ready V4 checkpoint.

```bash
MACQWEN_QWEN27B_PYTHON=/path/to/python \
  ./chat.sh BUILD --profile plain
```

Read the [Qwen3.8-27B handoff](docs/qwen27b/handoff.md) for setup and validation details.

## Troubleshooting

- If no checkpoint appears, pass its full path with `--checkpoint`.
- If the checkpoint is incomplete, resume the `hf download` command.
- If generation becomes slower, close memory-heavy apps and retry.
- If multiple checkpoints exist, select `oq4`, `oq3`, or a full path.
- If a command changed, run `/help` for the active command list.

The loader checks the checkpoint configuration, index, and required shard files.

## Repository layout

```text
macqwen/                 Shared chat, commands, settings, and tools
models/flashnext/        Flash-Next runtime and benchmarks
models/flashnext/settings/ FlashNext setting registry and launch defaults
models/flashnext/tests/  FlashNext interactive research test catalog
models/qwen27b/          Qwen3.8-27B runtime and research utilities
docs/                    Current guides, results, and historical records
docs/MLX/               MLX Metal backend source notes
docs/flashnext/graphics/ FlashNext trace screenshots and plots
```

`chat.sh` selects the model, checkpoint, and Python environment before loading the runtime.

## Tests

Run shared tests:

```bash
.venv/bin/python -m unittest discover \
  -s macqwen -p 'test_*.py'
```

Run Flash-Next tests:

```bash
.venv/bin/python -m unittest discover \
  -s models/flashnext -p 'test_*.py' -q
```

## Documentation

The [documentation index](docs/README.md) links to each current brief, research record, and handoff.
Historical experiments stay in [docs/archive](docs/archive/README.md).

## License

MACQWEN source code uses the MIT License.
Models and dependencies use their own licenses.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

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
- The Qwen team provides the model architecture, tokenizer, sampling guidance,
  and checkpoint family used by MACQWEN.
- Vontra provides the public FlashNext MLX checkpoints used in this work.
- Tavily and Context7 provide optional search and documentation tools for the
  repository-tool chat profile.
- Python provides the runtime and standard-library components.
- GitHub provides repository hosting, issue tracking, and GitHub Actions CI.
- The MACQWEN research record credits external reports and source notes where
  they affect a measurement or a decision.

Each dependency, model, checkpoint, and platform tool keeps its own license
and terms. Review those terms before redistribution.
