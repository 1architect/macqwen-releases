# MACQWEN

[![CI](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml/badge.svg)](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml)

MACQWEN runs large Qwen models locally on low-memory Apple Silicon Macs.
It keeps the core model in unified memory and streams selected data from SSD.

The tested system is an M4 Mac with 16 GB of unified memory and a 256 GB SSD.
The project includes no model weights.

## Reference performance

| Operation | Result |
|---|---:|
| Short terminal generation | About 2.0 to 2.5 tok/s |
| Complete oQ4 benchmark decode | 2.71 tok/s |
| Cache-aware oQ4 benchmark decode | 2.91 tok/s |
| Long-prompt prefill near 5,000 tokens | About 40 to 50 tok/s |

These results come from the reference Mac.
Speed changes with memory pressure, SSD state, and the macOS file cache.
See [measurement evidence](docs/flashnext/measurements/) for test conditions.

## Current support

| Runtime | Status | Typical use |
|---|---|---|
| Qwen3.8-Flash-Next | Primary | Local chat and compatible local APIs |
| Qwen3.8-27B | Research | Existing custom V4 installations |

Flash-Next supports regular local testing. The project remains experimental.

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
| oQ4 | 111.7 GB | Quality baseline; recommended for code and accurate API names |
| oQ3-MTP | 86.2 GiB | Public quick start; faster and smaller |

oQ4 passed a recorded API coding test that oQ3-MTP failed.
Use oQ3-MTP for prose, general chat, and the smallest supported installation.
Read the [Flash-Next research record](docs/flashnext/research.md) for the comparison.

The production runtime does not use the MTP weights included with oQ3-MTP.

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
models/qwen27b/          Qwen3.8-27B runtime and research utilities
docs/                    Current guides, results, and historical records
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
