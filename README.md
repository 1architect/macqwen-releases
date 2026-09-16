# MACQWEN

[![CI](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml/badge.svg)](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml)

MACQWEN runs large Qwen models locally on low-memory Apple Silicon Macs.
It keeps the core model in unified memory and streams selected data from SSD.

The tested system is an M4 Mac with 16 GB of unified memory and a 256 GB SSD.
The project includes no model weights.

## Reference performance

| Operation | Result |
|---|---:|
| REAP terminal sanity, 32 tokens | 3.74 tok/s median, 3.45 tok/s tail, 193.3 MB/token |
| Historical Q4/G32 60-slot control | 3.08 tok/s gen, 3.00 tok/s tail, 279.7 MB/token |
| Long-prompt prefill near 5,000 tokens | About 40 to 50 tok/s; 62.19 tok/s synthetic result |

These results come from the reference Mac. Speed changes with memory pressure,
SSD state, and the macOS file cache. See the
[measurement evidence](docs/flashnext/measurements/) for test conditions.

The listed controlled results preserve token IDs.

## Quick start

You need an Apple Silicon Mac, Python 3.12, a fast SSD, and enough free space
for one checkpoint. We test on an M4 Mac with 16 GB of unified memory.

Clone MACQWEN and create its Python environment:

```bash
git clone https://github.com/1architect/macqwen-releases.git
cd macqwen-releases
./chat.sh setup
```

For the smallest public setup, download oQ3-MTP:

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ3-MTP"
```

Start chatting:

```bash
./chat.sh --checkpoint oq3
```

MACQWEN remembers the selected checkpoint. After the first run, `./chat.sh` is
enough.

## Choose a checkpoint

Model weights are not included in this repository.

| Checkpoint | Disk size | Choose it when... |
|---|---:|---|
| oQ3-MTP | 86.2 GiB | You want the smallest public general-chat setup |
| oQ4 | 111.7 GB | Code quality and accurate external API names matter most |
| REAP-288 | 73.5 GB | You are helping test the current research checkpoint |

The production runtime does not use the MTP weights included with oQ3-MTP.
Our recorded SketchUp API test passed on oQ4 and failed on oQ3-MTP, so prefer
oQ4 for code that depends on exact third-party APIs. REAP-288 support is
available, but its complete quality and performance gates remain open.

Download oQ4:

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ4 \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

Download REAP-288:

```bash
hf download sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-REAP-288-MLX-4bit"
```

When only one compatible checkpoint is installed, MACQWEN finds it
automatically. Otherwise, select it by alias or full path:

```bash
./chat.sh --checkpoint oq4
./chat.sh --checkpoint oq3
./chat.sh --checkpoint /path/to/a/compatible-checkpoint
```

Set `MACQWEN_MODEL_ROOT` when checkpoints are outside `~/models`.
Set `MACQWEN_FLASHNEXT_PYTHON` when the Python environment uses another path.

## Daily use

Run `/help` inside the chat to see the current commands. The essentials are:

```text
/help [all]
/new
/session save|load|list|delete [name]
/config [section] ...
/status
/quit
```

Use `/status` to inspect the model, routing mode, context, and memory. Use
`/config display animate off` if you prefer output without the text animation.

## Routing modes

The default `exact-quality` mode is the right choice for most users:

```bash
./chat.sh --exact-quality
```

Other modes are available for controlled experiments:

```bash
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

Read the [Flash-Next brief](docs/flashnext/brief.md) for current mode status.
Read the [Flash-Next research record](docs/flashnext/research.md) for full results.

## How it works

MACQWEN keeps the core model in unified memory and reads the routed experts it
needs from SSD. Flash-Next is a good fit because its experts are small enough
to stream selectively. The normal launcher uses MLX and Metal; it does not use
our abandoned native-runtime prototype.

For REAP-288, normal chat uses generic MLX for Q4/G64 expert execution. Our
custom G64 executor, G64 slabs, and expert-major stream packing remain opt-in
research paths. You do not need to configure these paths for regular chat.

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
