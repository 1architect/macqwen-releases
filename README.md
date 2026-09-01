# MACQWEN

[![CI](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml/badge.svg)](https://github.com/1architect/macqwen-releases/actions/workflows/ci.yml)

MACQWEN is an experimental runtime for local LLM inference on low-memory Apple
Silicon systems. The reference machine is an M4 Mac with 16 GB of unified
memory and a 256 GB SSD.

The project currently focuses on Qwen models. Qwen3.8-Flash-Next is the primary
runtime. Qwen3.8-27B remains a research runtime.

MACQWEN does not include model weights.

## Latest retained measurements

**Results from the reference M4 Mac with 16 GB of unified memory and a 256 GB
SSD:**

| Runtime | Prefill measurement | Decode measurement | Memory measurement |
|---|---:|---:|---|
| Qwen3.8-Flash-Next | Amortises with prompt length; about 40 to 50 tok/s near 5,000 tokens | Exact: 2.71 tok/s. Cache-aware: 2.79 tok/s in its hot comparison | 4.66 GB pinned; about 4.19 GB baseline resident |
| Qwen3.8-27B | 47.2 tok/s, 2026-08-20 | 4 to 6 tok/s V4, 2026-08-26 | 12.10 GB prefill peak; about 12.65 GB V4 resident weights |

### Decode

The standard harness, `models/flashnext/bench_production.py`, measures 2.713
tok/s for a complete decode and 2.650 tok/s for the pinned tail, over ten kept
arms at 390 MB of physical reads per token. Evidence is in
[measurements](docs/flashnext/measurements/).

The optional `cache-aware` profile measured 2.79 tok/s against 2.54 for exact
routing in one hot interleaved run. It read 347.6 MB per token instead of
417.8 MB. Pairing adjacent arms showed a mean 8.3 percent gain. Seven of eight
pairs were faster, and all eight read fewer bytes. This run does not replace
the 2.713 exact-quality baseline because its machine state was different.

An older harness, `bench_resident_tail.py`, measured a 2.59 tok/s pinned-tail
mean over ten arms, range 2.42 to 2.73. It reloads the model for each arm, so
it starts colder. The difference between 2.59 and 2.65 is page-cache state,
not a code change.

Terminal `gen` on short chat turns runs near 2.0 to 2.5 tok/s. This value is
the visible rate. The tail figures exclude the first eight decode tokens and
expert pinning, so they are not the same span.

The rate depends on how much of the checkpoint the page cache holds. Free
memory correlates with rate at -0.84 across thirteen arms: more free memory
means less of the checkpoint is cached and a slower run. Identical code spans
about 21 percent between a cold cache and a warm one, so quote a rate with its
conditions or not at all.

A 2.83 tok/s figure appeared in earlier documents. It is the mean of two arms
from a warmup sweep and is not a baseline.

### Prefill

Prefill is faster than decode because it amortises, not because it uses the
machine better. A layer reads each distinct expert once and serves every token
in the prompt with it, so distinct experts per layer saturates toward 512 while
the token count keeps rising.

| Tokens | tok/s | MB/token | Drive |
|---:|---:|---:|---:|
| 128 | 8.72 | 160.6 | 1.40 GB/s |
| 512 | 20.66 | 58.8 | 1.21 GB/s |
| 1024 | 30.75 | 34.1 | 1.05 GB/s |
| 2048 | 41.97 | 19.5 | 0.82 GB/s |

Sixteen times the tokens cost 1.94 times the bytes. The drive rate falls as
prefill speeds up, and decode sustains 1.06 GB/s, more than a 2048-token
prefill. Short prompt rates do not predict large-prompt throughput.

### Ceiling

With every expert read served from memory, the same model runs at 5.33 tok/s.
That figure comes from a synthetic test that pins one fixed expert set and does
not generate the model's real reply. It is a bound on the drive's contribution,
not a production rate. The drive accounts for the whole gap between it and
2.71: the GPU and the Python host cost 188 ms per token together.

The Qwen3.8-27B prefill test used 256-token steps. The later V4 decode test ran
without swap. These values come from separate controlled runs.

Do not use this table as a direct cross-model benchmark. See the
[Flash-Next research](docs/flashnext/research.md) and
[Qwen27B research](docs/qwen27b/research.md) for full measurement conditions.

## Supported runtimes

| Runtime | Architecture | Status | Low-memory method |
|---|---|---|---|
| Qwen3.8-Flash-Next | Sparse Mixture-of-Experts | Primary, tested checkpoint available | Stream selected experts and n-gram rows from SSD |
| Qwen3.8-27B | Dense hybrid | Research, local V4 build required | Mixed quantization, external ends, and paged KV |

The tested Flash-Next checkpoint has 176B parameters and uses about 111.7 GB
on disk.

## How MACQWEN works with low memory

### Dense Qwen3.8-27B

Each token uses almost all dense model weights. MACQWEN uses measured mixed-bit
quantization to keep important weights within the memory budget.

The runtime keeps the embedding and selected output rows on SSD. Quantized
paged KV bounds context memory. A 256-token prefill step controls temporary
memory. Append-only chat state avoids repeated prefill of old turns.

### Sparse Flash-Next

The complete MoE checkpoint cannot fit in 16 GB. Each token uses only a small
set of its 512 experts per routed layer.

MACQWEN keeps the dense core in memory. It reads selected expert rows and
n-gram rows from the original SSD shards. Bounded reads control temporary
memory. Page pinning keeps frequently used expert data in available RAM.

### Shared runtime

Both models save reusable state. The launcher loads only the selected backend
and its Python environment. The chat interface stays the same for both models.

## Flash-Next quick start

Requirements:

- Apple Silicon with Metal support.
- Python 3.12.
- About 120 GB of free storage.
- A fast local SSD.

Clone the repository:

```bash
git clone https://github.com/1architect/macqwen-releases.git
cd macqwen-releases
```

Create the pinned environment:

```bash
./chat.sh setup
```

This command creates `.venv` and installs the pinned Flash-Next dependencies.

Download the tested public checkpoint:

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ4 \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

Verify the download:

```bash
find "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4" \
  -name 'model-*-of-*.safetensors' | wc -l
du -sh "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

The first command must report `22`. The second should report about `112G`.

Run the default tuned profile:

```bash
./chat.sh
```

MACQWEN finds `.venv` automatically. It also finds the sole compatible
checkpoint under `~/models`. Use `MACQWEN_MODEL_ROOT` for another model folder.

When multiple compatible checkpoints exist, select one:

```bash
./chat.sh --checkpoint oq3
./chat.sh --checkpoint oq4
./chat.sh --checkpoint /path/to/checkpoint
```

You can also install the package in an existing Python 3.12 environment:

```bash
python -m pip install -e '.[flashnext]'
macqwen --checkpoint /path/to/checkpoint
```

This command always starts Flash-Next with `exact-quality`, threshold `0.85`,
32 resident experts, and an eight-token warmup.

Use explicit settings when needed:

```bash
./chat.sh --model flashnext --profile plain --exact-quality
```

Run the repository agent profile:

```bash
./chat.sh --model flashnext --profile agent --exact-quality \
  --workspace /path/to/repository
```

### Flash-Next routing quality

`--exact-quality` is the default profile. It preserves the token trajectory for
the `0.85` routing threshold while changing storage placement.

The name does not mean complete router selection. Use `--threshold 1.0` to keep
all experts selected by the shipped router.

```bash
./chat.sh --model flashnext --standard
./chat.sh --model flashnext --threshold 1.0
./chat.sh --model flashnext --fast
./chat.sh --model flashnext --fast-quality
./chat.sh --model flashnext --cache-aware
./chat.sh --model flashnext --fused-quality
```

The `fast` profiles change routing more aggressively. Treat their output as
approximate.

`--cache-aware` starts from `exact-quality`. It can replace a cold selected
expert with a discarded resident expert whose score is within `0.02`. This
reduced physical reads by 16.8 percent and improved paired decode by 8.3
percent in the retained run.

Cache-aware routing changes replies. A small factual gate lost no correct
answer. A later long-context dream-analysis comparison stayed coherent, but
we preferred the exact-quality answer. Use cache-aware as an optional
speed profile. Keep exact-quality for the strongest observed answer quality.

The convenience launcher selects this profile directly:

```bash
./chat-swap.sh
```

`--fused-quality` uses a one-shot 4B draft and exact target verification. It is
experimental. Its measured reasoning gate failed, so do not treat it as equal
to `--exact-quality`.

## Qwen3.8-27B research setup

The 27B runtime is not a turnkey installation. It needs a custom MLX
environment and a locally built compatible V4 checkpoint.

The launcher accepts V4 build directories with vocabulary size `248320`.
Select a build by its directory suffix:

```bash
MACQWEN_QWEN27B_PYTHON=/path/to/python \
  ./chat.sh BUILD --profile plain
```

The repository does not provide a ready V4 checkpoint. See the
[Qwen27B handoff](docs/qwen27b/handoff.md) for model paths, memory rules, and
validation commands.

## Shared chat

### Profiles

- `plain` provides direct conversation without tools.
- `agent` provides repository tools with approval controls.

Model and profile selection are independent.

### Main commands

```text
/thinking on|off|show|hide
/max-tokens N|off
/think-budget N|off
/effort low|medium|xhigh
/stream on|off
/animate on|off
/approval ask|auto                 agent profile only
/workspace PATH                   agent profile only
/profile plain|agent
/prompt [text|edit|default]
/keys [list|set SERVICE|delete SERVICE]
/status
/settings [NAME VALUE|defaults]
/server
/save [name]
/load [name]
/sessions
/delete NAME
/reset
/quit
```

Replies appear one complete word at a time, never a partial token. Each word
fades in through four shades of grey before it lands in its final colour.
Visible reasoning uses a darker version of the same animation. It starts on
the line below the input prompt. One blank line separates reasoning and answer
text.

The fade runs on an output worker and does not stop token generation. Its
default budget is 96 ms per word. `MACQWEN_FADE_MS=0` turns the fade off and
keeps the word buffering.

Use `/animate off` to disable the fade for the current and later chat runs.

Run `/help` for the complete current command list.

`/help` shows only the primary command names. Compatibility aliases remain
accepted but do not appear in the command descriptions.

### Token limits and statistics

`/max-tokens` controls the answer allowance. `off` restores the profile
default. The default is 4,096 tokens for `plain` and 2,048 tokens for `agent`.

`/think-budget` adds generation capacity when thinking is enabled. Its default
is 512 tokens. The runtime uses this total ceiling:

```text
answer allowance + thinking capacity
```

This extra capacity reduces answer truncation after long reasoning. It does
not enforce two independent phase counters inside the model.

The command-line equivalents are `--max-tokens N` and `--think-budget N`.
Use `-1` to select the corresponding default or disabled state.

The final statistics show new prompt tokens, prefill speed, complete decode
speed, pinned-tail speed, context size, and total turn time. The tail excludes
the first eight routing-observation tokens and the expert pin operation. Agent
statistics combine all model segments from the request, including segments
after tool results.

### Prefill and tool activity

The prefill line uses the same moving colour animation as normal output. Its
bar contains only `█` and `░` cells. It has no surrounding brackets.

Flash-Next drives the bar from completed streamed MoE layers. The displayed
progress converts the completed-layer share into prompt-token-equivalent
progress. The live rate and remaining time use these measured updates. Final
prefill statistics use the complete model prefill time.

The agent hides tool protocol markup. It shows `Preparing tool` when the model
starts a tool call. It shows the action name when that name becomes available.
The final line contains the completed action and real execution time.

Successful tool lines have no icon, emoji, or green status word. A short
minimum display interval keeps very fast tools visible. This interval does not
change the reported execution time.

### Profiles and system prompts

`/profile` shows the active profile. `/profile plain|agent` changes it and
resets the conversation. The model stays loaded. The agent toolbox follows
the selected profile.

The ready line shows the selected model, chat profile, thinking status,
routing profile, and resident memory use.

`/prompt` shows the active system prompt and its file. Each profile has a
separate file:

```text
~/.macqwen/system-prompt-plain.txt
~/.macqwen/system-prompt-agent.txt
```

Edit the active file with any text tool, then use `/reset`. `/prompt default`
removes the custom file and restores the built-in prompt. MACQWEN migrates a
legacy prompt from the preferences file on first use.

### Local API server

Start server mode directly:

```bash
./chat.sh /server
```

`./chat.sh --server` is an equivalent command. The `/server` chat command also
starts server mode. This command closes chat before it starts the server.

The default address is `http://127.0.0.1:8080`. Server mode provides these
text and tool-calling APIs:

| Client protocol | Endpoint |
|---|---|
| OpenAI Responses | `/v1/responses` |
| OpenAI Chat Completions | `/v1/chat/completions` |
| Anthropic Messages | `/v1/messages` |

The server handles one generation at a time. It resets the model when it
starts and when it stops, not on each request. The client must send the
required conversation history.

The server keeps its cache warm between requests. When the new conversation
extends the one the cache already holds, only the new tokens go through a
prefill. Measured with thinking off: a follow-up turn prefilled 18 tokens
instead of the whole conversation.

Reuse needs the client to send the previous assistant turn back exactly as the
model produced it. With thinking on, a client that drops the thinking text
sends a different token sequence, so the prompt diverges and the cache is
rebuilt. A diverging prompt always rebuilds, because the recurrent layers
carry state that cannot be rewound.

Terminal chat input stays disabled while the server runs.

Command line clients and SDKs work without changes. The server refuses a
browser page unless its origin is allowed. A page on any site can reach a server
bound to localhost:

```bash
./chat.sh --server --allow-origin http://localhost:3000
```

A non-local `--host` requires `MACQWEN_SERVER_API_KEY` or `--server-api-key`.
See SECURITY.md.

Use this Codex provider configuration in `~/.codex/config.toml`:

```toml
model = "macqwen-flashnext"
model_provider = "macqwen"

[model_providers.macqwen]
name = "MACQWEN"
base_url = "http://127.0.0.1:8080/v1"
env_key = "MACQWEN_API_KEY"
wire_api = "responses"
```

Then start Codex with a local placeholder key:

```bash
export MACQWEN_API_KEY=local
codex
```

Use these variables with Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=macqwen-flashnext
claude
```

Localhost mode accepts any client key. To bind another address, set a real
shared key:

```bash
export MACQWEN_SERVER_API_KEY="PRIVATE_VALUE"
./chat.sh /server --host 0.0.0.0
```

Do not expose this experimental server directly to the internet.

### Chat output

Chat displays completed words and phrases. It holds partial words until a
boundary arrives. Interactive terminals use a short fade for each completed
unit. API streaming does not add this terminal animation.

### Model settings

Run `/settings` to view the settings for the loaded model. Flash-Next supports
live routing and expert controls:

```text
/settings routing exact-quality
/settings routing cache-aware
/settings routing fused-quality
/settings swap-epsilon 0.02
/settings threshold 1.0
/settings resident-experts 32
/settings pinned-experts 32
/settings pin-budget-gb 6
/settings tail-experts 6
/settings tail-warmup 8
/settings fusion-block 23
/settings fusion-min-margin 1.0
/settings fusion-min-block 20
/settings fusion-margin-tokens 8
/settings fusion-max-prompt 512
/settings fusion-model <path to a draft model>
/settings defaults
```

`fusion-model` has no default. The `fused-quality` profile needs a draft
model that shares the target vocabulary. Download one and set its path before
selecting that profile.

`pinned-experts` is an alias for `resident-experts`. It controls the selected
experts pinned per layer in `exact-quality`, `cache-aware`, and `fused-quality`
modes.
`pin-budget-gb` limits their total pinned storage. `tail-experts` controls the
different pinned tail set used by `fast-quality`.

`swap-epsilon` controls the maximum score difference accepted by
`cache-aware`. Its default is `0.02`. Lower values change fewer expert choices.

`/settings` also reports the active pinned count, pinned memory, model path,
and session directory. The model path and session directory need a restart.

Changes apply on the next turn and stay inside the current process. A new
`./chat.sh` launch returns to the tuned `exact-quality` defaults.

Select `fused-quality` before the first turn. Use `/reset` before selecting it
for a new conversation. Use CLI flags when a custom mode must start immediately.

`speculative-fast` and MTP remain research-only. They require different model
loading and measured slower on the reference machine.

## API keys

Tavily requires an API key. Context7 accepts an optional key.

```text
/keys
/keys set tavily
/keys set context7
/keys delete tavily
```

Key input does not echo. MACQWEN stores managed keys in:

```text
~/Library/Application Support/MACQWEN/api_keys.json
```

The directory uses mode `0700`. The file uses mode `0600`.

## Persistent chat data

| Data | Default location |
|---|---|
| Preferences | `~/.macqwen/preferences.json` |
| API keys | `~/Library/Application Support/MACQWEN/api_keys.json` |
| Flash-Next sessions | `~/.cache/flashnext/sessions/` |
| Qwen27B sessions | `~/.frankenstein/sessions/` |

Session files can contain private prompt state. Do not publish them.

## Repository layout

```text
macqwen/                 shared session, profiles, commands, and tools
  backends/              model adapters loaded in their own environments
models/qwen27b/          Qwen3.8-27B runtime and research utilities
models/flashnext/        Flash-Next streaming runtime and benchmarks
docs/                    current briefs, research records, and handoffs
```

`chat.sh` selects the model, checkpoint, and interpreter before the runtime
loads.

## Validation

Run shared tests:

```bash
python3 -m unittest discover -s macqwen -p 'test_*.py'
```

Run Flash-Next tests:

```bash
.venv/bin/python -m unittest discover \
  -s models/flashnext -p 'test_*.py' -q
```

Model handoffs contain live validation commands.

## Documentation

The [documentation index](docs/README.md) provides three active documents for
the shared chat and each model:

- A brief defines purpose and support status.
- A research file preserves measurements and decisions.
- A handoff contains current operating and validation instructions.

Historical logs and superseded runbooks remain under `docs/archive/`.

## Research rules

- Measure the complete runtime path.
- Hold prompts and token limits constant.
- Compare generated token IDs before accepting an exact speed change.
- Use a quality gate when a routing mode intentionally changes token IDs.
- Record memory pressure and cache state.
- Preserve failed experiments, so later work does not repeat them.

## Security

Read [SECURITY.md](SECURITY.md) before exposing the web terminal or agent tools.
Do not commit credentials, model sessions, or private prompt data.

## License and third-party software

MACQWEN source code uses the MIT License. Models and dependencies use their own
licenses. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
