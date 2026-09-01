# Restructure plan

Written 2026-08-31. One chat, two profiles, model-exclusive work in subfolders.

## Background

The repository began as a Qwen3.8-27B runtime. That model grew a chat with
agentic tools. `flashnext/` then appeared as a folder inside it for a different
model and grew a second chat, which added features the first lacks and lacks
features the first has. Neither is a superset. A fix to one does not reach the
other, which is the same failure mode as issue #1 at repository scale.

## The binding constraint

The two models need different Python environments:

| Runtime | MLX | mlx-lm | mlx-vlm | transformers |
|---|---|---|---|---|
| `~/mlx-qwen38-kernel-lab` | 0.32.1.dev+3a62199 | 0.32.0 | none | 5.15.1 |
| `~/models/.venv-qwen4exp` | 0.32.2 | 0.31.3 | 0.6.17 | 5.16.1 |

The 27B runtime uses a custom MLX build. Merging the environments risks that
work, so **one process cannot serve both models**. "One chat interface" means
one shared codebase with a per-model launcher, not one interpreter.

Consequence: the shared package must import cleanly in both environments. It
stays pure Python. Anything touching MLX lives in a backend, imported only by
the launcher that owns it.

## Target layout

```text
macqwen/                 the chat, shared by every model
  cli.py                 parse args, resolve model and profile, exec the venv
  session.py             the loop: prompt, generate, stream, commands
  terminal.py            bracketed paste and line editing (today flashnext/)
  preferences.py         one preferences file, one schema
  commands.py            one command table, aliases included
  profiles/
    agent.py             large system prompt, tools, approval, budget
    plain.py             minimal prompt, no tools
  tools/                 api_guard, code_check, context7, search, repo cache
  backends/
    base.py              the interface a model runtime implements
    frankenstein.py      Qwen3.8-27B V4
    flashnext.py         Qwen3.8-Flash-Next
models/
  qwen27b/               bit_allocator, quantize_v4, sensitivity, bf16_ends,
                         paged_kv, speculative_prefill, eval and probe scripts
  flashnext/             loader, store, expert_cache, adaptive_topk, sessions,
                         qsa_chunk, ngram, speculative, benches
docs/
```

Profile and model are orthogonal. `--profile agent` selects prompt and tools;
the model selects the backend. Either profile can run on either model.

## Backend interface

Every runtime implements the same small surface, so `session.py` never learns
which model it is driving:

```python
class Backend(Protocol):
    def load(self, path, options): ...
    def prefill(self, ids): ...          # returns the first token
    def step(self, token): ...           # returns the next token
    def make_cache(self): ...
    def save_session(self, name): ...
    def load_session(self, name): ...
    def stats(self) -> dict: ...         # rate, memory, profile-specific counters
```

Session persistence differs by model: the 27B uses prompt-cache images, and
Flash-Next uses exact safetensors snapshots. Both satisfy save and load, so the
difference stays inside the backend.

## Sequencing

Each step is verifiable on its own and leaves both chats working.

1. Create `macqwen/` and copy the shared pieces into it. Change nothing else.
   Both chats keep running from their current files.
2. Point both chats at `macqwen/` for the copied pieces. Delete the duplicates.
   Verify: both still start, and their token output is unchanged.
3. Build `macqwen/session.py` from the union of the two command tables, with
   the feature gaps closed in both directions.
4. Add the two profiles and model selection. `./chat.sh --model flashnext
   --profile plain` and so on.
5. Retire `frankenstein_chat.py` and `flashnext/chat.py`.
6. Move model-exclusive files into `models/`.

## Feature gaps to close in step 3

Into the agent chat, from flashnext: `/thinking on|off|show|hide`,
`/max-tokens`, the routing profiles, exact sessions, `--benchmark-json`.

Into the plain chat, from the agent: `/effort`, `/stream`, `/prompt`,
workspace awareness.

Kept apart on purpose: tools and approval belong to the agent profile only.

## Rules this restructure exists to enforce

- A setting is read from one place. See issue #1: four settings were defined
  twice and silently discarded tonight.
- A feature added to the chat reaches every model, because there is one chat.
- Model-exclusive code cannot be imported by the shared layer.
