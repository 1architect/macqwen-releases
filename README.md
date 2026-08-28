# MACQWEN Releases: FlashNext

FlashNext runs Qwen3.8-Flash-Next from SSD on Apple Silicon.

The checkpoint has 176 billion parameters and uses 111.7 GB on disk.
FlashNext keeps only 4.2 GB resident during text generation on the tested system.

It streams two large tensor families from SSD:

- Routed MoE experts.
- The n-gram embedding table.

## Requirements

- An Apple Silicon Mac.
- macOS with Metal support.
- Python 3.12.
- At least 120 GB of free storage.
- The `Vontra/Qwen3.8-Flash-Next-MLX-oQ4` checkpoint.

The tested system uses an M4 Mac with 16 GB of unified memory.

## Install

```bash
git clone https://github.com/1architect/macqwen-releases.git
cd macqwen-releases

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ4 \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

The repository does not contain model weights.

## Run

```bash
./flashnext/chat.sh
```

Use another checkpoint path when necessary:

```bash
FLASHNEXT_MODEL=/path/to/Qwen3.8-Flash-Next-MLX-oQ4 ./flashnext/chat.sh
```

You can also use the installed command:

```bash
flashnext-chat --model /path/to/Qwen3.8-Flash-Next-MLX-oQ4
```

## Generation profiles

```bash
./flashnext/chat.sh                      # threshold 0.85
./flashnext/chat.sh --threshold 1.0      # full shipped routing
./flashnext/chat.sh --exact-quality      # default trajectory with selective RAM
./flashnext/chat.sh --fast               # aggressive approximation
./flashnext/chat.sh --fast-quality       # approximate quality recovery
```

Threshold `0.85` is approximate relative to full routing.
Threshold `1.0` retains the shipped router selection.

`--exact-quality` preserves the default threshold `0.85` trajectory.
It does not change that trajectory into full routing.

## Chat commands

The chat prints this command list at startup:

```text
/thinking on|off
/thinking show|hide
/max-tokens N|off
/status
/salvar NAME
/carregar NAME
/sessoes
/apagar NAME
/reset
/sair
```

Replies wait for EOS by default.
Use `/max-tokens N` when you need a fixed reply limit.

`/thinking hide` hides reasoning text without disabling model reasoning.

## Persistent sessions

`/salvar NAME` stores the exact live model state.
`/carregar NAME` restores that state without repeating the old prefill.

Session files can contain conversation data.
FlashNext writes them with private file permissions.

The default session directory is `~/.cache/flashnext/sessions`.

## Measured performance

These measurements use the tested M4 Mac and the checkpoint above:

| Metric | Result |
|---|---:|
| Resident memory | 4.19 GB |
| Load time | 2.1 seconds |
| Decode, threshold 0.85 | 1.00 token/s |
| Prefill | 0.77 token/s |

SSD speed and macOS cache state affect these results.

## Optional MTP experiment

MTP is exact but slower on the tested system.

```bash
python flashnext/fetch_mtp.py \
  --output "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4/model-mtp.safetensors"

./flashnext/chat.sh --mtp-depth 1
```

## Tests

```bash
python -m unittest flashnext.test_sessions -v
```

The real-model session test is optional:

```bash
python -m flashnext.test_session_roundtrip_real \
  --model "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

## Security

FlashNext does not need API keys.
Do not commit Hugging Face tokens or local environment files.

Saved sessions can contain private prompt state.
Do not publish session files.

See [SECURITY.md](SECURITY.md) for vulnerability reports.

## License

The FlashNext source code uses the MIT License.
The model and dependencies use their own licenses.

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
