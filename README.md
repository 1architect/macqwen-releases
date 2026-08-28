# MACQWEN Releases: FlashNext

FlashNext runs Qwen3.8-Flash-Next from SSD on Apple Silicon.

**Current `--exact-quality` result: about 1.9 tok/s prefill and 1.9 tok/s decode.**

The checkpoint has 176 billion parameters and uses 111.7 GB on disk.
FlashNext keeps only 4.2 GB resident during text generation on the tested system.

It streams two large tensor families from SSD:

- Routed MoE experts.
- The n-gram embedding table.

## How FlashNext makes the model fit

The standard MLX loader creates every model tensor in unified memory.
This checkpoint needs much more memory than a 16 GB Mac provides.

FlashNext replaces that load path before MLX materializes the large tensors.
It keeps the dense model core resident and leaves the large sparse tensors on SSD.

### Expert streaming

The model has 512 experts in each routed MoE layer.
Only a small expert set processes each token.

FlashNext reads only those expert rows from the original safetensors shards.
It reads the packed weights, scales, and biases for each selected expert.

Sixteen parallel readers keep the NVMe queue active.
Large prefill gathers use bounded chunks to avoid oversized temporary buffers.

The GPU receives one contiguous expert tensor for the current layer.
FlashNext releases that temporary tensor after the layer completes.

### N-gram streaming

The quantized n-gram table uses about 19.2 GB.
Each token needs only a small set of hashed rows.

FlashNext reads those rows directly and dequantizes them on demand.
It never loads the complete n-gram table.

### Adaptive routing

The router normally selects ten experts.
Some selected experts contribute very little probability mass.

The default threshold keeps experts until their cumulative router mass reaches `0.85`.
This reduces SSD traffic while retaining the default FlashNext trajectory.

Threshold `1.0` retains all experts selected by the shipped router.

### Selective RAM in `--exact-quality`

The profile watches the experts used during the first eight generated tokens.
It identifies recurring expert rows for each layer.

FlashNext then pins those file-backed pages with `mlock`.
It does not create another model copy.

The profile can pin about 1.24 GB of useful expert pages.
Later tokens reuse these pages at RAM speed instead of SSD speed.

Routing and logits stay identical to the default threshold `0.85` trajectory.
The profile changes storage placement only.

Selective placement reduces repeated expert I/O during longer replies.

The measured 1.9 tok/s prefill also uses parallel chunked reads and a warm file cache.
The measured 1.9 tok/s decode includes selective pinned pages after warmup.

### Exact session restore

FlashNext stores every recurrent state and attention cache in one safetensors file.
It also stores QSA index data, position state, and consumed token IDs.

Loading a session restores the cache directly.
The model processes only the new turn.

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
```

## Download the model

FlashNext uses this exact Hugging Face checkpoint:

```text
Vontra/Qwen3.8-Flash-Next-MLX-oQ4
```

This is an MLX checkpoint for the `qwen4_exp` architecture.
It contains 176 billion parameters in an optimized Q4 format.

The complete download uses about 111.7 GB.
It contains 22 safetensors shards, the tokenizer, and the model configuration.

The `huggingface-hub` dependency provides the `hf` command.
Run this command after activating the virtual environment:

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ4 \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

The Hugging Face repository is public.
This download does not require an API key.

The Hub client resumes interrupted downloads.
Do not delete partial files while the command runs.

Verify the local checkpoint:

```bash
find "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4" \
  -name 'model-*-of-*.safetensors' | wc -l

du -sh "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

The first command must report `22`.
The second command must report approximately `112G`.

The default FlashNext path matches this download directory.
No extra model argument is necessary after this command.

Other conversions can use different tensor layouts.
FlashNext currently supports only this tested checkpoint layout.

This source repository does not contain model weights.

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
./flashnext/chat.sh --exact-quality      # about 1.9 tok/s prefill and decode
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
| `--exact-quality` prefill | about 1.9 tok/s |
| `--exact-quality` decode | about 1.9 tok/s |
| Baseline resident memory | 4.19 GB |
| Selectively pinned expert pages | about 1.24 GB |
| Load time | 2.1 seconds |

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
