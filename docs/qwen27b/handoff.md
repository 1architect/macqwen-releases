# Qwen3.8-27B reference

## Environment

The launcher checks the active environment, then `.venv-qwen27b`:

```text
.venv-qwen27b/bin/python
```

Override it with `MACQWEN_QWEN27B_PYTHON`.

## Model selection

Place V4 builds under:

```text
~/models/Qwen3.8-27B-Apple-MLX-V4-BUILD
```

Run a build by suffix:

```bash
./chat.sh BUILD --profile plain
./chat.sh BUILD --profile agent
```

The launcher also accepts `--model-path`. It rejects an incompatible vocabulary.

## Main files

| Path | Responsibility |
|---|---|
| `macqwen/backends/frankenstein.py` | Shared chat adapter |
| `models/qwen27b/frankenstein_engine.py` | Stateful model engine |
| `models/qwen27b/paged_kv.py` | Paged attention cache and SSD spill |
| `models/qwen27b/bf16_ends.py` | External embedding and shortlist head |
| `models/qwen27b/quantize_v4.py` | Score, plan, and build V4 checkpoints |
| `models/qwen27b/bit_allocator.py` | Activation calibration |
| `models/qwen27b/repo_context_image.py` | Exact repository context images |

## Memory rules

- Keep resident model memory near or below 12.6 GB on a 16 GB machine.
- Keep `PREFILL_STEP=256` for the measured low-memory path.
- Use `KV_BITS=4` and `KV_START=0` for the retained V4 configuration.
- Do not raise `MLX_QMM_BK` above `32` without a new kernel measurement.

## Validation

Run backend and parser tests in the 27B environment:

```bash
~/mlx-qwen38-kernel-lab/bin/python3 -m unittest \
  macqwen.test_qwen27b_backend macqwen.test_tools
```

Run the lightweight import check:

```bash
~/mlx-qwen38-kernel-lab/bin/python3 -c \
  'from models.qwen27b.frankenstein_engine import FrankensteinEngine'
```

A live test requires an installed compatible checkpoint.

## Closed directions

The research file contains the measurements. Do not retry these without a new mechanism:

- Dense-model FFN streaming.
- Stock-weight grafting into the retained V4 model.
- Low-rank selector heads.
- DWQ on a 16 GB machine.
- Larger quantized matrix kernel tiles.
- Lazy mapped full weights.

## Next work

Current work is tracked in the public issue tracker:

- [#12](https://github.com/1architect/macqwen-releases/issues/12) Install a compatible Qwen27B V4 checkpoint for live regression.
- [#13](https://github.com/1architect/macqwen-releases/issues/13) Compare pure-knapsack and bit-floor builds at equal size.
- [#14](https://github.com/1architect/macqwen-releases/issues/14) Calibrate bit allocation with output-loss sensitivity.
- [#15](https://github.com/1architect/macqwen-releases/issues/15) Validate sparse paged-attention gather performance and quality.
- [#16](https://github.com/1architect/macqwen-releases/issues/16) Run an interleaved fp16 versus Q4 decode benchmark.
- [#17](https://github.com/1architect/macqwen-releases/issues/17) Extend the SketchUp code guard checks.
