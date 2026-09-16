# K2-Horizon runtime

This package contains everything specific to K2-Horizon 7B:

- checkpoint discovery and the `k2` alias;
- Python environment and session defaults;
- checkpoint-owned MLX-LM model loading;
- K2 chat-template handling;
- reasoning and JSON tool-call protocol adaptation;
- session persistence and runtime tests.

The shared chat continues to consume its existing `<think>` and
`<tool_call>` contract. We translate K2's `<ifm|...>` protocol here, at the
model boundary, rather than teaching shared parsers about one model.

Run the model:

```bash
./chat.sh --model k2-horizon --checkpoint k2
```

Run its checkpoint-free tests:

```bash
.venv/bin/python -m unittest discover \
  -s models/k2_horizon -p 'test_*.py' -q
```

The checkpoint supplies executable `model.py` code. We only load a compatible
checkpoint from a source we trust.
