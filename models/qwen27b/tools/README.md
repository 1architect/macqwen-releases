# Qwen3.8-27B offline tools

Research scripts that are neither runtime code nor tests: quantization
(`quantize_v4`, `bit_allocator`, `sensitivity`), quality comparisons
(`eval_models`, `bits_vs_quality`), FFN sparsity probes (`ffn_oracle_bound`,
`ffn_sparsity_probe`), the speculative-prefill prototype, and repository
context images. Run them as modules from the repository root, for example
`python -m models.qwen27b.tools.eval_models`. Anything they measure belongs in
`results/qwen27b/`; see [docs/testing.md](../../../docs/testing.md).
