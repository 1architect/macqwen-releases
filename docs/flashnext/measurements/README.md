# Measurements behind published numbers

A performance number in `README.md`, `CHANGELOG.md` or any brief must be
traceable to a file in this directory. `logs/` is scratch and is not tracked,
so a claim citing it cannot be checked by anyone reading the repository.

Produce a file with the standard harness:

```bash
python models/flashnext/bench_production.py --json docs/flashnext/measurements/NAME.json
```

Each file records every arm, its physical MB per token, free memory, seconds
since the run began, and the correlation between rate and elapsed time. Quote
the median and the range. Never quote a mean of two arms, and never quote the
best arm: a two-arm subset of a warmup sweep published as a 2.83 tok/s
production baseline is what made this directory necessary.

## Retained files

| File | Comparison |
|---|---|
| `sort-reads.json` | Sorted expert reads against current order |
| `pin-parts.json` | Whole experts against scales and biases |
| `prewarm.json` | Session expert prewarm against no prewarm |
| `stacked.json` | Scales-only pinning plus prewarm |
| `track-resident.json` | Tracked resident reads against pinned-only reads |
| `swap-resident.json` | Exact routing against cache-aware routing |

`swap-resident.json` supports the cache-aware claim. Its medians are 2.539 and
2.790 tok/s. Physical reads are 417.8 and 347.6 MB per token. The arms
alternate. Pairing all eight adjacent condition arms gives a mean 8.3 percent
gain, with seven pairs faster.
