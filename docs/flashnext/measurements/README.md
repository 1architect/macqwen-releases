# Published measurement records

A performance number in `README.md`, `CHANGELOG.md`, or an overview must link to a file in this directory. `logs/` contains untracked
scratch data and cannot support a published result.

Produce a file with the standard harness:

```bash
python models/flashnext/bench_production.py --json docs/flashnext/measurements/NAME.json
```

Each file records every arm, physical MB per token, free memory, elapsed time, and rate correlation. Quote the median and range. Do not
quote the best arm or a two-arm mean. The older warm two-arm subset produced
an invalid 2.83 tok/s baseline. The current accepted 2.83 result comes from
the 12-arm clean-boot `buffer-chunk2` comparison.

## Retained files

| File | Comparison |
|---|---|
| `sort-reads.json` | Sorted expert reads against current order |
| `pin-parts.json` | Whole experts against scales and biases |
| `prewarm.json` | Session expert prewarm against no prewarm |
| `stacked.json` | Scales-only pinning plus prewarm |
| `track-resident.json` | Tracked resident reads against pinned-only reads |
| `swap-resident.json` | Exact routing against cache-aware routing |

`swap-resident.json` supports the cache-aware claim. Its medians are 2.539 and 2.790 tok/s. Physical reads are 417.8 and 347.6 MB per token.
The arms alternate. Pairing all eight adjacent condition arms gives a mean 8.3 percent gain, with seven pairs faster.
