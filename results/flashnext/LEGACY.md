# Flash-Next measurement records

This directory contains retained Flash-Next evidence. New live runs use the
project terminal:

```bash
./tests/run.sh --model flashnext --checkpoint PATH
```

Follow the [measurement standard](../../docs/measurement-standard.md). Older JSON
files below were produced by the historical Flash-Next benchmark harness; keep
their commands and schema for provenance, but do not use them as a template
for new records.

A performance number in `README.md`, `CHANGELOG.md`, or an overview must link
to a retained record with clear checkpoint, harness, controls, and raw-arm
provenance. Do not quote a best arm, a synthetic fixed-route ceiling, or an
incomplete run as a product rate.

## Retained files

| File | Comparison |
|---|---|
| `sort-reads.json` | Sorted expert reads against current order |
| `pin-parts.json` | Whole experts against scales and biases |
| `prewarm.json` | Session expert prewarm against no prewarm |
| `stacked.json` | Scales-only pinning plus prewarm |
| `track-resident.json` | Tracked resident reads against pinned-only reads |
| `swap-resident.json` | Exact routing against cache-aware routing |

`swap-resident.json` supports the historical cache-aware claim. Its medians are
2.539 and 2.790 tok/s, with 417.8 and 347.6 MB of physical reads per token.
The arms alternate; the paired eight-arm subset measured an 8.3% mean gain.
See the Flash-Next brief and research record for scope and limitations.
