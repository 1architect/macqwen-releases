#!/bin/zsh
# Exact opt-in bundle vs chat defaults, fresh process per arm, 2026-09-23.
# Order: baseline, bundle, bundle, baseline. 128 greedy tokens, 8 pins,
# keep-warm on in both. Expected digest in every arm: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
BUNDLE=(
  FLASHNEXT_STREAM_EMBED=1
  FLASHNEXT_COMPILE_HC=1 FLASHNEXT_COMPILE_NORM=1 FLASHNEXT_NORM_WEIGHT_CACHE=1
  FLASHNEXT_IO_QOS=user-interactive
  FLASHNEXT_NGRAM_PARALLEL_MIN=64
  FLASHNEXT_QSA_CACHE_POOLED_KEYS=1 FLASHNEXT_QSA_SCATTER_DECODE=1
  FLASHNEXT_OVERLAP=0
  FLASHNEXT_RDAHEAD=0
  FLASHNEXT_IO_WORKERS=8
  FLASHNEXT_STREAM_PACK=1 FLASHNEXT_STREAM_PACK_CHUNK=2
)
arm() {
  echo "=== arm $1: $2" | tee -a results/flashnext/20260923-030140-exact-bundle/output.log
  if [[ $2 == bundle ]]; then
    env $BUNDLE $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
      --conditions bundle --json results/flashnext/20260923-030140-exact-bundle/arm-$1-bundle.json 2>&1 | tee -a results/flashnext/20260923-030140-exact-bundle/output.log
  else
    $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
      --conditions keepwarm --json results/flashnext/20260923-030140-exact-bundle/arm-$1-baseline.json 2>&1 | tee -a results/flashnext/20260923-030140-exact-bundle/output.log
  fi
}
arm 1 baseline
arm 2 bundle
arm 3 bundle
arm 4 baseline
echo "=== done" | tee -a results/flashnext/20260923-030140-exact-bundle/output.log
