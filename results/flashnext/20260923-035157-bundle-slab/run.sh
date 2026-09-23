#!/bin/zsh
# Exact opt-in bundle (now the chat default) vs the pre-bundle settings,
# slab on in both. Fresh process per arm, 128 greedy tokens, 8 pins.
# Order: prior, bundle, bundle, prior, prior, bundle. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
PRIOR=(
  FLASHNEXT_STREAM_EMBED=0 FLASHNEXT_COMPILE_HC=0 FLASHNEXT_COMPILE_NORM=0
  FLASHNEXT_NORM_WEIGHT_CACHE=0 FLASHNEXT_IO_QOS=default FLASHNEXT_NGRAM_PARALLEL_MIN=0
  FLASHNEXT_QSA_CACHE_POOLED_KEYS=0 FLASHNEXT_QSA_SCATTER_DECODE=0 FLASHNEXT_OVERLAP=1
  FLASHNEXT_RDAHEAD=1 FLASHNEXT_IO_WORKERS=16 FLASHNEXT_STREAM_PACK=0 FLASHNEXT_STREAM_PACK_CHUNK=0
)
BUNDLE=(FLASHNEXT_UNUSED=0)
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a results/flashnext/20260923-035157-bundle-slab/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json results/flashnext/20260923-035157-bundle-slab/arm-$n-$name.json 2>&1 | tee -a results/flashnext/20260923-035157-bundle-slab/output.log
}
arm 1 prior $PRIOR
arm 2 bundle $BUNDLE
arm 3 bundle $BUNDLE
arm 4 prior $PRIOR
arm 5 prior $PRIOR
arm 6 bundle $BUNDLE
echo "=== done" | tee -a results/flashnext/20260923-035157-bundle-slab/output.log
