#!/bin/zsh
# 4 tok/s lever 1: read workers 8 (bundle default) vs 16 at P15, keep-warm on
# stream-pack waits. Fresh process per arm, 128 greedy tokens, 8 pins.
# Order: 8 16 16 8 8 16. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-115307-workers-p15
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a $OUT/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json $OUT/arm-$n-$name.json 2>&1 | grep -v PyTorch | tee -a $OUT/output.log
}
arm 1 w8 FLASHNEXT_IO_WORKERS=8
arm 2 w16 FLASHNEXT_IO_WORKERS=16
arm 3 w16 FLASHNEXT_IO_WORKERS=16
arm 4 w8 FLASHNEXT_IO_WORKERS=8
arm 5 w8 FLASHNEXT_IO_WORKERS=8
arm 6 w16 FLASHNEXT_IO_WORKERS=16
echo "=== done" | tee -a $OUT/output.log
