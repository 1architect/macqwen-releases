#!/bin/zsh
# Keep-warm on stream-pack waits (fix) vs the previous behaviour, where the
# 12 slab-pack layers waited for their cold reads without keep-warm.
# Chat defaults otherwise (slab on, bundle on, keep-warm on), 8 pins.
# Fresh process per arm, 128 greedy tokens. Order: C F F C C F.
# Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-112333-keepwarm-stream-pack
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a $OUT/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json $OUT/arm-$n-$name.json 2>&1 | tee -a $OUT/output.log
}
C=(FLASHNEXT_GPU_KEEPWARM_STREAM_PACK=0)
F=(FLASHNEXT_GPU_KEEPWARM_STREAM_PACK=1)
arm 1 control $C
arm 2 fix $F
arm 3 fix $F
arm 4 control $C
arm 5 control $C
arm 6 fix $F
echo "=== done" | tee -a $OUT/output.log
