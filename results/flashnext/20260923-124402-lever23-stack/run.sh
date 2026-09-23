#!/bin/zsh
# Levers 2 and 3 stacked: one sync per decode layer, decode n-gram rows on the
# read pool, compiled GDN q/k normalization and gated norm (C) against chat
# defaults (D). Fresh process per arm, 128 greedy tokens, 8 pins.
# Order: D C C D D C. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-124402-lever23-stack
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a $OUT/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json $OUT/arm-$n-$name.json 2>&1 | grep -v PyTorch | tee -a $OUT/output.log
}
D=(FLASHNEXT_ONE_SYNC=0)
C=(FLASHNEXT_ONE_SYNC=1 FLASHNEXT_NGRAM_PARALLEL_MIN=16 FLASHNEXT_COMPILE_GDN=1)
arm 1 defaults $D
arm 2 stack $C
arm 3 stack $C
arm 4 defaults $D
arm 5 defaults $D
arm 6 stack $C
echo "=== done" | tee -a $OUT/output.log
