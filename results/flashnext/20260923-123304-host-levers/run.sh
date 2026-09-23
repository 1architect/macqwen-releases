#!/bin/zsh
# Lever 2 (host): one sync per decode layer (S, FLASHNEXT_ONE_SYNC=1) and
# decode n-gram rows on the read pool (N, FLASHNEXT_NGRAM_PARALLEL_MIN=16)
# against chat defaults (D). Fresh process per arm, 128 greedy tokens, 8 pins.
# Order: D S N N S D D S N. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-123304-host-levers
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a $OUT/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json $OUT/arm-$n-$name.json 2>&1 | grep -v PyTorch | tee -a $OUT/output.log
}
D=(FLASHNEXT_ONE_SYNC=0)
S=(FLASHNEXT_ONE_SYNC=1)
N=(FLASHNEXT_NGRAM_PARALLEL_MIN=16)
arm 1 defaults $D
arm 2 onesync $S
arm 3 ngram16 $N
arm 4 ngram16 $N
arm 5 onesync $S
arm 6 defaults $D
arm 7 defaults $D
arm 8 onesync $S
arm 9 ngram16 $N
echo "=== done" | tee -a $OUT/output.log
