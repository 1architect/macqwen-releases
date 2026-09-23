#!/bin/zsh
# One candidate at a time vs chat defaults, fresh process per arm.
# D defaults, S 120-slot slab over 20 layers, C FLASHNEXT_COMPILE, P prewarm.
# Order: D S C P D P C S D. 128 greedy tokens, 8 pins. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a results/flashnext/20260923-034046-extras-split/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json results/flashnext/20260923-034046-extras-split/arm-$n-$name.json 2>&1 | tee -a results/flashnext/20260923-034046-extras-split/output.log
}
D=(FLASHNEXT_UNUSED=0)
S=(FLASHNEXT_SLAB_GLOBAL=120 FLASHNEXT_SLAB_NUM_LAYERS=20)
C=(FLASHNEXT_COMPILE=1)
P=(FLASHNEXT_PREWARM=1)
arm 1 defaults $D
arm 2 slab120 $S
arm 3 compile $C
arm 4 prewarm $P
arm 5 defaults $D
arm 6 prewarm $P
arm 7 compile $C
arm 8 slab120 $S
arm 9 defaults $D
echo "=== done" | tee -a results/flashnext/20260923-034046-extras-split/output.log
