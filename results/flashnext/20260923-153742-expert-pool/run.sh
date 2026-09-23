#!/bin/zsh
# Expert pool (FLASHNEXT_EXPERT_POOL_GB=6, pins off with it) against the chat
# defaults. Fresh process per arm, 128 greedy tokens. Order: D P P D D P.
# Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-153742-expert-pool
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name  swap $(sysctl -n vm.swapusage | awk '{print $6}')" | tee -a $OUT/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json $OUT/arm-$n-$name.json 2>&1 | grep -v PyTorch | tee -a $OUT/output.log
}
D=(FLASHNEXT_EXPERT_POOL_GB=0)
P=(FLASHNEXT_EXPERT_POOL_GB=6)
arm 1 defaults $D
arm 2 pool6 $P
arm 3 pool6 $P
arm 4 defaults $D
arm 5 defaults $D
arm 6 pool6 $P
echo "=== done  swap $(sysctl -n vm.swapusage | awk '{print $6}')" | tee -a $OUT/output.log
