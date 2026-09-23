#!/bin/zsh
# Locked expert working set (FLASHNEXT_EXPERT_LOCK_GB=3) vs none.
# Loaded phase: a helper holds 3 GB of active anonymous memory throughout.
# Order loaded: lock0 lock3 lock3 lock0; then quiet: lock3 lock0. 128 greedy tokens, 8 pins,
# chat defaults otherwise. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name  swap $(sysctl -n vm.swapusage | awk '{print $6}')" | tee -a results/flashnext/20260923-042253-expert-lock/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json results/flashnext/20260923-042253-expert-lock/arm-$n-$name.json 2>&1 | tee -a results/flashnext/20260923-042253-expert-lock/output.log
}
L0=(FLASHNEXT_EXPERT_LOCK_GB=0)
L3=(FLASHNEXT_EXPERT_LOCK_GB=3)
$PY -m models.flashnext.tests.bench.memory_load --gb 3 >> results/flashnext/20260923-042253-expert-lock/output.log 2>&1 &
LOAD=$!
sleep 15
arm 1 loaded-lock0 $L0
arm 2 loaded-lock3 $L3
arm 3 loaded-lock3 $L3
arm 4 loaded-lock0 $L0
kill $LOAD; wait $LOAD 2>/dev/null
sleep 15
arm 5 quiet-lock3 $L3
arm 6 quiet-lock0 $L0
echo "=== done" | tee -a results/flashnext/20260923-042253-expert-lock/output.log
