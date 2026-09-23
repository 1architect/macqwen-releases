#!/bin/zsh
# Wired Metal memory (FLASHNEXT_WIRED_GB=3.5, set before load) vs none.
# Loaded phase: a helper holds 3 GB of active anonymous memory throughout.
# Order loaded: w0 w35 w35 w0; then quiet: w35 w0. 128 greedy tokens, 8 pins,
# chat defaults otherwise. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name  swap $(sysctl -n vm.swapusage | awk '{print $6}')" | tee -a results/flashnext/20260923-041256-wired-load/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json results/flashnext/20260923-041256-wired-load/arm-$n-$name.json 2>&1 | tee -a results/flashnext/20260923-041256-wired-load/output.log
}
W0=(FLASHNEXT_WIRED_GB=0)
W35=(FLASHNEXT_WIRED_GB=3.5)
$PY -m models.flashnext.tests.bench.memory_load --gb 3 >> results/flashnext/20260923-041256-wired-load/output.log 2>&1 &
LOAD=$!
sleep 15
arm 1 loaded-w0 $W0
arm 2 loaded-w35 $W35
arm 3 loaded-w35 $W35
arm 4 loaded-w0 $W0
kill $LOAD; wait $LOAD 2>/dev/null
sleep 15
arm 5 quiet-w35 $W35
arm 6 quiet-w0 $W0
echo "=== done" | tee -a results/flashnext/20260923-041256-wired-load/output.log
