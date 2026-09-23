#!/bin/zsh
# Offline read replay of the recorded 256-token routes (first 128 decode tokens
# per turn, three turns), no model. Chat environment, 3.4 GB anonymous ballast
# for the model's memory, 2.6 ms sleep per layer for the GPU phase, expert pages
# evicted from the file cache before each run. Production read path against an
# the same destinations read by one native call per layer. Order: P N N P.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-133612-read-replay-native
run() {
  local n=$1 name=$2; shift 2
  echo "=== run $n: $name" | tee -a $OUT/output.log
  $PY -m models.flashnext.tests.bench.bench_read_replay --evict --tokens 128 "$@" \
    --json $OUT/run-$n-$name.json 2>&1 | grep -v PyTorch | tee -a $OUT/output.log
}
run 1 production --engine production
run 2 native --engine production-native
run 3 native --engine production-native
run 4 production --engine production
echo "=== done" | tee -a $OUT/output.log
