#!/bin/zsh
# Offline read replay of the recorded 256-token routes (first 128 decode tokens
# per turn, three turns), no model. Chat environment, 3.4 GB anonymous ballast
# for the model's memory, 2.6 ms sleep per layer for the GPU phase, expert pages
# evicted from the file cache before each run. Production read path against an
# application-owned LRU pool with direct fills. Order: P D4 D3 D3 D4 P.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-131822-read-replay
run() {
  local n=$1 name=$2; shift 2
  echo "=== run $n: $name" | tee -a $OUT/output.log
  $PY -m models.flashnext.tests.bench.bench_read_replay --evict --tokens 128 "$@" \
    --json $OUT/run-$n-$name.json 2>&1 | grep -v PyTorch | tee -a $OUT/output.log
}
run 1 production --engine production
run 2 pool4 --engine pool-direct --pool-gb 4
run 3 pool3 --engine pool-direct --pool-gb 3
run 4 pool3 --engine pool-direct --pool-gb 3
run 5 pool4 --engine pool-direct --pool-gb 4
run 6 production --engine production
echo "=== done" | tee -a $OUT/output.log
