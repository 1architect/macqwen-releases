#!/bin/zsh
# 12-layer small-row sidecar vs checkpoint reads on the stream-pack layers.
# Chat defaults, fresh process per arm, bench_split_p15 (unprofiled 128 tokens
# then profiled 128 tokens). Order: off side side off off side.
# Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-120412-small-sidecar
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a $OUT/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_split_p15 --tokens 128 \
    --json $OUT/arm-$n-$name.json 2>&1 | grep -v PyTorch | tee -a $OUT/output.log
}
OFF=(FLASHNEXT_SMALL_SIDECAR=)
SIDE=(FLASHNEXT_SMALL_SIDECAR=/Users/gioma/.cache/flashnext/small-sidecar-9b74bb32b36fc9ab)
arm 1 off $OFF
arm 2 side $SIDE
arm 3 side $SIDE
arm 4 off $OFF
arm 5 off $OFF
arm 6 side $SIDE
echo "=== done" | tee -a $OUT/output.log
