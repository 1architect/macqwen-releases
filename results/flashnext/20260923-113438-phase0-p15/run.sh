#!/bin/zsh
# Phase 0 of the 5 tok/s path: zero-read floor and a decode split at P15,
# chat defaults (slab, bundle, keep-warm incl. stream-pack waits).
# Order: floor, split, floor. Fresh process each.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-113438-phase0-p15
CHAT=($($PY -c "from models.flashnext.settings.launch import CHAT_ENV; print(' '.join(f'{k}={v}' for k,v in CHAT_ENV.items()))"))
echo "=== 1 floor pool32" | tee -a $OUT/output.log
env $CHAT $PY -m models.flashnext.tests.bench.bench_read_ceiling --mode ram --miss 0 --pool 32 --tokens 60 2>&1 | tee -a $OUT/output.log
echo "=== 2 split p15" | tee -a $OUT/output.log
$PY -m models.flashnext.tests.bench.bench_split_p15 --tokens 128 --json $OUT/split-p15.json 2>&1 | tee -a $OUT/output.log
echo "=== 3 floor pool32" | tee -a $OUT/output.log
env $CHAT $PY -m models.flashnext.tests.bench.bench_read_ceiling --mode ram --miss 0 --pool 32 --tokens 60 2>&1 | tee -a $OUT/output.log
echo "=== done" | tee -a $OUT/output.log
