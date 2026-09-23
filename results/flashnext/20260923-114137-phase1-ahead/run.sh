#!/bin/zsh
# Phase 1: perfect-prediction look-ahead at P15. Synthetic routes, 3 of 8
# experts cold per layer, chat defaults (keep-warm on). --ahead reads the next
# layer's route one layer early on a separate pool; same routes and bytes.
# Order: off ahead ahead off off ahead. Fresh process each, 60 tokens.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
OUT=results/flashnext/20260923-114137-phase1-ahead
CHAT=($($PY -c "from models.flashnext.settings.launch import CHAT_ENV; print(' '.join(f'{k}={v}' for k,v in CHAT_ENV.items()))"))
arm() {
  echo "=== arm $1: $2" | tee -a $OUT/output.log
  shift 2
  env $CHAT $PY -m models.flashnext.tests.bench.bench_read_ceiling --mode ram --miss 0.375 --tokens 60 "$@" 2>&1 | grep -v "^prof\|PyTorch" | tee -a $OUT/output.log
}
arm 1 off
arm 2 ahead --ahead
arm 3 ahead --ahead
arm 4 off
arm 5 off
arm 6 ahead --ahead
echo "=== done" | tee -a $OUT/output.log
