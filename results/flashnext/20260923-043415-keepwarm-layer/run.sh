#!/bin/zsh
# Whole-turn keep-warm (FLASHNEXT_GPU_KEEPWARM_MODE=layer) vs read-wait keep-warm,
# alone and with the two byte savers it may unlock. Quiet machine, fresh process
# per arm, 128 greedy tokens, 8 pins, chat defaults otherwise.
# Order: D L LK LS D LS LK L D. Expected digest: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
arm() {
  local n=$1 name=$2; shift 2
  echo "=== arm $n: $name" | tee -a results/flashnext/20260923-043415-keepwarm-layer/output.log
  env "$@" $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
    --conditions defaults --json results/flashnext/20260923-043415-keepwarm-layer/arm-$n-$name.json 2>&1 | tee -a results/flashnext/20260923-043415-keepwarm-layer/output.log
}
D=(FLASHNEXT_GPU_KEEPWARM_MODE=reads)
L=(FLASHNEXT_GPU_KEEPWARM_MODE=layer)
LK=(FLASHNEXT_GPU_KEEPWARM_MODE=layer FLASHNEXT_EXPERT_LOCK_GB=3)
LS=(FLASHNEXT_GPU_KEEPWARM_MODE=layer FLASHNEXT_SLAB_GLOBAL=120 FLASHNEXT_SLAB_NUM_LAYERS=20)
arm 1 defaults $D
arm 2 layer $L
arm 3 layer-lock3 $LK
arm 4 layer-slab120 $LS
arm 5 defaults $D
arm 6 layer-slab120 $LS
arm 7 layer-lock3 $LK
arm 8 layer $L
arm 9 defaults $D
echo "=== done" | tee -a results/flashnext/20260923-043415-keepwarm-layer/output.log
