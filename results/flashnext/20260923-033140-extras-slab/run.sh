#!/bin/zsh
# Remaining exact candidates vs current chat defaults, fresh process per arm.
# extras = 120-slot slab over 20 layers + FLASHNEXT_COMPILE + FLASHNEXT_PREWARM.
# Order: defaults, extras, extras, defaults. 128 greedy tokens, 8 pins.
# Expected digest in every arm: e19af44d5268e9d1.
cd /Users/gioma/Developer/MACQWEN
PY=~/models/.venv-qwen4exp/bin/python
EXTRAS=(FLASHNEXT_SLAB_GLOBAL=120 FLASHNEXT_SLAB_NUM_LAYERS=20 FLASHNEXT_COMPILE=1 FLASHNEXT_PREWARM=1)
arm() {
  echo "=== arm $1: $2" | tee -a results/flashnext/20260923-033140-extras-slab/output.log
  if [[ $2 == extras ]]; then
    env $EXTRAS $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
      --conditions defaults --json results/flashnext/20260923-033140-extras-slab/arm-$1-extras.json 2>&1 | tee -a results/flashnext/20260923-033140-extras-slab/output.log
  else
    $PY -m models.flashnext.tests.bench.bench_long_states --tokens 128 --window 64 \
      --conditions defaults --json results/flashnext/20260923-033140-extras-slab/arm-$1-defaults.json 2>&1 | tee -a results/flashnext/20260923-033140-extras-slab/output.log
  fi
}
arm 1 defaults
arm 2 extras
arm 3 extras
arm 4 defaults
echo "=== done" | tee -a results/flashnext/20260923-033140-extras-slab/output.log
