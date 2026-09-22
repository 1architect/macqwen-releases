#!/bin/bash
cd /Users/gioma/Developer/MACQWEN
for arm in "0.25 0" "0.25 1" "0.25 1" "0.25 0" "0.5 1" "0.5 0"; do
  set -- $arm
  echo "=== miss $1 keepwarm $2"
  FLASHNEXT_GPU_KEEPWARM=$2 MACQWEN_RESULTS_DIR=/Users/gioma/Developer/MACQWEN/results/flashnext/20260922-185656-hump-keepwarm-sweep ~/models/.venv-qwen4exp/bin/python -u -m models.flashnext.tests.bench.bench_read_ceiling --mode ram --miss $1 --tokens 30
  sleep 3
done
echo "=== sweep done"
