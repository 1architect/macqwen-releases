#!/bin/bash
cd /Users/gioma/Developer/MACQWEN
for arm in "0 20000" "1 60000" "1 150000" "1 400000" "1 20000" "0 20000"; do
  set -- $arm
  echo "=== keepwarm $1 iters $2"
  FLASHNEXT_GPU_KEEPWARM=$1 FLASHNEXT_GPU_KEEPWARM_ITERS=$2 MACQWEN_RESULTS_DIR=/Users/gioma/Developer/MACQWEN/results/flashnext/20260922-185916-hump-keepwarm-tuning ~/models/.venv-qwen4exp/bin/python -u -m models.flashnext.tests.bench.bench_read_ceiling --mode ram --miss 0.25 --tokens 30
  sleep 3
done
echo "=== sweep done"
