#!/bin/bash
cd /Users/gioma/Developer/MACQWEN
for arm in "0 0" "1 40000" "1 60000" "1 80000" "1 60000" "0 0"; do
  set -- $arm
  echo "=== keepwarm $1 iters $2"
  FLASHNEXT_GPU_KEEPWARM=$1 FLASHNEXT_GPU_KEEPWARM_ITERS=$2 MACQWEN_RESULTS_DIR=/Users/gioma/Developer/MACQWEN/results/flashnext/20260922-190129-hump-keepwarm-fine ~/models/.venv-qwen4exp/bin/python -u -m models.flashnext.tests.bench.bench_read_ceiling --mode ram --miss 0.25 --tokens 30
  sleep 3
done
echo "=== sweep done"
