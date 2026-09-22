#!/bin/bash
cd /Users/gioma/Developer/MACQWEN
for m in 0.25 0 1.0 0.125 0.5; do
  echo "=== miss $m"
  MACQWEN_RESULTS_DIR=/Users/gioma/Developer/MACQWEN/results/flashnext/20260922-185137-hump-gpu-pstates ~/models/.venv-qwen4exp/bin/python -u -m models.flashnext.tests.bench.bench_read_ceiling --mode ram --miss $m --tokens 30
  sleep 3
done
echo "=== sweep done"
