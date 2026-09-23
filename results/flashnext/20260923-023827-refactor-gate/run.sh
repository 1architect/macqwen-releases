#!/bin/zsh
# Refactor digest gate plus shared-expert overlap pairs, 2026-09-23.
# Expected digest (recorded 20260922-230354, 128 tokens, 8 pins): e19af44d5268e9d1
cd /Users/gioma/Developer/MACQWEN
~/models/.venv-qwen4exp/bin/python -m models.flashnext.tests.bench.bench_long_states \
  --tokens 128 --window 64 \
  --conditions keepwarm keepwarm-nooverlap keepwarm-nooverlap keepwarm keepwarm keepwarm-nooverlap \
  --json results/flashnext/20260923-023827-refactor-gate/long-states.json 2>&1 | tee results/flashnext/20260923-023827-refactor-gate/output.log
