#!/bin/bash
# Terminal chat with cache-aware routing on.
#
# The router scores ten experts per layer and keeps about eight. When a kept
# expert has to come off the drive and a discarded one is already in memory
# within FLASHNEXT_SWAP_EPSILON of its score, the resident one is taken. That
# removes a physical read and changes the reply slightly.
#
# Measured against exact routing: 16.8 percent fewer physical reads, and
# +8.3 percent decode paired arm by arm, seven of eight pairs ahead.
# The quality gate lost no answer that exact routing reached, but it is only
# four prompts wide. Report anything that reads wrong.
#
# The ready line shows `routing=cache-aware` when it is active.
cd /Users/gioma/Developer/MACQWEN
exec ./chat.sh --model flashnext --profile "${1:-plain}" --cache-aware \
  --swap-epsilon "${FLASHNEXT_SWAP_EPSILON:-0.02}" "${@:2}"
