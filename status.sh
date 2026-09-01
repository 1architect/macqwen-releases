#!/bin/bash
# Live status dashboard for model runs. Keep this window open.
D="$(cd "$(dirname "$0")" && pwd)"
while true; do
  clear
  printf '\033[1m=== FRANKENSTEIN STATUS ===\033[0m   %s\n\n' "$(date '+%H:%M:%S')"

  vm_stat | awk -v sw="$(sysctl -n vm.swapusage | awk '{print $6}')" '
    /Pages free/{gsub(/\./,"",$3); f=$3}
    /Pages inactive/{gsub(/\./,"",$3); i=$3}
    /Pages active/{gsub(/\./,"",$3); a=$3}
    END{printf "MEMORY  free %.2f GB   inactive %.2f GB   active %.2f GB   swap %s\n",
        f*16384/1e9, i*16384/1e9, a*16384/1e9, sw}'

  P=$(ps -A -o pid=,stat=,pcpu=,rss=,command= | grep -E "[p]aged_kv|[b]ench_decode|[f]rankenstein_engine|[m]lx_lm server" | head -3)
  if [ -n "$P" ]; then
    echo "$P" | awk '{printf "RUNNING pid %-6s %-4s cpu %5s%%  rss %.2f GB  %s\n", $1,$2,$3,$4/1048576, $6}'
    echo "        stat U = uninterruptible disk wait (swapping, bad sign)"
  else
    echo "RUNNING none"
  fi

  L=$(ls -t "$D"/*.log "$D"/*.stdout 2>/dev/null | head -1)
  if [ -n "$L" ]; then
    printf '\n\033[1mLOG\033[0m %s  (%s)\n' "$(basename "$L")" "$(date -r "$L" '+%H:%M:%S')"
    printf -- '------------------------------------------------------------------\n'
    tail -14 "$L" | cut -c1-150
  fi
  sleep 2
done
