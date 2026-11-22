#!/usr/bin/env bash
# AutoDL B1 chain: wait for env -> per-segment extract + shingle (14 workers, overlaps downloads).
set -uo pipefail
W=/path/to/scratch/dedupjoin
cd $W
while ! grep -q '^SETUP_DONE' logs/setup_env.log; do
  pgrep -f b1_setup_env.sh >/dev/null || { echo "ENV_SETUP_FAILED"; exit 1; }
  sleep 20
done
echo "[$(date -u +%T)] env ready; segments start"
env/bin/python tools/b1_segment_worker.py 14 > logs/segment_worker.log 2>&1
echo "[$(date -u +%T)] $(tail -1 logs/segment_worker.log)"
