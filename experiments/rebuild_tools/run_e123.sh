#!/usr/bin/env bash
# Run E1 -> E2 -> E3 (post-reveal drivers) sequentially on server.
set -euo pipefail
W=/path/to/home/dedupjoin-postreveal
D=$W/dedupjoin-newsnap-selection/code/post_reveal
export OMP_NUM_THREADS=12
cd "$W"
echo "[$(date -u +%T)] E1 start"; env/bin/python -B "$D/e1_outside_group.py"; echo "[$(date -u +%T)] E1_DONE"
echo "[$(date -u +%T)] E2 start"; bash "$D/e2_run.sh"; echo "[$(date -u +%T)] E2_DONE"
echo "[$(date -u +%T)] E3 start"; env/bin/python -B "$D/e3_filter50.py"; echo "[$(date -u +%T)] E3_DONE"
