#!/usr/bin/env bash
# AutoDL, after b1_rungs_chain.sh: 3M rung (64-bit exact join; 5M would exceed the 100 GiB quota) -> F on 5M. (B2 runs on server.)
set -uo pipefail
W=/path/to/scratch/dedupjoin
cd $W
export OMP_NUM_THREADS=12
until grep -q 'RUNGS_CHAIN_DONE' logs/rungs_chain.log; do sleep 60; done
until grep -q B1_SEGMENTS_DONE logs/segment_worker.log; do sleep 60; done
echo "[$(date -u +%T)] rung 3M"; EJ64=1 env/bin/python tools/b1_rung.py rung 3000000 > logs/rung_3000000.log 2>&1 || echo RUNG_3M_FAILED
echo "[$(date -u +%T)] F 3M"; env/bin/python tools/f_estimators.py rung_3000000 > logs/F_rung_3000000.log 2>&1 || echo F_3M_FAILED
echo "CHAIN2_DONE"
