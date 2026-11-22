#!/usr/bin/env bash
# AutoDL, after b1_rungs_chain.sh: 5M rung (64-bit exact join) -> F on 5M. (B2 runs on server.)
set -uo pipefail
W=/path/to/scratch/dedupjoin
cd $W
export OMP_NUM_THREADS=12
until grep -q 'RUNGS_CHAIN_DONE' logs/rungs_chain.log; do sleep 60; done
until grep -q B1_SEGMENTS_DONE logs/segment_worker.log; do sleep 60; done
echo "[$(date -u +%T)] rung 5M"; EJ64=1 env/bin/python tools/b1_rung.py rung 5000000 > logs/rung_5000000.log 2>&1 || echo RUNG_5M_FAILED
echo "[$(date -u +%T)] F 5M"; env/bin/python tools/f_estimators.py rung_5000000 > logs/F_rung_5000000.log 2>&1 || echo F_5M_FAILED
echo "CHAIN2_DONE"
