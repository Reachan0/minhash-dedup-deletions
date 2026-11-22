#!/usr/bin/env bash
# AutoDL: B1 rungs 0.5M -> 1M -> 2M, then the F estimators on each (5M waits for exactjoin64).
set -uo pipefail
W=/path/to/scratch/dedupjoin
cd $W
export OMP_NUM_THREADS=12
for N in 531597 1000000 2000000; do
  echo "[$(date -u +%T)] rung $N start"
  env/bin/python tools/b1_rung.py rung $N > logs/rung_$N.log 2>&1 || { echo "RUNG_${N}_FAILED"; tail -5 logs/rung_$N.log; exit 1; }
  echo "[$(date -u +%T)] rung $N done"
  env/bin/python tools/f_estimators.py rung_$N > logs/F_rung_$N.log 2>&1 || echo "F_${N}_FAILED"
  echo "[$(date -u +%T)] F $N: $(tail -1 logs/F_rung_$N.log)"
done
echo "RUNGS_CHAIN_DONE"
