#!/usr/bin/env bash
# server: B2 on the filtered 2025 snapshot -> cost runs -> quality runs.
set -uo pipefail
G=/path/to/home/dedupjoin-postreveal
cd $G
export B2_W=$G OMP_NUM_THREADS=12
T=dedupjoin-pvldb-v4/experiments/b1_tools
until grep -q W_SNAPSHOT_DONE logs/w_snapshot.log; do sleep 60; done
echo "[$(date -u +%T)] inject"; env/bin/python $T/b2_inject_snapshot.py > logs/b2_inject.log 2>&1 || { echo B2_INJECT_FAILED; exit 1; }
echo "[$(date -u +%T)] cost runs"; env/bin/python $T/b2_runs.py base exact_1000 exact_10000 exact_100000 near_1000 near_10000 near_100000 > logs/b2_runs.log 2>&1 || echo B2_RUNS_FAILED
for v in base exact_1000 exact_10000 exact_100000 near_1000 near_10000; do
  echo "[$(date -u +%T)] quality $v"; env/bin/python $T/b1_rung.py file b2_$v $G/b2/$v.npz > logs/b2q_$v.log 2>&1 || echo "B2Q_${v}_FAILED"
done
echo B2_CHAIN_DONE
