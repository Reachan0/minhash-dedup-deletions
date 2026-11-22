#!/usr/bin/env bash
# server: rerun the B2 quality analyses that failed when the shared disk filled up.
# Each variant's working directory under runs/ is removed once its results file is written.
set -uo pipefail
G=/path/to/home/dedupjoin-postreveal
cd $G
export B2_W=$G OMP_NUM_THREADS=12
T=dedupjoin-pvldb-v4/experiments/b1_tools
for v in exact_10000 near_1000 near_10000 exact_100000; do
  rm -rf runs/b2_$v results/b2_$v.jsonl
  echo "[$(date -u +%T)] quality $v"
  env/bin/python $T/b1_rung.py file b2_$v $G/b2/$v.npz > logs/b2q_$v.log 2>&1 || echo "B2Q_${v}_FAILED"
  rm -rf runs/b2_$v
done
echo B2Q_RERUN_DONE
