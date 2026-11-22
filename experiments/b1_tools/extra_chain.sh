#!/bin/bash
# Signature-only verification, operator cost on the nested corpora and synthetic buckets, on AutoDL, run one after another so that timings do not overlap.
W=/path/to/scratch/dedupjoin
PY=$W/env/bin/python
T=$W/tools
OP=$W/dedupjoin-newsnap-selection/code/operator
export NUMBA_NUM_THREADS=12
for N in 531597 1000000 2000000 3000000; do
  seeds=1
  [ "$N" = 531597 ] && seeds=1,2,3,4,5
  $PY $T/sig_verify.py $N $seeds
  $PY $T/ladder_ops.py $N
done
ADV_KINDS=stream,cap,cap1,W $PY $T/adv_bucket.py $OP $W/results/adv_bucket.jsonl noisy 1000 10000 30000 100000
ADV_KINDS=cap $PY $T/adv_bucket.py $OP $W/results/adv_bucket.jsonl two 1000 10000 30000 100000
ADV_KINDS=cap $PY $T/adv_bucket.py $OP $W/results/adv_bucket.jsonl distinct 1000 10000 30000 100000
echo EXTRA_DONE
