#!/bin/bash
# capped B rerun: same runs as cap3, recording the component bound (sum over capped buckets of min(U_b, c_b - 1))
W=/path/to/scratch/dedupjoin
PY=$W/env/bin/python
T=$W/tools
OP=$W/dedupjoin-newsnap-selection/code/operator
export NUMBA_NUM_THREADS=12
export NUMBA_CACHE_DIR=/path/to/scratch/nbcache_cap4
for g in two distinct noisy; do
  ADV_KINDS=cap $PY $T/adv_bucket.py $OP $W/results/adv_bucket_cap4.jsonl $g 1000 10000 30000 100000
done
$PY - <<'PYEOF'
import os, sys
os.environ["B2_W"] = "/path/to/scratch/dedupjoin"
sys.path.insert(0, "/path/to/scratch/dedupjoin/tools")
import b2_runs as R
for v in ["base", "exact_1000", "exact_10000", "exact_100000", "near_1000", "near_10000", "near_100000"]:
    for q in (100, 1000):
        R.run(v, "cap", q, cap_prefix=False, out="b2_cap4.jsonl")
R.run("exact_100000", "cap", 1000, cap_prefix=True, out="b2_cap4.jsonl")
PYEOF
echo CAP4_DONE
