#!/bin/bash
# capped B with counted (not enumerated) untested pairs: synthetic buckets and B2 variants
W=/path/to/scratch/dedupjoin
PY=$W/env/bin/python
T=$W/tools
OP=$W/dedupjoin-newsnap-selection/code/operator
export NUMBA_NUM_THREADS=12
until grep -q COV_DONE $T/cov_chain.log; do sleep 30; done
ADV_KINDS=cap,cap1 $PY $T/adv_bucket.py $OP $W/results/adv_bucket.jsonl noisy 1000 10000 30000 100000
ADV_KINDS=cap $PY $T/adv_bucket.py $OP $W/results/adv_bucket.jsonl two 1000 10000 30000 100000
ADV_KINDS=cap $PY $T/adv_bucket.py $OP $W/results/adv_bucket.jsonl distinct 1000 10000 30000 100000
$PY - <<'PYEOF'
import os, sys
os.environ["B2_W"] = "/path/to/scratch/dedupjoin"
sys.path.insert(0, "/path/to/scratch/dedupjoin/tools")
import b2_runs as R
for v in ["base", "exact_1000", "exact_10000", "exact_100000", "near_1000", "near_10000", "near_100000"]:
    for q in (100, 1000):
        R.run(v, "cap", q, cap_prefix=False, out="b2_cap3.jsonl")
R.run("exact_100000", "cap", 1000, cap_prefix=True, out="b2_cap3.jsonl")
PYEOF
echo CAP3_DONE
