#!/bin/bash
cd /path/to/scratch/dedupjoin/pool_rebuild
export OMP_NUM_THREADS=3 NUMBA_NUM_THREADS=3 WRAP_THREADS=3
while pgrep -f tools/build_caches.py >/dev/null; do sleep 20; done
date > logs/accuracy.start
/path/to/scratch/dedupjoin/env/bin/python -I tools/run_accuracy_wrap.py root/dedupjoin-newsnap-selection/code/core > logs/run_accuracy.log 2>&1
date > logs/accuracy.end
