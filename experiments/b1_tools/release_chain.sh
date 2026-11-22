#!/bin/bash
W=/path/to/scratch/dedupjoin
export NUMBA_NUM_THREADS=12
until grep -q CAP3_DONE $W/tools/cap3_chain.log; do sleep 30; done
$W/env/bin/python $W/tools/release_snapshot.py $W/release/snapshot2025
