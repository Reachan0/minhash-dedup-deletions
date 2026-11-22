#!/bin/bash
W=/path/to/scratch/dedupjoin
PY=$W/env/bin/python
export NUMBA_NUM_THREADS=12 OMP_NUM_THREADS=12
$PY $W/tools/b2_autodl.py inject || { echo INJECT_FAILED; exit 1; }
$PY $W/tools/b2_autodl.py runs
echo B2_CHAIN_DONE
