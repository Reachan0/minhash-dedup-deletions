#!/bin/bash
W=/path/to/scratch/dedupjoin
export NUMBA_NUM_THREADS=12
for N in 531597 1000000 2000000 3000000; do
  $W/env/bin/python $W/tools/coverage_sim.py $N 20000
done
echo COV_DONE
