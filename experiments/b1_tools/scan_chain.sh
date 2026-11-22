#!/bin/bash
W=/path/to/scratch/dedupjoin
export NUMBA_CACHE_DIR=/path/to/scratch/nbcache_scan
until grep -q CAP4_DONE $W/tools/cap4_chain.log; do sleep 20; done
for r in rung_531597 rung_1000000 rung_2000000 rung_3000000; do
  $W/env/bin/python $W/tools/audit_scan.py $r || echo "FAILED $r"
done
echo SCAN_DONE
