#!/usr/bin/env bash
# AutoDL: R1 + anchor + streaming B on the original corpora, then R7 (official datatrove, matched truth).
set -uo pipefail
A=/path/to/scratch/dedupjoin
PR=$A/pool_rebuild
cd $PR/root
export OMP_NUM_THREADS=3 NUMBA_NUM_THREADS=3
echo "[$(date -u +%T)] r1_pool"
$A/env/bin/python $A/tools/pool/r1_pool.py $PR/path/to/dedupjoin-newsnap-selection/code/core \
  $A/dedupjoin-newsnap-selection/code/operator $A/results/r1_pool.jsonl > $A/logs/r1_pool.log 2>&1 || echo R1_POOL_FAILED
echo "[$(date -u +%T)] r7"
for ls in 0.8 0.99; do
  $A/env/bin/python $A/tools/pool/r7_official_matched.py --wet-dir $PR/wetjson --work $A/r7_ls$ls \
    --repo $A/r7repo --lang-score $ls > $A/logs/r7_ls$ls.log 2>&1 || echo "R7_${ls}_FAILED"
  grep -q '"matches_earlier_run": true' $A/r7_ls$ls/r7_result.json 2>/dev/null && { echo "r7 matched at lang score $ls"; break; }
done
echo POOL_CHAIN_DONE
