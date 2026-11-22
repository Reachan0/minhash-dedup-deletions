#!/usr/bin/env bash
# Regime panel at fixed size: 100k-document samples from the length and duplicate-density
# strata of the pool (all 37,713 members where the stratum is smaller than 100k).
# Fixed size holds N constant so that only length or duplication changes.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$(pwd)
LOG="$ROOT/.work/dedupjoin/ladder/logs"
mkdir -p "$LOG"
for b in uniform L1_lt50 L5_ge1500 D4_ge16 D1_isolated; do
  echo "=== strata:$b $(date -u +%H:%M:%S) ===" >> "$LOG/strata_driver.log"
  python3 output/dedupjoin-pvldb/code/scale_ladder.py run --workload "strata:$b" \
      --mem-cap-gib 58 --threads 16 > "$LOG/strata_$b.log" 2>&1
  rc=$?
  echo "=== strata:$b exit=$rc $(date -u +%H:%M:%S) ===" >> "$LOG/strata_driver.log"
  if [ $rc -ne 0 ]; then
    printf '{"workload":"strata:%s","exit":%d,"log":"%s"}\n' "$b" "$rc" "strata_$b.log" \
      >> output/dedupjoin-pvldb/results/scale_ladder_failures.jsonl
  fi
done
