#!/usr/bin/env bash
# Same-corpus size ladder: C4 only, first N documents. The prefix ladder mixes
# FineWeb -> C4 -> WET as it grows, so size and duplicate density co-vary there; this
# ladder holds the corpus fixed and changes only N. Runs after the prefix ladder so the
# two never compete for cores.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$(pwd)
LOG="$ROOT/.work/dedupjoin/ladder/logs"
mkdir -p "$LOG"
for spec in c4:100000 c4:1000000 c4:2850000; do
  tag=${spec/:/_}
  echo "=== source:$spec $(date -u +%H:%M:%S) ===" >> "$LOG/source_driver.log"
  python3 output/dedupjoin-pvldb/code/scale_ladder.py run --workload "source:$spec" \
      --mem-cap-gib 58 --threads 16 > "$LOG/source_$tag.log" 2>&1
  rc=$?
  echo "=== source:$spec exit=$rc $(date -u +%H:%M:%S) ===" >> "$LOG/source_driver.log"
  if [ $rc -ne 0 ]; then
    printf '{"workload":"source:%s","exit":%d,"log":"%s"}\n' "$spec" "$rc" "source_$tag.log" \
      >> output/dedupjoin-pvldb/results/scale_ladder_failures.jsonl
  fi
done
