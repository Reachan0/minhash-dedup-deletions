#!/bin/bash
# Scale ladder driver: one process per rung so peak RSS and the address-space cap are
# per workload.  Sequential by design (one machine, identical thread count).
set -u
cd "$(dirname "$0")/../../.." || exit 1
LOG=.work/dedupjoin/ladder/logs
mkdir -p "$LOG"
for w in prefix:300000 prefix:1000000 prefix:2000000 prefix:3000000 prefix:4000000; do
  name=${w/:/_}
  echo "=== $w $(date -u +%H:%M:%S) ==="
  python3 output/dedupjoin-pvldb/code/scale_ladder.py run --workload "$w" \
      --mem-cap-gib 58 --threads 16 > "$LOG/$name.log" 2>&1
  rc=$?
  echo "=== $w exit=$rc $(date -u +%H:%M:%S) ==="
  if [ $rc -ne 0 ]; then
    echo "{\"workload_id\": \"$w\", \"status\": \"failed\", \"exit\": $rc, \"ts_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"provenance\": \"independently rerun (this run)\"}" \
      >> output/dedupjoin-pvldb/results/scale_ladder_failures.jsonl
  fi
done
echo "=== ladder done $(date -u +%H:%M:%S) ==="
