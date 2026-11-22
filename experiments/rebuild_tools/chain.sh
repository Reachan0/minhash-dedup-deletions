#!/usr/bin/env bash
# Chain: wait for env -> extract (12 workers, overlaps with downloads) -> check counts -> post-extract pipeline.
set -uo pipefail
W=/path/to/home/dedupjoin-postreveal
cd $W
echo "[$(date -u +%T)] waiting for env"
while ! grep -q '^SETUP_DONE' logs/setup_env.log; do
  if ! pgrep -f setup_env.sh >/dev/null; then echo "ENV_SETUP_FAILED"; exit 1; fi
  sleep 30
done
echo "[$(date -u +%T)] env ready; starting extraction"
env/bin/python tools/extract_driver.py 12 > logs/extract.log 2>&1
grep -q EXTRACT_DONE logs/extract.log || { echo "EXTRACT_FAILED"; exit 1; }
env/bin/python - <<'PY' || { echo "EXTRACT_COUNTS_FAILED"; exit 1; }
import json, sys
rows = [json.loads(l) for l in open("logs/extract_status.jsonl")]
ok = {r["rank"] for r in rows if r.get("status") == "ok" and r.get("counts_match")}
bad = [r for r in rows if not (r.get("status") == "ok" and r.get("counts_match"))]
print("segments ok:", len(ok), "bad:", len(bad)); [print(b) for b in bad[:10]]
sys.exit(0 if len(ok) == 104 and not bad else 1)
PY
echo "[$(date -u +%T)] extraction verified; deleting raw WET to free disk"
rm -f wet/*.warc.wet.gz
bash tools/post_extract_pipeline.sh > logs/pipeline.log 2>&1 && echo "CHAIN_DONE" || { echo "PIPELINE_FAILED"; tail -20 logs/pipeline.log; exit 1; }
