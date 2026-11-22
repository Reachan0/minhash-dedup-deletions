#!/usr/bin/env bash
# After extraction: assemble shards, verify identity, build the shingle cache, rerun the reveal's
# quality computation, and check it reproduces results/per_seed_quality.csv byte for byte.
# Stops at the first failed check. All outputs go to <package>/newsnap/ (never results/ or inputs/).
set -euo pipefail
W=/path/to/home/dedupjoin-postreveal
PKG=$W/dedupjoin-newsnap-selection
P=$W/env/bin/python
export OMP_NUM_THREADS=12
cd "$PKG"
ts() { date -u +%H:%M:%S; }

echo "[$(ts)] 1/7 assemble + identity check"
$P "$W/tools/assemble_verify.py"
$P -c "import json,sys; r=json.load(open('$W/logs/identity_check.json')); sys.exit(0 if r['identity_ok'] else 1)" \
  || { echo "IDENTITY_FAILED"; exit 1; }
cp "$W/dedupjoin-pvldb-v3/experiment/data/newsnap/manifest.json" newsnap/corpus/manifest.json

echo "[$(ts)] 2/7 shingle cache"
SNAP_WORKERS=14 $P inputs/build_cache.py
$P - <<'EOF'
import json, sys
r = json.load(open("newsnap/cache/build_report.json"))
ok = r["n_docs_match_manifest"] and r["nwords_total_match_manifest"] and r["empty_docs"] == 994
print("cache check:", {k: r.get(k) for k in ("n_docs", "n_shingles_total", "empty_docs", "nwords_total")}, "ok" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
EOF

echo "[$(ts)] 3/7 dup_sets"
$P code/run/dup_sets.py
$P - <<'EOF'
import json, sys
new = json.load(open("newsnap/dup_sets.json")); ref = json.load(open("results/dup_sets.json"))
keys = ("n", "n_unique_sets", "n_docs_in_duplicate_groups", "n_duplicate_groups", "largest_group")
diff = {k: (new.get(k), ref.get(k)) for k in keys if new.get(k) != ref.get(k)}
print("dup_sets vs reference:", "identical" if not diff else diff)
sys.exit(0 if not diff else 1)
EOF

echo "[$(ts)] 4/7 build_collapsed"
$P code/run/build_collapsed.py
echo "[$(ts)] 5/7 collapsed_truth"
$P code/run/collapsed_truth.py
echo "[$(ts)] 6/7 group_accuracy"
$P code/run/group_accuracy.py

echo "[$(ts)] 7/7 acceptance: regenerate per_seed_quality.csv and diff"
A=$W/acceptance; rm -rf "$A"; mkdir -p "$A/snap" "$A/out"
cp newsnap/runs.jsonl "$A/snap/"
for f in timing.jsonl timing_sidecar.jsonl failed.jsonl band_stats.json dup_sets.json collapsed_build.json largest_group.json; do
  cp "results/$f" "$A/snap/"
done
$P code/run/analyze_selection.py --snap "$A/snap" --inputs inputs --predictions prereg/selection_predictions.json --out "$A/out" > "$W/logs/acceptance_analyze.log" 2>&1
if cmp -s "$A/out/per_seed_quality.csv" results/per_seed_quality.csv; then
  echo "ACCEPTANCE_PASS per_seed_quality.csv identical"
else
  echo "ACCEPTANCE_FAIL"; diff "$A/out/per_seed_quality.csv" results/per_seed_quality.csv | head -20; exit 1
fi
echo "[$(ts)] PIPELINE_DONE"
