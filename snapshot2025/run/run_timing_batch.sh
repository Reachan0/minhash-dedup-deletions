#!/usr/bin/env bash
# Sequential timing batch for path A (the only literally executable path).
# One process at a time: the step's memory rule forbids parallel large runs.
# Each run is a fresh OS process wrapped by run_timed_tree.py, which appends a
# sidecar row (wall + process-tree peak RSS) and, on non-zero exit, a row to
# failed.jsonl. Timing rows append to newsnap/timing.jsonl.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SLUG=$(cd "$HERE/.." && pwd)
ROOT=$(cd "$SLUG/../.." && pwd)
cd "$ROOT" || exit 1
W="$HERE/run_timed_tree.py"
T="$HERE/wrap_timing.py"
SIDE="$SLUG/newsnap/timing_sidecar.jsonl"
FAIL="$SLUG/newsnap/failed.jsonl"
LOGS="$SLUG/logs"

one() {  # label tau mode repeat seed
  local label=$1 tau=$2 mode=$3 rep=$4 seed=$5
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $label" >&2
  python3 "$W" --label "$label" --log "$LOGS/$label.log" \
    --out "$SIDE" --failed "$FAIL" \
    -- python3 "$T" --method A --tau "$tau" --mode "$mode" \
    --seed "$seed" --repeat "$rep"
}

for r in 1 2 3; do one "snapA_pathcold_t0.8_r$r" 0.8 pathcold "$r" 1; done
for r in 1 2 3 4 5; do one "snapA_inc_t0.8_r$r" 0.8 incremental "$r" 1; done
for r in 1 2 3; do one "snapA_pathcold_t0.75_r$r" 0.75 pathcold "$r" 1; done
for r in 1 2 3; do one "snapA_inc_t0.75_r$r" 0.75 incremental "$r" 1; done
# Supplement only: literal text -> spacy shingle stage, once (D5).
one "snapA_textcold_t0.8_r1" 0.8 cold 1 1
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] batch done" >&2
