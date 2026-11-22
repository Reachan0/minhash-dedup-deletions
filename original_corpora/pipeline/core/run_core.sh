#!/usr/bin/env bash
# Reproduce the core cost benchmark.
#
#   ./run_core.sh mini    WET only, 2 seeds, 1 timing repeat: ~10 min, checks the whole chain
#   ./run_core.sh full    every workload/tau, 3-5 seeds, 2-5 repeats: hours, writes the paper table
#   ./run_core.sh analyze table + figures from whatever is already on disk
#   ./run_core.sh datatrove  official datatrove 0.6.0 cross-check (WET 1,400 docs)
#
# Official datatrove 0.6.0 schedule (verified in crosscheck_datatrove.py): the signature
# stage and the sort/pair stage MUST run as two separate executor jobs.  With
# workers=1, tasks=14 inside one pipeline the sort stage sees only the signature files
# whose rank was already finished (bucket 0: 45 pairs instead of 798).  The check below
# asserts the intermediate file count (14 signature shards) before stage 2 starts and
# then compares the pair set to this repository's chain edges.
#
# Fixed configuration (identical for every path; see ../env.txt):
#   MinHash 112 permutations, 14 bands x 8 rows, value seed 1, tau in {0.8, 0.75},
#   12 threads, one fresh OS process per timed configuration.
# Large data (pool shards, shingle/signature caches) lives OUTSIDE this directory and is
# reused in place; nothing here copies it.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
MODE=${1:-mini}

case "$MODE" in
  mini)
    $PY run_accuracy.py --workloads wet --taus 0.8
    $PY timing_campaign.py --workloads wet --taus 0.8 --modes inc,cold
    $PY analyze_core.py
    ;;
  full)
    $PY run_accuracy.py                       # 5 seeds c4/fw at 0.8, 3 elsewhere
    $PY timing_campaign.py --modes inc
    $PY timing_campaign.py --modes cold
    $PY timing_campaign.py --modes inc1,cold1
    $PY scaling_campaign.py
    $PY analyze_core.py
    $PY figures/make_figures.py
    ;;
  analyze)
    $PY analyze_core.py
    $PY figures/make_figures.py
    ;;
  datatrove)
    # Runs the official pipeline as two separate jobs and asserts 14 signature shards
    # exist between them, then compares pairs and removed sets.  Needs the 0.6.0 wheel.
    $PY crosscheck_datatrove.py
    ;;
  *)
    echo "usage: $0 [mini|full|analyze|datatrove]" >&2; exit 2
    ;;
esac
