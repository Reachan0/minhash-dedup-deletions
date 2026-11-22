# Reproduction package

This directory documents how the numbers in the paper were produced and what an
external reader can recompute without the original working tree.

## Data

Large data is reused in place and is not shipped. The layout expected by
`code/core/core_lib.py` is:

    .work/dedupjoin/rerun/{c4_700000,fw_1000000}.npz        cached 5-gram shingles
    .work/dedupjoin/main/{mix_1000000,wet_175272}.npz       cached 5-gram shingles
    .work/dedupjoin/main/{c4,fw,mix,wet}_ranks.npz          frozen rank order
    .work/dedupjoin/pool/{fw,c4,wet}/shard-*.parquet       4,074,391-document pool
    .work/dedupjoin/pool/meta/index-*.parquet              pool metadata
    .work/dedupjoin/pool/strata.parquet                    length/density/domain strata

The pool manifest (published with the Draft 2 tree) records 85 files, their sha256 and
the source URLs of the raw C4/FineWeb/WET shards. Re-check them with
`analysis/reproduction/verify_shards.py` (85/85 matched on 2026-11-19).

## Environment

Python 3.11.2, numpy 2.4.6, scipy 1.17.1, pandas 3.1.1, pyarrow 26.0.1,
numba 0.68.1, xxhash 4.0.2, datatrove 0.6.0. See `environment.txt`.
CPU-only; the reported timings use 12 threads on an Intel Xeon E5-2686 v4.

## Commands

    cd code/core
    bash run_core.sh mini        # WET, one threshold, all four paths, ~10 min (12 threads)
    bash run_core.sh full        # every cell, hours, writes the paper table
    bash run_core.sh analyze     # table + figures from records already on disk
    bash run_core.sh datatrove   # official datatrove 0.6.0 cross-check

    python3 ../code/paper/make_tables_v3.py       # six tables + manifest
    python3 ../code/paper/make_paper_figures.py   # three figures

The clean-room acceptance run and its commands are in `ACCEPTANCE.md`; the log of the
WET re-run is `logs/acceptance_accuracy.log`.

## What each tier means

- `rerun`: computed in this run from the shingle/rank caches (e.g. the clean-room WET
  cell, the core accuracy and timing campaigns).
- `log-recomputed`: recomputed from per-seed/per-run records (the 672 decision-table
  fields, the correctness ledger, the variance decomposition).
- `redrawn`: re-plotted or re-typeset from stored result JSON/CSV (figures and tables).

## Known limits

The independent snapshot (CC-MAIN-2025-30) has a frozen pre-registration and a built
corpus, but no deletion set; its reveal is not part of this package. Human labels do
not exist; the content audit uses model and script labels, recorded separately.
