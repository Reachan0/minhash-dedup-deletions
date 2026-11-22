# Clean-room reproduction acceptance

Run 2026-11-19 22:47–22:52 UTC. A fresh working root was assembled under
`.work/dedupjoin-pvldb-v3/cleanroom/ws/` containing only a copy of the published
`experiment/code/core` and symlinks to the frozen corpora (`.work/dedupjoin`). No file
of the working tree, no cached truth graph and no cached signature was reused: the
clean room's `core/cache/` started empty, so the truth graph and every MinHash signature
were recomputed from the shard caches.

Hardware: 4 vCPU (Intel Xeon E5-2686 v4 @ 2.30 GHz), 16 GiB cgroup limit, no GPU.
Python 3.11.2; numpy 2.4.6, scipy 1.17.1, pandas 3.1.1, pyarrow 26.0.1, numba 0.68.1,
xxhash 4.0.2, datatrove 0.6.0 (installed from the pinned requirements).

## 1. Input integrity — `rerun`

`verify_shards.py` recomputed sha256 for every file listed in the pool manifest
(`output/dedupjoin-pvldb/results/data_manifest.json`):

    {"checked": 85, "mismatch": [], "bytes": 5188986555}

85/85 files match, 5,188,986,555 bytes. The manifest also carries the source URLs and
sha256 of the raw C4/FineWeb/WET shards; those raw shards were not re-hashed in this
run (the pool files derived from them were).

## 2. End-to-end accuracy re-run — `rerun`

Command (clean room root):

    cd cleanroom/ws/code/core
    python3 run_accuracy.py --workloads wet --taus 0.8

The run started from an empty cache: truth graph recomputed (28,483 edges, 9.1 s),
three MinHash seeds recomputed from the shingle caches (10–11 s per seed), and A, B, W
and C run under both representative rules. Wall time 58 s.

Comparison with the published per-run records (`core/runs.jsonl`), all 15 A/B/W rows
(3 seeds x 4 method/representative combinations) and both C rows:

| Workload | tau | Method | Representative | Deleted (published / rerun) | Unsupported (published / rerun) | Pair recall (published / rerun) |
|---|---|---|---|---|---|---|
| WET | 0.8 | A | min-id | 6800 / 6800 | 2425 / 2425 | 0.184847 / 0.184847 |
| WET | 0.8 | B | min-id | 4221 / 4221 | 0 / 0 | 0.992487 / 0.992487 |
| WET | 0.8 | W | min-id | 4115 / 4115 | 0 / 0 | 0.992487 / 0.992487 |
| WET | 0.8 | C | min-id | 4245 / 4245 | 0 / 0 | — |

Seeds 2 and 3 and the datatrove representative match in the same way (17 rows
compared, 17 identical). Log: `experiment/logs/acceptance_accuracy.log`.

## 3. Decision table from per-run records — `log-recomputed`

`verify_decision_table.py` recomputes every accuracy and timing field of
`main_decision_table.csv` from `runs.jsonl` (156 rows) and `timing.jsonl` (176 rows)
using the documented aggregation (mean over seeds; median over timing repeats; W cold
time composed as measured B cold preprocessing plus W incremental algorithm time):

    {"fields_checked": 672, "failures": 0, "rows": 56, "runs": 156, "timing_rows": 176}

56 table rows, 672 compared fields, 0 mismatches. This checks the published table
against the published records; it does not re-time the runs.

## 4. Tables and figures — `redrawn`

`code/paper/make_tables_v3.py` was re-run from the stored result files; all six
`tables/tab*.tex` files are byte-identical to the published ones (manifest sha256
unchanged for tables 1 and 3–6; table 2 changed only by a typographic
`\Sigma` substitution, no number changed). `code/paper/make_paper_figures.py` was
re-run; all three figures are byte-identical PNGs.

## 5. Not covered by this acceptance

- The 32-cell accuracy grid and the timing campaign were not re-run from scratch
  (hours of compute); only the WET tau=0.8 cell was recomputed end to end and every
  table field was re-derived from the per-seed records.
- The independent snapshot (CC-MAIN-2025-30) has a frozen pre-registration and a built
  corpus, but no deletion set: its reveal was not executed (see `newsnap/PREREG.md`).
- Human labels do not exist; the content audit uses model and script labels only.
- Raw C4/FineWeb/WET downloads were not re-fetched; their sha256 are recorded in the
  pool manifest and the derived pool was verified byte for byte.
