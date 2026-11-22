# Post-reveal snapshot drivers

Run from the workspace after the six rebuilt shards and the finished shingle cache are in place. Run the commands sequentially. E1 publishes the accepted mask directory that E2 and E3 consume. Each invocation starts a new campaign and appends uniquely keyed rows.

```bash
cd /path/to/home/dedupjoin-postreveal
D=dedupjoin-newsnap-selection/code/post_reveal
export OMP_NUM_THREADS=12
env/bin/python -B "$D/e1_outside_group.py" &&
  bash "$D/e2_run.sh" &&
  env/bin/python -B "$D/e3_filter50.py"
```

## Synthetic smoke tests

```bash
cd /path/to/home/dedupjoin-postreveal
D=dedupjoin-newsnap-selection/code/post_reveal
env/bin/python -B "$D/e1_outside_group.py" --smoke &&
  bash "$D/e2_run.sh" --smoke &&
  env/bin/python -B "$D/e3_filter50.py" --smoke
```

Each invocation builds its own deterministic 3,000-row parquet corpus and spaCy shingle cache under `$W/smoke/`. The fixture includes an interlinked duplicate group, distinct raw strings that normalize to the same shingles, near-duplicate families, empty shingle sets, and documents on both sides of 50 words. The literal exact join, document bucket expansion, union-find, and greedy MIS provide independent document masks for all 32 reference quality cells. E2 exercises every method, threshold, collapse key and cache mode once. E3 also checks the lifting identity on the filtered subset. Smoke timings use the same process isolation, 12-thread configuration and cap as the snapshot. A successful campaign prints `PASS`.

## Inputs and acceptance

- **E1.** Stream parquet shards in frozen order and check every raw hash against `inputs/doc_text_xxh64.npy`, both sequence digests against `inputs/INPUT_PATHS.json`, the document and word totals, and the snapshot cache SHA-256. Check 455,796,881 shingle entries, 994 empty documents, 431,116 distinct shingle sets, and the 86,771-document largest group even if the cache SHA matches. A cache with different bytes must still pass all structural and quality checks. Run the frozen `collapsed_lib` exact join and `group_accuracy.main()` in the campaign directory. Compare deleted and unsupported counts against **every** row of `results/per_seed_quality.csv`. Any mismatch aborts before computing outside-group metrics. Retain the new masks and the accepted input file fingerprint. The E1 report gives the seed-1 ratio, both five-seed means and ranges, and the P8 comparison with 35.9%.
- **E2.** Both collapse keys order groups by minimum original row. Empty shingle sets have no edges. Shingle-set keys are xxh64 of the sorted, unique uint64 slice. Raw keys come from the frozen UTF-8 xxh64 array validated by E1. Pathcold times loading shingles, computing or reading keys and grouping, packing representatives, 112-permutation seed-1 signatures, 14×8 banding, group bucket enumeration, exact verification, union-find or greedy MIS, lifting and writing the document mask. Verification uses the representative rows in the loaded document CSR. Incremental reads the shingles, a cached group map, and cached signatures, then runs banding through mask writing. Cache preparation and reference-mask checks are outside the measured wall time. Literal A runs at both thresholds on the same machine. Every timed mask must equal its E1 reference element by element. At τ=0.8 seed 1, B must delete 113,625 and W 112,798. Matching the same document mask also checks equality across the two keys, modes and repeats.
- **E3.** The extractor stores `len(text.split())` in parquet `nwords`. Its frozen total is 522,207,916 across 531,597 rows, giving mean 982.3379665. The NPZ field with the same name counts normalized tokenizer output. Filtering reads parquet `nwords >= 50`, retains source order, writes a mask-index-to-source-row-and-ID table, and builds exact truth from the retained rows. Five literal A seeds and seed-1 literal B/W are evaluated at τ=0.8. Every worker starts under a 24 GiB address-space cap. A literal allocation failure records the stage, exit code, log and sampled peak. B completes its scheduled attempts, and a failed B or W gets a separate shingle-collapsed quality run. The fallback has its own key and mode. A worker terminated by a signal gets the same fallback and records the signal with cause unconfirmed. Other errors abort the campaign. Quality rows report exact deletions missed, residual retained pairs, witness coverage, largest bucket and Σ C(m,2)/N. Bucket statistics refer to the nodes used for that run. The literal A rows provide document-level bucket statistics at every seed.

## Time and memory

The spec estimates about one hour for E1, two hours for E2 and two hours for E3 on the rebuilt snapshot. These are planning estimates. E2 schedules 80 timed processes (two thresholds, four collapsed method/key combinations plus literal A, three pathcold and five incremental repeats). E3 schedules eight repeats each for literal A and B, one pathcold W, and four additional A quality seeds. Input verification, filtered cache construction, truth, cache preparation and possible fallbacks add work.

Allow 24 GiB of available memory per active experiment and run one campaign at a time. The full document flat shingle array alone occupies about 3.40 GiB. Building groups or packing the filtered corpus can temporarily hold a second similarly sized array. The prior collapsed exact-truth worker reported about 12.85 GiB peak self RSS. Real literal B/W peak usage on the filtered snapshot is measured by E3. Every worker is CPU-only with 12 threads. The cap limits virtual address space, so an allocation failure may occur at lower resident memory.

## Results and interpretation

- The append-only ledger is `dedupjoin-newsnap-selection/results_post_reveal/post_reveal.jsonl`. Select `smoke == false` for the paper. Each row includes a campaign-specific key, experiment, method, seed, threshold, collapse key, mode, repeat, hardware and UTC timestamp. Summary rows link their input keys and have null measurement fields. Per-campaign files include the worker logs, process-tree sampler sidecar, deletion masks, environment versions and report fragment. Real campaign directories are under `results_post_reveal/campaigns/`. Synthetic campaigns use the same layout under `$W/smoke/`.
- The combined real-data report is `results_post_reveal/POST_REVEAL_RESULTS.md`. E1 supplies P1/P3/P5/P8, E2 supplies P2/P4/P6, and E3 supplies P7. Each reported number links to a ledger key. Original results and inputs are read-only.
- `wall_s` for a successful timed path runs from loading shingles through closing the written mask. `process_wall_s` includes interpreter startup, output checks, and sampler overhead. The frozen sampler sums RSS of the worker and its descendants every 0.5 seconds. `peak_tree_rss_mb` uses MiB. `load_shingles_s`, per-stage durations and `load_shingles_read_bytes` are recorded for each timed path. Pages are left in the OS cache. A zero storage-read delta is consistent with a warm cache. Fresh process timing does not imply a flushed filesystem cache.
- Timing summaries report median wall and stage seconds, and maximum sampled tree RSS across repeats. Each individual S1–S3 cost inequality is recorded for each configuration. S1 uses the E2 pathcold boundary for the 400-second comparison. A text-to-mask claim under the original end-to-end preregistration would also require the text shingling cost. S1–S3 cost checks alone do not establish the frozen quality tolerances.
