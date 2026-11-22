# Code provenance — `audit-claims` step

Rule: any Draft 2 script reused is **copied** into this directory (never moved)
with source path and sha256 recorded. The read-only baseline
`output/dedupjoin-pvldb/` is not modified.

## Reused Draft 2 scripts

**None.** The four audit scripts below are new; they read Draft 2 result files
and parquet metadata, and import no Draft 2 module (`grep` over imports shows
only the standard library, numpy, pyarrow). The definitions in
`audit/glossary.md` were derived by reading the Draft 2 sources listed under
"Sources read, not copied".

## Audit scripts written in this step

| File | sha256 | Output |
|---|---|---|
| `code/audit/reconcile_missed.py` | `84d1033871779a2101c77e1102ce5bb3fa4e739062578a190df01127822121db` | `audit/missed_reconciliation.csv`, `audit/missed_summary.json` |
| `code/audit/number_provenance.py` | `e5eb9d45ef56eb2ee6d0c805fdcf14c24793737cebefd82abd6f03b461a69389` | `audit/number_provenance.json` |
| `code/audit/provenance_overlap.py` | `3d1f300b7f125ae46800ec5631f586f9d2df2c84e062b97724b2fda259fad5d8` | `audit/provenance_overlap.json` |
| `code/audit/make_claims_manifest.py` | `074b9c3aeccbdbc5cfcb52a3f7c347c38d88e774310d72de894c80e6c8b2f5f8` | `audit/claims_manifest.csv`, `audit/naming_mismatches.csv` |

Vendored dependency, unpacked for source inspection only (no import):
`code/audit/_wheel/datatrove-0.6.0-py3-none-any.whl`,
sha256 `f231eb894fd79189a4c96b4d2188ece1a3804a922dc49bdf3ddcecd274f68ab9`.

## Sources read, not copied (baseline, read-only)

| File | sha256 |
|---|---|
| `output/dedupjoin-pvldb/code/main_compare.py` | `3164500c83827b3e931dabc8310e87b60df4c6fdcf654621f7039f9b869513d9` |
| `output/dedupjoin-pvldb/code/verify_c03.py` | `569a8278608652f28c3f86db3a8ffed8beacbb7281693ad8b4a50d4a6acca151` |
| `output/dedupjoin-pvldb/code/banding_sweep.py` | `3b75e130e75595a68572513dc58e75ac44c7c8de5e4d2b77b45f58b59691f21e` |
| `output/dedupjoin-pvldb/code/make_tables_banding.py` | `f5729d0634831a7312ba15832b24ca4ad3f34931aa7c061cd714957fdafb9246` |
| `output/dedupjoin-pvldb/code/make_tables_step.py` | `29332e685e7525b5e97ac7998aec25ea97da98d66d7c82ce67d8ebf431d8eb38` |
| `output/dedupjoin-pvldb/artifacts/vendor/exactjoin.py` | `939afcb236c15508af5f278c7aef1a46ed0564776d76f633cdcfa1d9984587e1` |
| `.work/vldb2027-ai-db-ideas/experiment/src/C03/lsh.py` | `191c9b5086f7e5409557a4908e7ef4de135363a939b3bac2d1b69204e5671234` |

The min-id union rule lives in `lsh.py:117-136`; the datatrove union-by-size
rule is read from the unpacked wheel (`datatrove/pipeline/dedup/minhash.py`,
stage 3). The exact self-join implementation is `exactjoin.py`; its brute-force
cross-check is `results/exact_validation_bruteforce.json`.


# Appended — `core-cost-benchmark` step (2026-11-19)

Rule: Draft 2 scripts reused by this step are **copied** (never moved) into
`code/core/`; the read-only baseline `output/dedupjoin-pvldb/` is untouched.

## Copied Draft 2 files

| Copy | Source (baseline, read-only) | sha256 |
|---|---|---|
| `code/core/vendor/common.py` | `output/dedupjoin-pvldb/artifacts/vendor/common.py` | `32418023a8bebcf14656549d2d7ffd54ff06260eb1230f19ca2a86a084cc2802` |
| `code/core/vendor/exactjoin.py` | `output/dedupjoin-pvldb/artifacts/vendor/exactjoin.py` | `939afcb236c15508af5f278c7aef1a46ed0564776d76f633cdcfa1d9984587e1` |
| `code/core/vendor/lsh.py` | `.work/vldb2027-ai-db-ideas/experiment/src/C03/lsh.py` | `191c9b5086f7e5409557a4908e7ef4de135363a939b3bac2d1b69204e5671234` |
| `code/core/reference/main_compare.py` | `output/dedupjoin-pvldb/code/main_compare.py` | `3164500c83827b3e931dabc8310e87b60df4c6fdcf654621f7039f9b869513d9` |

`reference/main_compare.py` is kept for reading the Draft 2 pipeline definition; no
script in `code/core/` imports it. The three `vendor/` modules are imported through
`core_lib.py` so that tokenisation, shingling, signatures and exact join are the same
code as Draft 2 (only the orchestration is new).

## New scripts in this step

`core_lib.py` (shared stage library), `make_workloads.py` (freeze row-id prefixes),
`crosscheck_bruteforce.py`, `crosscheck_datatrove.py`, `run_accuracy.py`,
`run_timing.py`, `timing_campaign.py`, `run_bucket_scaling.py`,
`scaling_campaign.py`, `analyze_core.py`. Their sha256 are in `code/core/SHA256SUMS`.
Diagnostics `diag_sig.py`, `diag_pairs.py`, `diag_stage2.py` reproduce the datatrove
stage-scheduling finding and are kept under `code/core/diagnostics/`.
