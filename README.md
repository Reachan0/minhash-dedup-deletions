# What Does MinHash-LSH Deduplication Delete? — reproducibility package

Code, result files and per-document release files for the PVLDB paper
*What Does MinHash-LSH Deduplication Delete? A Document-Level Experimental Analysis*.

The paper audits four deduplication paths under a duplicate criterion `J >= tau` over word
5-gram shingles, at `tau = 0.8` and `0.75`, with 14x8 banding:

- **A** is the production MinHash-LSH chain: transitive closure over bucket collisions.
- **B** verifies every within-bucket pair before the closure.
- **W** is the witness rule.
- **C** is the exact self-join that serves as the reference.

Every table and figure is backed by a result file in `results/`. Section 4 of this README
lists, for each one, the result file and the script that produced it.

## 1. Layout

| Directory | Contents |
|---|---|
| `operator/` | Streaming B with the connectivity skip and the star pass (`stream_b.py`, capped mode `stream_b_cap`), exact W (`exact_w.py`), anchor-verified baseline (`anchor_b.py`), shingle-set collapse (`collapse.py`), exact join with 64-bit offsets (`exactjoin64.py`), GPU verification (`gpu_verify.py`, `gpu_native.py`, `gpu_verify_kernels.mu`), tests (`tests/`). |
| `original_corpora/pipeline/` | Pipeline for C4, FineWeb, their mixture and WET-2019: signatures, banding, paths A/B/W/C, exact reference, cross-checks against brute force and datatrove (`core/`), deletion classes, components and seed variance (`correctness/`). |
| `original_corpora/draft/` | Earlier scripts for the same corpora: banding sweep, seed variance, the content-audit prompt (`content_audit/llm_label.py`). |
| `snapshot2025/` | Pre-registered test on CC-MAIN-2025-30, Section 9. Frozen protocols with their SHA-256 (`prereg/`). Input identity references: per-document xxh64, shard hashes (`inputs/`). Snapshot build (`newsnap/`), frozen evaluation code (`core/`, `run/`, each with `SHA256SUMS`), post-reveal experiments (`post_reveal/`). |
| `experiments/` | Experiments added after the reveal. Nested corpora up to 3M documents, injected copies, capped B, estimators, synthetic buckets, GPU runs (`b1_tools/`). Rebuild of the original corpora (`pool_tools/`) and of the 2025 snapshot (`rebuild_tools/`, segment list `wet_list.tsv`). |
| `paper_tables/` | Generators for the paper's tables and figures from `results/`. |
| `results/paper/` | Result files behind Sections 5–11. |
| `results/original_corpora/` | Runs, timings and analyses on the four established corpora. |
| `results/snapshot2025_selection/` | Runs, timings and selection outcome of the pre-registered test. |
| `results/snapshot2025_post_reveal/` | Runs on Snapshot-2025 after the reveal: A outside the template group (E1), timings (E2), the 50-word filter (E3), datatrove's representative rule, and streaming B, exact W and the anchor check on the uncollapsed documents (`operator_E.jsonl`). `POST_REVEAL_RESULTS.md` summarises them. |
| `data/snapshot2025/` | Per-document release files for Snapshot-2025 (no document text); see its README. |
| `environment/` | Machine description and package versions of the snapshot runs. |

## 2. Environment

The experiments used Python 3.11 with the packages listed in `requirements.txt`. Path A follows
datatrove 0.6.0. The machines were:

| Machine | Use |
|---|---|
| Intel Xeon E5-2686 v4, 16 vCPU, 48 GiB | Original corpora (Figure 1) |
| Intel Xeon Platinum 8259CL, 16 cores, 32 GiB | Pre-registered test (Section 9) |
| Intel Xeon Silver 4110, 16 cores, 62 GiB | Post-reveal runs on Snapshot-2025, injected variants (Table 5) |
| Intel Xeon Gold 6430, 15 vCPU, 100 GiB | Nested corpora, estimators, synthetic buckets |
| Moore Threads MTT S4000 (MUSA 5.2) and NVIDIA RTX 4090 | GPU verification |

```bash
python3.11 -m venv env && env/bin/pip install -r requirements.txt
env/bin/pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl  # original corpora only
```

The experiment scripts were run on these machines, and their first lines name the workspace they
used, for example `W=/path/to/scratch/dedupjoin`. Point that variable to your own workspace.

## 3. Data

- **C4, FineWeb, Mix, WET-2019.** `original_corpora/data_manifest.json` lists every source shard
  with its URL, size and SHA-256, the fastText language-ID model, and the processing steps.
  `experiments/pool_tools/download.py original_corpora/data_manifest.json RAWDIR` fetches the
  shards and checks them. `build_caches.py` shingles them and `assemble_pool.py` assembles the
  corpora. The pipeline in
  `original_corpora/pipeline/core/` runs the paths.
- **Snapshot-2025.** 531,597 English documents of CC-MAIN-2025-30, taken from the 104 WET segments
  in `experiments/rebuild_tools/wet_list.tsv`.
  - `fetch_one.sh` and `extract_driver.py` fetch and extract the segments.
  - `snapshot2025/newsnap/build_snapshot.py` builds the shingle cache.
  - `snapshot2025/inputs/verify_inputs.py` checks every document against
    `inputs/doc_text_xxh64.npy`.
- **Nested corpora (1M, 2M, 3M).** Further segments of the same crawl.
  `experiments/b1_tools/b1_plan.py` picks the segments, `b1_fetch_one.sh` and
  `b1_segment_worker.py` build them, and `b1_rung.py` runs one size.
- **Injected variants.** `experiments/b1_tools/b2_inject_snapshot.py` builds the filtered snapshot
  with 10^3, 10^4 or 10^5 injected copies or near-copies of one page.

### Check without any corpus

`data/snapshot2025/documents.parquet` holds each path's deletions and the exact soundness label.
The unsupported share of Table 1 can be recomputed from it directly:

```python
import pyarrow.parquet as pq
t = pq.read_table("data/snapshot2025/documents.parquet").to_pandas()
for m in ["A", "A_dt", "B", "W"]:
    d = (t[m].values & 1) > 0                          # deleted with hash seed 1
    print(m, d.sum(), (~t.nb_080.values[d]).mean())    # A: 130442 deleted, 0.117 unsupported
```

For another deduplicator, take its deletion set by `doc`. Its unsupported share is the share of
those deletions with `nb_080 == False`.

## 4. Tables and figures

| Paper | Result files | Produced by | Table or figure from |
|---|---|---|---|
| Lifted B and W (Section 4) | `results/snapshot2025_post_reveal/operator_E.jsonl` | `operator/tests/run_snapshot.py`, `run_snapshot_w.py` | — |
| Figure 1 | `results/original_corpora/main_decision_table.json`, `timing.jsonl` | `original_corpora/pipeline/core/run_accuracy.py`, `run_timing.py` | `paper_tables/make_fig_12.py` |
| Table 1 | `results/original_corpora/main_decision_table.csv`, `results/paper/r1_pool.jsonl`, `results/snapshot2025_selection/per_seed_quality.csv`, `results/snapshot2025_post_reveal/post_reveal.jsonl`, `r1_datatrove_rule.jsonl`, `results/paper/b2_base.jsonl` | `original_corpora/pipeline/core/`, `experiments/pool_tools/r1_pool.py`, `snapshot2025/post_reveal/e1_outside_group.py`, `r1_datatrove_rule.py`, `e3_filter50.py` | — |
| Figure 2 | `results/original_corpora/correctness/component_cases.json` | `original_corpora/pipeline/correctness/run_components.py` | `paper_tables/make_fig_12.py` |
| Table 2 | `results/original_corpora/correctness/variance_decomposition.json` | `original_corpora/pipeline/correctness/run_variance.py` | — |
| Figure 3, Table 3 | `results/paper/rung_{531597,1000000,2000000,3000000}.jsonl`, `b2_base.jsonl` | `experiments/b1_tools/b1_rung.py` | `paper_tables/make_fig_skew.py` |
| Table 4 | `results/paper/ladder_ops.jsonl`, `rung_*.jsonl` | `experiments/b1_tools/ladder_ops.py` | `paper_tables/make_tab_ladder.py` |
| Table 5 | `results/paper/b2_runs.jsonl`, `b2_cap2.jsonl`, `b2_cap4.jsonl`, `b2_{exact,near}_*.jsonl` | `experiments/b1_tools/b2_runs.py`, `b2_autodl.py` | `paper_tables/make_tab_cost.py` |
| Table 6 | `results/paper/rung_531597.jsonl`, `sig_verify.jsonl`; `results/snapshot2025_post_reveal/post_reveal.jsonl`, `operator_E.jsonl` | `experiments/b1_tools/sig_verify.py`, `operator/tests/run_snapshot_anchor.py` | `paper_tables/make_tab_rules.py` |
| Table 7 | `results/paper/adv_bucket.jsonl`, `adv_bucket_cap4.jsonl` | `experiments/b1_tools/adv_bucket.py` | `paper_tables/make_tab_adv.py` |
| GPU (Section 8) | `results/paper/gpu_runs.jsonl`, `gpu_runs_4090.jsonl` | `experiments/b1_tools/gpu_runs.py`, `operator/gpu_*` | — |
| Partitioned verification (Section 8) | `results/paper/move_stats.jsonl` | `experiments/b1_tools/move_stats.py`, `comp_stats.py` | — |
| Table 8 | `results/snapshot2025_selection/`, `snapshot2025/prereg/` | `snapshot2025/run/` | — |
| Section 10 | `results/paper/F_rung_*.jsonl`, `F_scan_rung_*.jsonl`, `F_b2.jsonl`, `coverage_sim.jsonl` | `experiments/b1_tools/f_estimators.py`, `audit_scan.py`, `f_est_b2.py`, `coverage_sim.py` | — |
| Table 9 | `results/paper/content_audit_kept_summary.json`, `content_audit_kept_extra.json`, `content_audit_retest.json`; `data/snapshot2025/content_audit_labels.parquet` | `snapshot2025/post_reveal/c_content_audit.py`, `experiments/b1_tools/audit_autodl.py`, `paper_tables/audit/` | — |
| Section 12 | `results/paper/extra_audit.jsonl`, `banding_hi.jsonl`, `banding_hi2.jsonl` | `experiments/b1_tools/extra_audit.py`, `experiments/pool_tools/banding_hi.py` | — |

The generators write into `out/`:

```bash
for s in make_tab_adv make_tab_cost make_tab_ladder make_tab_rules make_fig_12; do env/bin/python paper_tables/$s.py; done
env/bin/python paper_tables/make_fig_skew.py results/paper results/original_corpora out/figures
```

### Content audit

The audit sends sampled documents to two language models through an OpenAI-compatible
chat-completions endpoint. Set three environment variables before running it:

- `LLM_GATEWAY`: the endpoint URL;
- `LLM_API_KEY`: the key;
- `LLM_PROXY`: optional proxy.

No key is stored in this repository. Model outputs vary between calls. The labels used in the
paper are in `data/snapshot2025/content_audit_labels.parquet`.

## 5. Tests

`operator/tests/test_operator.py` checks streaming B and capped B against literal B on random
buckets, exact W on its counterexample, and the 64-bit exact join against the vendored one.
`test_gpu_sim.py` (numba CUDA simulator) and `test_gpu_native.py` (MUSA or CUDA build) check the
GPU verification against literal B. `snapshot2025/post_reveal/e1_outside_group.py --smoke` runs the post-reveal pipeline on
a synthetic 3,000-document corpus.

## License

Code: MIT (see `LICENSE`). Files under `data/` describe Common Crawl documents and follow the
Common Crawl terms of use.
