# Pre-registration — deployment-time pipeline selection rules, frozen before the 2025-30 reveal

- **Project:** `dedupjoin-pvldb-v3`
- **Freeze (UTC):** recorded in `SELECTION_PREREG.sha256` next to this file's sha256.
- **Snapshot to be revealed:** the corpus already frozen in `newsnap/PREREG.md` —
  CC-MAIN-2025-30, 531,597 English WET documents, 6 parquet shards, mean 982.3 words per
  document, sha256 per shard in `PREREG.md`. **No deduplication metric of any kind has been
  computed on this corpus.** The only snapshot-level quantities that exist are document
  counts, word-length statistics (`PREREG.md`), and exact-hash / URL overlap against the
  project pool (`overlap_vs_pool.json`, provenance, not a similarity result).
- **Purpose:** fix, before seeing any snapshot deduplication result, what a deployer in three
  use scenarios should choose among paths A (raw LSH chain), B (LSH candidates + exact
  verification), W (direct-witness deletion) and C (full exact self-join), including the
  quality tolerance, cost target, parameters, and the predictions that will be checked at the
  reveal.

## 1. Results already viewed before this freeze

| Viewed | Scope |
|---|---|
| Core benchmark accuracy | `core/runs.jsonl`: A/B/W/C on pool workloads C4 700k, FineWeb 1M, Mix 1M, WET 175k; τ ∈ {0.8, 0.75}; 3–5 hash seeds. Deletions, unsupported, raw/verified/witness missed, hijacked, residual pairs, witness coverage. |
| Core benchmark timing | `core/timing.jsonl`: cold and incremental, 12 threads and 1 thread, fresh OS process per configuration, on the same four workloads. |
| Draft 2 | published numbers on the pool (read-only baseline, `output/dedupjoin-pvldb/`). |
| 2025-30 snapshot | counts, word-length statistics, exact-hash/URL overlap only. No shingle, signature, band, pair, cluster or deletion quantity. |

The cost predictions below are fitted on the viewed pool timings (`newsnap/code/
selection_prereg_predict.py` → `newsnap/selection_predictions.json`). This is legitimate
pre-deployment knowledge: it is the benchmark the deployer would have run on its own corpora
before facing the new snapshot.

## 2. Fixed configuration (identical for every scenario)

- Normalisation `simplify_text`, word tokenizer, 5-gram shingles, xxhash64 (datatrove 0.6.0
  definitions, as in `PREREG.md` §3).
- MinHash 112 permutations, **14 bands × 8 rows**, value seed 1 for the primary run; accuracy
  checked with seeds {1,…,5}.
- τ = 0.80 primary, τ = 0.75 secondary; exact Jaccard over shingle sets.
- Representative: minimum document id (deployment default). The datatrove union-by-size
  representative is reported separately as a sensitivity, not used for the rule.
- 12 threads; one fresh OS process per timed configuration; timing boundary as in the core
  benchmark (cold: text → shingles → signatures/bands or ranks → method; incremental: cached
  shingles + signatures).

## 3. The three scenarios and the frozen rule

Predictions are medians of the linear model in total shingle count, with the fit residuals as
the stated uncertainty. The snapshot has S ≈ 519.9 million 5-grams.

### S1 — every deletion must be auditable (direct witness is a hard constraint)

| Item | Frozen value |
|---|---|
| Hard constraint | Every deleted document has a **retained direct exact neighbour** at τ (witness coverage = 1.0); unsupported deletions = 0 by construction |
| Quality tolerance | Residual duplicate pairs in the retained set ≤ 0.1% of input (≤ 532 documents); verified_missed ≤ 5% of the exact deletion set |
| Cost target | Cold end-to-end ≤ 400 s; incremental (cached shingles + signatures) ≤ 20 s; peak RSS ≤ 16 GiB |
| **Recommended path** | **W** (B's candidate expansion and verification, deletion by greedy independent set over the verified graph) |
| Parameters | τ = 0.80; 14×8; 112 permutations; min-id; 12 threads |
| Predicted cost | Cold ≈ 190 s, incremental ≈ 10.0 s, peak RSS ≈ 8.9 GiB |
| Why not the others | A has no witness guarantee; B deletes whole verified components, so a deleted document's retained neighbour can be absent when the component is a path/star; C is exact but its deletion set is not a witness set and costs ≈ 573 s cold / 11.6 GiB |

### S2 — memory-limited serving (peak RSS ≤ 6 GiB, incremental ingestion)

| Item | Frozen value |
|---|---|
| Hard constraint | Peak RSS ≤ 6 GiB (6144 MiB) with cached shingles and signatures; unsupported deletions = 0 |
| Quality tolerance | verified_missed ≤ 5% of the exact deletion set; residual duplicate pairs ≤ 1% of input |
| Cost target | Incremental ≤ 30 s |
| **Recommended path** | **B** |
| Parameters | τ = 0.80; 14×8; 112 permutations; min-id; 12 threads |
| Predicted cost | Incremental ≈ 9.9 s, peak RSS ≈ 5.24 GiB |
| Why not the others | A fails the unsupported = 0 tolerance; C's incremental state (exact-join ranks) needs ≈ 6.48 GiB, above the budget, and 446 s to build if absent; W costs the same as B while adding a guarantee this scenario does not require |
| If the prediction fails | If measured B peak exceeds 6 GiB, no path in {A,B,W,C} satisfies both the memory and the unsupported = 0 tolerances; the pre-registered recommendation is then "raise the budget to the measured W/B peak (predicted ≤ 9 GiB cold) or accept A's unsupported share", not a silent switch to A |

### S3 — signature cache already exists (the preprocessing has been paid for)

| Item | Frozen value |
|---|---|
| Hard constraint | Reuse cached signatures and bands; no signature recomputation; unsupported deletions = 0 |
| Quality tolerance | verified_missed ≤ 5% of the exact deletion set; residual duplicate pairs ≤ 1% of input |
| Cost target | Incremental ≤ 15 s |
| **Recommended path** | **B** |
| Parameters | τ = 0.80; 14×8; 112 permutations; min-id; 12 threads |
| Predicted cost | Incremental ≈ 9.9 s, peak RSS ≈ 5.24 GiB |
| Why not the others | A fails unsupported = 0; C at 6.25 s would be cheaper but needs the exact-join ranks, which are not in a signature-only cache and cost ≈ 446 s to rebuild; W is B plus a guarantee not required here |
| Note | If the policy were changed to accept A's unsupported deletion share, A at ≈ 7.0 s would be the cheapest qualifying path; that policy is not this scenario |

### Boundary case (recorded, not recommended)

If a deployment needs a **closed exact answer** and can pay for it, C is the reference: predicted
573 s cold, 11.6 GiB, incremental 6.25 s **only when the rank cache exists**. Its incremental
figure must never be quoted without the 446 s rank-build cost that precedes it.

## 4. Cheap sampling plan (cost counted, not used to change the rule)

Before the full reveal, a 2,000-document sample is drawn by evenly spaced row indices
(`floor(k·n/2000)`, k = 0…1999; deterministic, no similarity involved). For the sample:
shingles, 112-permutation signatures (seed 1), band buckets, candidate pairs, largest bucket.
The sample costs at most ~2 s and ~0.1 GiB (measured value recorded at the reveal). It is used
only as a diagnostic: if sampled candidate pairs per document exceed 10× the pool C4 rate
(0.19), the reveal report must flag that B's bucket expansion may exceed the prediction, and a
small-scale exact audit is recommended before deployment. The frozen rule in §3 does not
change with the sample.

## 5. What the reveal must report

1. Whether the recommended path satisfies every hard constraint and quality tolerance of its
   scenario, on the snapshot, for seeds {1,…,5} (W/B) or once (C).
2. The **measured cheapest qualifying path** under each scenario's constraints, compared with
   the recommendation, and the cost of a wrong choice:
   - recommending B when W was needed → the share of B's deletions lacking a retained direct
     exact neighbour;
   - recommending A when justification was required → unsupported share of deletions and of
     all input, and residual duplicate pairs;
   - recommending C when it was not needed → excess cold seconds and GiB over B.
3. Predicted vs measured cold wall time, incremental wall time, peak RSS, deletions,
   unsupported share, residual pairs and witness coverage for every path.
4. The sampling diagnostics and their measured cost.
5. If the snapshot's duplicate density is so low that several paths have identical deletion
   sets within seed noise, say so and report that the choice is cost-driven only; if it is too
   high for B's expansion to stay under the cost target, recommend a small exact audit first,
   as pre-registered here.

## 6. Freeze

Any deviation after the hash is written goes to `newsnap/SELECTION_PREREG_DEVIATIONS.md`,
appended, never edited. From the hash onward the snapshot may be analysed.
