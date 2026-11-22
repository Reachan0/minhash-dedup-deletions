# Pre-registration — independent-snapshot replication of the MinHash-LSH deletion audit

- **Project:** `dedupjoin-pvldb-v3` (PVLDB Vol. 20, EA&B track)
- **Step:** `prereg-snapshot-novelty`
- **Freeze date (UTC):** 2026-11-19 (hash and UTC time recorded in `PREREG.sha256`)
- **Status at freeze:** corpus extracted and sharded; **no deduplication metric of any kind has
  been computed on this corpus.** At freeze time only document counts and text-length statistics
  exist (reported below). The exact-hash / URL overlap against the project pool is computed
  *after* this file is hashed and is provenance measurement, not a deduplication result.

This document fixes, before the outcome is known, what will be run on the new snapshot, how the
outcome will be measured, and what each possible outcome means for the paper's claims. Any
deviation after the freeze is recorded in `newsnap/PREREG_DEVIATIONS.md`, appended, never edited.

## 1. Snapshot and segment sample

| Item | Value |
|---|---|
| Snapshot | **CC-MAIN-2025-30** (July 2025 crawl; not used in earlier development; the pool used CC-MAIN-2019-18) |
| Segment list | `https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-30/wet.paths.gz` |
| Segment-list file | `newsnap/wet.paths.gz` |
| Segment-list sha256 | `b00996a663900fbb38df54c3dd29be792bdc58cb488401ed00bc59056cf53338` |
| Segments in list | 100,000 |
| Segments downloaded | **104** (ranks 0–103 of the seeded order) |
| WET records read | 2,456,751 |
| English-detected documents kept | **531,597** (21.6% of records) |
| Total compressed WET bytes | 7,418,462,506 B (7.42 GB) |
| Corpus size | 6 parquet shards, 1,252,283,980 B (1.25 GB), 522,207,916 words |

Per-segment source URL, compressed bytes, compressed sha256 and parquet sha256 are recorded in
`corpus/manifest.json`; download and extraction wall times are in `corpus/segment_stats.jsonl`.

### Sampling method (fixed before download)

The 100,000 segment paths were read in file order and shuffled with
`random.Random(20261119).shuffle` over the indices `0..99,999`. **Seed = 20261119.** The full
ordered plan is `sampling_plan.json` (all 100,000 ranks, each with index and path). Segments were
then processed in plan order in batches of 8; the run stopped after the first batch at which the
cumulative count of English-detected documents reached the target of 500,000. This is a
**sequential seeded without-replacement draw**: the stopping rule depends only on the language-ID
counts, never on any similarity or duplication quantity. 104 of 100,000 segments were consumed.

### Corpus shards

| File | Rows | Bytes | sha256 |
|---|---:|---:|---|
| shard-00000.parquet | 100,000 | 238,383,754 | `615d46e156e749e89fbcfcc833d243eeb682e729d5e8cb40145549731d0a470c` |
| shard-00001.parquet | 100,000 | 234,577,118 | `25690a668b5898f7d67ec4457d81d5514ce504adb62b4b2ee6f4eb4906f6e3d6` |
| shard-00002.parquet | 100,000 | 230,052,485 | `005703987d1d2080b530d70d8eaf1cae3fff8333c9b5bc831af9edd63759d511` |
| shard-00003.parquet | 100,000 | 236,561,775 | `0d84f7cda02a69c33247b2e439e15eeee214487c4a6304ddf80c5df251bc8b66` |
| shard-00004.parquet | 100,000 | 239,822,036 | `ae015a2aa9894c21ac874900f5436230afa57ecb6d921c3b7ad067677f1e23be` |
| shard-00005.parquet | 31,597 | 72,886,812 | `cd86ac62afa2bb0618703c6768804440c08c78d636a8783d5c87e1d01c35f140` |

Length statistics (words per document, project `nwords` field): mean 982.3, median 586, p10 12,
p90 2035, min 1, max 399,477. Texts were **not** truncated for the corpus; the 399k-word maximum
is a single pathological WET record and is retained (a preprocessing choice, stated here so it
cannot be changed after seeing results).

## 2. Document unit

**One WET conversion record = one document.** The text is the record's content as stored in the
WET file, decoded UTF-8, after the project extractor's existing cleanup only (the exact text
written to parquet is the unit; it is never further truncated or normalised for identity).
Duplicate documents are **kept** in the corpus: exact duplicates are part of the input whose
downstream handling is being measured, not something to remove at ingestion.

Language rule: fastText `lid.176.bin`, top-1 label `en` and score ≥ 0.8 — the same rule as the
project pool, so the new corpus is comparable to it.

Exact-hash identities used only for provenance overlap (not for any dedup claim):
`raw_hash = xxh64(UTF-8 text as stored)`, `norm_hash = xxh64(" ".join(text.lower().split()))`,
the definitions of `build_pool.py`.

## 3. Tokenization, shingles, hash family

- **Normalisation:** datatrove 0.6.0 `simplify_text` with the default `TextNormConfig`
  (lowercase, remove accents, normalise quotes/dashes/whitespace — the pipeline the audited
  systems use).
- **Tokenisation:** datatrove 0.6.0 `load_word_tokenizer("en")` → `spacy.blank("en")`
  rule-based word tokenizer (no statistical model).
- **Shingles:** contiguous **5-grams of word tokens** (datatrove `n_grams = 5`).
- **Shingle hash:** xxhash64 per shingle (`HashConfig(precision=64, hash_fc="xxhash")`,
  the datatrove 0.6.0 default).
- **Signature:** 112 permutations, drawn from the MinHash seed, `_mersenne_prime = 2^61 − 1`.
- **Banding (production setting):** **14 bands × 8 rows** = 112 hashes, the FineWeb /
  datatrove 0.6.0 setting.
- **Design target:** similarity **0.75**, per FineWeb §3.4 / §E.1 (the setting for which the
  14×8 banding was chosen).
- **Similarity for ground truth:** exact Jaccard over the 5-gram shingle sets,
  `J(A,B) = |S_A ∩ S_B| / |S_A ∪ S_B|`. The exact self-join at τ is the reference truth.

Band-inclusion probability: `P(s) = 1 − (1 − s^8)^14`.

| Banding | 50% point s | P(s=0.75) | P(s=0.80) |
|---|---:|---:|---:|
| **14×8 (this study)** | **0.6847** | **0.7716** | **0.9235** |
| 20×5 (Data-Juicer-style) | 0.5087 | 0.9956 | 0.9996 |
| 10×10 | 0.7631 | 0.4399 | 0.6789 |
| 26×11 (OLMo 3) | 0.7184 | 0.6744 | 0.9032 |

The other rows are context only; the audited production configuration is 14×8.

## 4. Thresholds τ

Two values, both pre-specified, both reported in full:

- **τ = 0.75** — the design target of the audited configuration.
- **τ = 0.80** — the operational threshold used by FineWeb and OLMo 3 (and the S-curve point at
  which 14×8 is usually deployed).

A deletion is *supported* at τ if the deleted document has at least one **retained** document with
exact Jaccard ≥ τ at the moment of deletion; otherwise it is **unsupported** (glossary in
`.work/dedupjoin-pvldb-v3/audit/glossary.md`). The retained set is the pipeline's output, so the
judgement is made against what survives, not against the pre-deletion corpus.

## 5. Seeds

- **≥ 5 MinHash seeds**, fixed set `{1, 2, 3, 4, 5}` (seed 1 is the datatrove default). The seed
  controls the 112 permutation coefficients and therefore the LSH banding; the corpus, tokenizer,
  shingles and τ are held fixed.
- The **official datatrove pipeline** is run under the same five seeds.
- The **exact join (pipeline C)** is seed-free and run once.
- Seeds are not selected or re-drawn after seeing results. If a seed fails technically, the
  failure and its cause are recorded and the seed is replaced by the next integer, with the
  replacement logged in `PREREG_DEVIATIONS.md`.

## 6. Pipelines

| ID | Pipeline | Verification | Role |
|---|---|---|---|
| **OFF** | Official datatrove 0.6.0 `MinhashDedupPipeline` (signature files → buckets → cluster → filter), 5-grams, 112 hashes, 14×8, union-find | none | the production system being audited |
| **A** | Project implementation: raw LSH buckets + union-find, keep minimum document id | none | reproduces the production decision rule inside the project's own code |
| **B** | Project implementation: A's candidate pairs, then **exact Jaccard verification at τ**, keep minimum id among verified neighbours | exact, all candidate pairs | the repair being evaluated |
| **C** | Full exact self-join at τ (prefix filtering over exact shingle sets), keep minimum id | exact, all pairs | ground truth |

The subset relation `K_wit ⊆ K_ver ⊆ K_exact` from the audit glossary must hold for every seed;
a violation is a pipeline bug and stops the analysis for that seed.

## 7. Metrics (frozen definitions)

Let `K_raw` be the documents deleted by the audited pipeline, `K_exact` those deleted by the exact
join, `n` the corpus size, and `N(G_τ)` the documents having at least one retained exact-τ
neighbour.

1. **Unsupported share of raw deletions:** `|K_raw ∩ ¬N(G_τ)| / |K_raw|`.
2. **Unsupported share of all input:** `|K_raw ∩ ¬N(G_τ)| / n`.
3. **Verified extra cost ratio:** cost(B) / cost(A), reported **cold** (whole corpus from scratch)
   and **incremental** (corpus grows by fixed 100k-document increments; cost of processing only
   the new increment plus its candidate pairs against the accumulated state). Wall time and CPU
   seconds, threads stated, cache state stated.
4. **Pair-level vs document-level agreement:** pair recall `|E_LSH ∩ E_τ| / |E_τ|` (candidate
   edges at τ) against document-level deletion recall `|K_raw ∩ K_exact| / |K_exact|`. Both are
   reported for every seed and τ, with the difference.
5. **Cluster-equal-weight variant:** the unsupported share computed with each connected component
   weighted equally (components are the LSH/union-find deletion clusters), so one giant component
   cannot dominate the aggregate.
6. **Drop-largest-cluster variant:** the same metrics after removing the single largest connected
   component.
7. **Coverage / witness metrics from the frozen glossary:** `raw_missed`, `verified_missed`,
   `hijacked`, witness coverage, residual duplicate pairs — reported alongside (1)–(6) so the new
   snapshot is directly comparable with Draft 2.

Uncertainty: 1,000 document-level bootstrap resamples for shares (1) and (2); 1,000
*cluster-level* bootstrap resamples for the cluster-weighted variant; 95% intervals reported.
Length-decile strata are pre-specified secondary cuts.

## 8. Decision table (fixed before results)

| Outcome (τ = 0.80 unless stated) | Reading | Consequence for the paper |
|---|---|---|
| Unsupported share of raw deletions **≥ 10%**, consistency across ≥ 4 of 5 seeds, and same direction at τ = 0.75 | The effect is systematic and not tied to the development corpora | Keep the "production MinHash-LSH systematically deletes unsupported documents" framing; the new snapshot becomes the replication evidence; "universal" may be used with the corpus family named |
| **5–10%** | Material but corpus-dependent | Report as a robust but conditional phenomenon; remove "universal"; condition every headline number on the corpus and configuration |
| **2–5%** | Small but non-zero | The claim becomes "non-trivial residual unsupported deletions"; the paper's emphasis shifts to the cost/benefit of verification and the concentration of the effect |
| **< 2%** | The effect does not replicate at scale on a raw 2025 crawl | Withdraw the general claim; report the development-corpus results as corpus-specific, state the negative replication prominently, and re-examine which corpus property (boilerplate, crawl duplication, length distribution) drives the earlier numbers |
| Effect present at τ = 0.75 but absent at τ = 0.80 | Threshold-driven, not banding-driven | Attribute the phenomenon to the deployed threshold rather than to LSH chaining; revise the mechanism discussion |
| `abs(pair_recall − doc_recall) ≤ 0.05` | Pair- and document-level views agree | Report once, as "the same judgment at both granularities" |
| `abs(pair_recall − doc_recall) > 0.10` | The granularity changes the conclusion | Report both views separately; the paper must not present pair-level recall as a proxy for retention |
| Pipeline B leaves **> 5%** of exact deletions unverified (`verified_missed / n_exact > 0.05`, where `n_exact` is the size of `K_exact`) | Candidate generation, not verification, is the bottleneck | The "verification suffices" claim is withdrawn; the paper recommends candidate-recall improvements |
| B leaves **< 1%** | Verification suffices in practice | Keep the "exact verification of candidates closes the gap" recommendation, with the residual quantified |
| Incremental cost ratio **> 2×** cold ratio | Verification is disproportionately expensive incrementally | The cost recommendation is conditioned on batch (not streaming) operation |
| Cluster-equal-weight share **> 2×** the document-weighted share, or drop-largest changes the sign | The aggregate is driven by one component | All headline numbers reported both with and without the largest component; concentration discussed explicitly |
| The effect vanishes (unsupported < 1% at both τ) | Strong negative replication | The paper reports it as the main finding of the replication section; no "universal" wording anywhere; Draft 2 numbers kept side by side in `discrepancies.md` |

Threshold directions are fixed here. Intermediate values are reported as observed, without moving
the bands.

## 9. Known deviations from the work order (recorded now, before results)

1. **Segments needed:** the work order estimated ~44k records/segment and 12–20 segments. The
   measured CC-MAIN-2025-30 WET files average 23,623 records and 5,112 English-detected documents
   per segment; reaching 500k therefore required 104 segments.
2. **Download route:** `vela-data get` was measured at ~1.3 MB/s (≈90 min for the required
   volume); direct `curl` from `data.commoncrawl.org` ran at ~22 MB/s. The data contract permits
   direct download when the library route exceeds ~10 min; the reason is recorded in `DATA.md`.
3. **Raw WET `.gz` files are deleted after extraction**; the corpus keeps text, per-segment
   source bytes and sha256. Re-download is possible from the recorded URLs.
4. **No text truncation** at ingestion (see §1).

## 10. Not pre-registered (exploratory)

The following may be computed later but are **not** part of the confirmatory analysis and will be
labelled exploratory wherever reported: length-decile mechanism cuts beyond the pre-specified
secondary cuts, per-domain analysis, the BigCode independent baseline, the OLMo-style
cluster-only verification variant, and any sensitivity to banding other than 14×8.

## 11. Freeze

The sha256 of this file and the UTC timestamp are written to `newsnap/PREREG.sha256`. From that
moment the analysis scripts may compute deduplication metrics; nothing above may be changed
retroactively.
