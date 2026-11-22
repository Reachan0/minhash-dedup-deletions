# Audit summary — Draft 2 claims against the per-seed records

Step `audit-claims`, 2026-11-19. Draft 2
(`output/dedupjoin-pvldb/paper/`) was **not edited**. This summary is the
severity-ordered account of what the audit changed; the itemised evidence is in
`discrepancies.md`, the row-by-row ledger in `audit/claims_manifest.csv`
(86 claims), the vocabulary in `audit/glossary.md`.

Status at a glance: **86 claims audited — 65 confirmed, 11 corrected,
10 narrowed, 0 dropped; 24 naming/caption mismatches.** No claim is retracted;
every corrected number is replaced by one recomputed from the stored masks,
parquet metadata or per-seed JSONL.

## High severity — changes a definition or a scope

### H1. The paper's formal definition of the deletion set is wrong
`02-problem.tex:24` says each pipeline "deletes exactly the non-isolated
vertices". All three pipelines keep **one representative per connected
component** (minimum document id) and delete the rest. C4, τ=0.8: 5,123
non-isolated documents, 1,213 components, **3,910 deletions** — not 5,123. The
paper contradicts itself in `06-discussion.tex:23`, which states the min-id
rule. This is the definition of the quantity the whole paper measures, and it
must be fixed before any other edit.

### H2. The banding sweep covers two corpora, not four
`04-protocol.tex:19` says the sweep runs "over … all corpora at
τ∈{0.7,0.8,0.9}". Only **C4 and WET** have rows: `banding_sweep_rows.jsonl`
holds 120 rows = 2 corpora × 4 configurations × 3 thresholds × 5 seeds.
FineWeb and Mix were never swept. The table generator loops over four corpora
and silently skips absent ones (`code/make_tables_banding.py:41-47`), so the
rendered table hides the gap. Every banding conclusion must be scoped to C4
and WET.

### H3. "Verification residual" is raw-LSH missed deletions
`05-results.tex:42` reports verification's residual (missed exact deletions)
as 0, 13, 14, 109 (C4) and 0, 17, 20, 133 (WET). Those are `raw_missed`
(0.0, 12.8, 14.4, 108.6 and 0.0, 17.2, 19.6, 133.4). Verification was never run
for 20×5, 26×11 or 10×10; its `verified_missed` exists only for 14×8, where it
is **22.8** on C4. The same mislabel is in the `tab2_banding.tex:8` caption,
its generator (`code/make_tables_banding.py:31`) and the `banding_sweep.py:101`
field name (`missed`). This is the root of the 23-versus-13 question: **13 is
rounded `raw_missed` (12.8); 23 is rounded `verified_missed` (22.8); the gap is
the exact deletions raw deletes but verification keeps (10.0 per seed on
average, 12 on seed 1).**

## Medium severity — changes a headline number or a method description

### M1. 99.99% variance share has no valid denominator
`05-results.tex:73`, Figure 2's caption (`05-results.tex:86`) and
`01-introduction.tex:21` say one exact component carries 99.99% of the
variance. The empirical decomposition gives **99.9687% of the summed
per-component variance**, which is **100.0988% of the total** because the
covariance term is −0.13%. 99.99% matches neither denominator. Separately,
the 99.994% attributed to 873 shared documents is a share of the
all-or-nothing proxy `Σ_c E_c² p_c(1−p_c)`, not of the empirical variance; the
two must not share a sentence. The paper needs one named decomposition and one
number.

### M2. "Union-by-size" misdescribes the audited code
`04-protocol.tex:15` says the audited pipeline uses union-by-size clustering.
It uses **min-id union** (`lsh.py:117-136`); union-by-size is datatrove
0.6.0's stage 3 (`minhash.py:495-508`). Keep union-by-size only where the
official implementation is being compared.

### M3. The 0.05%/0.004% flip share omits its population
`07-conclusion.tex:3` says "0.05% of tokens and 0.004% of unique content ever
change status". That holds for the **689 of 5,123** documents that both have a
true neighbour and flip across seeds. Across all **18,529** flips the shares
are **1.14% of tokens and 0.31% of unique shingles**. The sentence needs "with
a true neighbour"; the 1.14% figure is already in `05-results.tex:75`.

### M4. Machine descriptions are not in the stored records
`04-protocol.tex:24` gives "15 threads on a 16-core machine with 48 GiB" for
the main comparison and "16 threads on 24 cores, 64 GiB, 58 GiB limit" for the
ladder. Only the ladder's 16 threads is recorded in a result file; the
main-comparison thread/core/RAM figures are not recoverable from
`main_comparison.json`. Memory peaks do confirm (3.97/6.69/5.21/1.66 GiB).
Either recover the run log or label the description as recorded outside the
result files.

### M5. `hijacked` is a mechanism sentence, not a definition
`02-problem.tex:28` defines `hijacked` as a false edge merging a document
under a lower-id document. The operational definition is
`|K_raw ∩ N(G_τ) ∩ ¬K_exact|`; the mechanism holds only under min-id union, and
datatrove's representative rule changes the count. The glossary fixes the
definition.

### M6. Synthetic "uniform sizes 0.98×" is a different regime, not an error
The uniform-size regime prints 0.98× at J=0.95 (`far`); the value 1.20× is the
uniform regime at J=0.85. Both are in `synthetic_variance.json`; the paper's
sentence is numerically correct but the two regimes are easy to conflate when
read across the paragraph. No change required; recorded so later steps do not
"fix" a correct number.

## Low severity — precision and units

- `01-introduction.tex:15`: "five configurations" → **four** banding
  configurations (14×8, 20×5, 10×10, 26×11); four methods.
- `01-introduction.tex:28`: "about nine LSH passes" at 1M is **9.5× (Mix) and
  11.6× (FineWeb)**; the 7.7–11.6× range in `05-results.tex:59` is correct.
- `04-protocol.tex:9`: largest C4-spanning component word range is **45–74**
  (not 49–74); largest WET component **1–4** (not 1–3).
- `04-protocol.tex:7`: the 147 URL-matched pairs were found over **8 of 1,024
  C4 shards** — a lower bound, not a crawl-wide count.
- `05-results.tex:59`: the "4.3 GiB / 12.7 GiB" earlier exact peaks are
  4,346 MB / 12,718 MB, i.e. **4.24 / 12.42 GiB**; the digits are decimal-GB
  conversions, the unit label is wrong.
- `05-results.tex:118`, `06-discussion.tex:21`: the rank assertion is on
  `V = S × 1.0476`; the shingle limit is 1.96×10^9, about **4.78M documents**
  (not 5.00M), and the memory fit gives 13.9M (not 14M). The qualitative point
  stands.
- `01-introduction.tex:13`: "missing edges leave their endpoints isolated" is
  stronger than the data — a missed edge can lie inside a component reached by
  other candidate edges. The measured quantity is edge recall (95.7% of 76,940
  pairs).
- `01-introduction.tex:18`, `07-conclusion.tex:3`: "no configuration removes
  the class" and the "45–77%" range are **C4 and WET only**; WET spans
  21.2–59.9%.
- `tables/tab6_methods.tex:6`: the "witness" column reports
  `1 − unsupported/|K_raw|` for raw LSH (0.317); witness coverage of `K_wit` is
  1.000 by construction. Rename the column or report the defined coverage.
- `results/paired_before_after.json` note: the official run used datatrove's
  `SpaCyTokenizer`, not NLTK. Internal metadata only, but it is the provenance
  of `04-protocol.tex:13`.

## What held up

The quantitative core of Draft 2 survives recomputation from the stored
records: the full main-table grid, the τ-independence of raw deletion counts,
the subset chain `K_wit ⊆ K_ver ⊆ K_exact ⊆ K_raw` (80/80 cells, 0
violations), the 25.8× overdispersion (34.6× at τ=0.7), the 13.4% at-risk flip
share and the 18,529 flip count, the 99.994% proxy for the 873 shared
documents, the risk-model AUC/AP and deciles, the official-comparison counts
(38,472 / 1,183 / 97.3%), tokenizer parity (+0.160% tokens, conditional
layers identical), downstream degradation (0.0009 bits/byte), the pool and
ladder figures, and the content-audit agreement rates. Details and the exact
denominators are in `discrepancies.md` §D.

## Provenance warnings that are not numerical errors but constrain claims

- **Mix is a renamed copy, not an independent corpus**: its first 500,000 rows
  are exactly FineWeb rows 0–499,999 and its second 500,000 are exactly C4
  documents 0–499,999 (byte-identical ids). It must be labelled as a
  concatenation.
- **WET is the same crawl as C4** (CC-MAIN-2019-18) at a different processing
  stage, not an independent snapshot; 147 URL pairs were found over 8 shards.
- **No independent validation corpus exists** among the four main-table
  corpora. `val_clean` is a held-out split of the same 700k C4 sample; the
  pairing set is a subset of WET. Cross-corpus contrasts measure processing
  stage and duplicate density, not independent replication.
- **Official comparison scope**: the exact truth is computed on the full
  175,272-document WET snapshot with the project tokenizer, while the official
  run sees 38,472 documents with datatrove's SpaCy tokenizer. The comparison
  is valid for the shared documents but is not an apples-to-apples tokenizer
  comparison.

## Files

| File | Contents |
|---|---|
| `audit/claims_manifest.csv` | 86 claims with printed/audited value, data file, generating command, population, status, evidence tier |
| `audit/naming_mismatches.csv` | 24 naming/caption mismatch locations |
| `audit/missed_reconciliation.csv` | 1,490 corpus × configuration × τ × seed cells with all four class counts |
| `audit/missed_summary.json` | cross-cell validation and the 23-vs-13 decomposition |
| `audit/number_provenance.json` | numerator/denominator/population for the five headline numbers |
| `audit/method_spec.md` | guaranteed-versus-measured for each pipeline; nine Draft 2 conflicts |
| `audit/provenance_overlap.json` | exact id/URL overlap, Mix construction, validation-set provenance |
| `audit/glossary.md` | frozen project terminology |
| `discrepancies.md` | append-only record of every difference, severity-grouped |

## Recommended edit order for the revision step

1. `02-problem.tex:24` (H1) — the definition every other number depends on.
2. `banding_sweep.py` field name + `tab2` caption + `05-results.tex:42` (H3) —
   one rename removes the entire 23-vs-13 family of confusion.
3. `04-protocol.tex:19` (H2) — scope of the banding claim.
4. Variance paragraph: one denominator, 99.97% within-component (M1).
5. `04-protocol.tex:15` union-by-size (M2).
6. `07-conclusion.tex:3` qualifier (M3).
7. Low-severity numbers and units in one pass.

Nothing in `output/dedupjoin-pvldb/` was modified; the audit reads it and the
per-seed records in `.work/dedupjoin/main/`, `.work/dedupjoin/pool_bands/` and
the result files under `output/dedupjoin-pvldb/results/` only.
