# Method specification — what each pipeline guarantees and what had to be measured

Definitions follow `audit/glossary.md`. "Guaranteed" means true for every input
under the stated representation and threshold, with no measurement needed;
"measured" means an empirical result on the named corpus that could have come
out differently.

## 1. Exact self-join (reference)

- **Graph**: `G_τ = (D, E_τ)`, `E_τ = {(d,e) : J(d,e) ≥ τ}` computed by the
  prefix-filter join in `artifacts/vendor/exactjoin.py` (`exact_join`:
  length filter, positional filter, global frequency order, threaded probe).
  Verified against a brute-force sparse product on 1,080 + 1,064 documents in
  `results/exact_validation_bruteforce.json` (310 edges, identical sets) and
  against constructed boundary cases in `results/correctness.json`
  (140 checks, 0 failures).
- **Who is kept per component**: `uf_components` links each component to its
  **minimum document id**; `removed_set(root) = root != arange` deletes every
  non-minimum member. One document per non-trivial component survives. It is
  *not* "all non-isolated vertices".
- **Guaranteed by definition**: `|K_exact| = |N(G_τ)| − #components with ≥2
  vertices`. Every vertex of a component is either its representative (kept)
  or deleted, so no true edge has both endpoints retained and the residual is
  0 for any component shape.
- **Measured**: component sizes and the number of components (`1213` on C4 at
  τ=0.8), cost of ranking/indexing/probing, peak memory.

## 2. Raw MinHash-LSH

- **Graph**: candidate graph from 14 bands × 8 rows (112 permutations).
  `band_keys` assigns each document one key per band; all pairs co-occurring in
  a bucket are candidate edges (`lsh_pairs` also adds consecutive chain edges
  between sorted document ids inside a bucket, which are a subset of the
  co-bucket pairs). `union-find` is run on these candidate edges with the same
  min-id rule.
- **Who is kept per component**: the minimum-id document of each candidate
  component. Every other member is deleted, whether or not it has a true
  neighbour.
- **Guaranteed by definition**: the deletion set is `K_raw`; the number of
  deletions is not a function of `τ` (the candidate graph does not depend on
  `τ`; only the downstream labelling does). In the measured cells `|K_raw|`
  is constant across all four thresholds.
- **Measured**: `unsupported` (deletions with no true neighbour), `hijacked`
  (deletions that the exact min-id rule would keep), `raw_missed`, pair recall,
  candidate precision, residual pairs, content retention, cost.
- **Not guaranteed**: any relation between `K_raw` and `K_exact`. The
  measured C4 τ=0.8 cell has `|K_raw|=10,397` against `|K_exact|=3,910`, with
  6,312 deletions that no true edge supports.

## 3. LSH + exact verification

- **Graph**: the subgraph of the candidate graph whose edges satisfy exact
  `J ≥ τ` (tolerance `τ − 1e-12` in `main_compare.py:333`). Union-find with the
  same min-id representative.
- **Who is kept**: the minimum-id document of each verified component.
- **Guaranteed by definition**:
  1. `K_ver ⊆ K_raw` (verified edges are a subset of candidate edges; a subset
     of unions cannot lower the representative of a component, and every
     deleted vertex stays non-minimum).
  2. `K_ver ⊆ K_exact`. Every verified edge is a true edge, so every verified
     component is contained in an exact component; the exact representative is
     the global minimum id. A document deleted by verification is non-minimum
     in its verified component, hence non-minimum in the containing exact
     component (whose minimum is ≤ the verified minimum), hence deleted by the
     exact rule. This is the subset identity, not an empirical finding.
  3. `unsupported(K_ver) = 0`: every deleted document has a verified (hence
     true) neighbour in its component. `ver_wrong` is 0 in every cell by
     construction.
  4. It cannot recover `raw_missed` or any exact deletion whose true edges were
     never candidates: those edges are not in the input graph.
- **Measured**: how many unsupported deletions are removed (all of them on
  every corpus, but the count differs), `verified_missed` (the residual),
  residual pairs, content retention (verification retains slightly more
  content than the exact join, because it deletes fewer documents), and cost
  (one Jaccard check per candidate edge; the verified pass costs about the same
  as raw LSH on the ladder).
- **Caveat for the official datatrove comparison**: the project's verified
  pipeline uses min-id representatives; datatrove stage 3 uses union-by-size
  and keeps the root of the largest set (`minhash.py:495-508`). The deletion
  *count* is one per component under either rule, but the identity of the kept
  document differs, and so does `hijacked`.

## 4. Witness rule

- **Graph**: the verified edge set. `greedy_mis(n, ei, ej)` visits documents
  `0..n−1`; a document is added to the independent set iff none of its
  verified neighbours is already in it. `K_wit` is the complement of that
  maximal independent set.
- **Who is kept**: every isolated document, plus the greedy MIS.
- **Guaranteed by definition**:
  1. `K_wit ⊆ K_ver`: the MIS is built on verified edges, and a document is
     deleted only when it has a kept verified neighbour; the min-id root of
     each verified component is always kept (it is visited first and has no
     kept neighbour when visited), so no verified component loses all members.
  2. Witness coverage 1.000: every deleted document has a retained neighbour
     at `J ≥ τ`. This is checkable from the removal list without recomputing
     the truth.
  3. `K_wit` has no residual true pair? No — the witness rule deletes fewer
     documents than verification, so true pairs can remain between two kept
     documents; residual pairs are measured, not guaranteed.
- **Measured**: `witness_missed` (exact deletions it gives up to keep the
  guarantee), residual pairs (44 at C4 τ=0.8 vs 19 for verification), content
  retention, and the fact that coverage is 1.0 in every cell (the construction
  proves it; the measurement confirms the implementation).

## 5. Guaranteed versus measured — summary

| Property | Status | Evidence |
|---|---|---|
| `K_ver ⊆ K_exact` | guaranteed | subset identity; `subset_check.json` 80/80 cells |
| `K_wit ⊆ K_ver` | guaranteed | greedy MIS construction; `subset_check.json` |
| `unsupported(K_ver) = 0` | guaranteed | every verified edge has `J ≥ τ` |
| witness coverage = 1 | guaranteed | every deletion has a kept verified neighbour |
| exact residual pairs = 0 | guaranteed | one representative per component |
| `unsupported(K_raw)` count | measured | 6,312 C4 τ=0.8; 773/4,289/2,423 other corpora |
| `hijacked(K_raw)` count | measured | 188 C4 τ=0.8; depends on min-id rule |
| `raw_missed` / `verified_missed` | measured | 12.8 / 22.8 C4 τ=0.8, five seeds |
| rescued documents by verification | measured | `|K_raw| − |K_ver|` = 6,510 C4 τ=0.8 |
| residual pairs / content loss / cost | measured | main table, Table 6, ladder |

## 6. Conflicts with Draft 2, sentence by sentence

1. `paper/sections/02-problem.tex:24`:
   > "The exact join deletes exactly the non-isolated vertices of `G_τ`; the
   > LSH pipeline deletes the vertices that are non-isolated in the graph
   > formed by its candidate edges after transitive closure, and the verified
   > pipeline deletes the non-isolated vertices in the graph formed by its
   > true candidate edges."

   Wrong for all three pipelines. Each keeps one representative per connected
   component (minimum id) and deletes the other members; isolated vertices are
   kept, but so is one vertex of every non-trivial component. C4 τ=0.8:
   5,123 non-isolated documents, 1,213 components, 3,910 deletions. The
   correct statement is in `paper/sections/06-discussion.tex:23` ("The exact
   join keeps the minimum-id document of each component"), so the paper
   contradicts itself. **Severity: high** — it is the formal definition of the
   audited decision.

2. `paper/sections/04-protocol.tex:15`:
   > "... stage-2 chain edges between consecutive documents in a bucket,
   > union-by-size clustering."

   The audited implementation is min-id union, not union-by-size
   (`lsh.py:117-136`). Union-by-size is what datatrove 0.6.0 does
   (`minhash.py:495-508`). The paper elsewhere analyses the min-id rule
   (`05-results.tex:33`: "components whose representative is not the longest
   document") and states it in `06-discussion.tex:23`. **Severity: medium** —
   it misdescribes the code being audited and blurs the official-versus-project
   comparison.

3. `paper/sections/02-problem.tex:28`:
   > "it happens when a false candidate edge merges it under a lower-id
   > document while it is itself the representative its exact component would
   > keep."

   The definition is correct only under the min-id rule. It is not the
   definition of `hijacked` in general and not the mechanism under datatrove's
   union-by-size. The operational definition is
   `hijacked = |K_raw ∩ N(G_τ) ∩ ¬K_exact|`; keep the mechanism sentence as an
   explanation, not as the definition. **Severity: low-medium**.

4. `paper/sections/04-protocol.tex:19`:
   > "The banding sweep ... runs four configurations ... over five seeds and
   > all corpora at τ∈{0.7,0.8,0.9}."

   Only C4 and WET were swept. FineWeb and Mix have no banding-sweep rows
   (`banding_sweep_rows.jsonl`: 120 rows = 2 corpora × 4 configurations ×
   3 thresholds × 5 seeds). The table generator loops over four corpora and
   silently `continue`s when a corpus is absent
   (`make_tables_banding.py:41-47`). **Severity: high** — it changes the
   scope of the banding conclusion.

5. `paper/sections/01-introduction.tex:15`:
   > "repeated over four corpora, four thresholds, five configurations, five
   > to 100 seeds"

   Four banding configurations (14×8, 20×5, 10×10, 26×11), not five. If the
   intended count is methods, there are four (raw, verify, witness, exact).
   **Severity: low**.

6. `paper/sections/05-results.tex:73` and `01-introduction.tex:21` and the
   Figure 2 caption ("99.99% ... carried by a single exact component"):

   The empirical quantity is 99.97% of the summed per-component variance,
   which is 100.1% of the total variance because covariance is −0.13%. It is
   not 99.99% of the total. **Severity: medium** — a headline number with the
   wrong denominator; see `number_provenance.json`.

7. `paper/sections/05-results.tex:42`:
   > "Verification removes the class under every configuration, and its
   > residual (missed exact deletions) is smallest where pair recall is high:
   > 0, 13, 14 and 109 ..."

   The quoted values are `raw_missed` from `banding_sweep.py:101`, not
   `verified_missed`; verification was never run for 20×5, 26×11 or 10×10.
   The first clause (verification removes the unsupported class) is guaranteed
   for every configuration, but the residual numbers are not verification
   residuals. **Severity: high** — this is the same mislabel as the tab2
   caption and is the source of the 23-versus-13 confusion.

8. `paper/sections/07-conclusion.tex:3`:
   > "0.05% of tokens and 0.004% of unique content ever change status"

   Correct only for documents that have a true neighbour. Across all flipping
   documents the shares are 1.14% of tokens and 0.31% of unique shingles. The
   sentence needs the qualifier. **Severity: medium**.

9. `paper/sections/05-results.tex:118`:
   > "The assertion fires at about 2.05×10^9 shingles, or five million
   > documents."

   The rank assertion is on `V` (shingles + repeated shingles), and
   `V/S = 1.0476` at 4M. The shingle limit is `2.05e9 / 1.0476 = 1.96e9`,
   about 4.78M documents, not 5.00M. The qualitative conclusion (just past
   the 4.07M pool) is unchanged. **Severity: low**.
