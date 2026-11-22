# Frozen terminology — DedupJoin audit

This file is the project-wide definition source for later steps. Every symbol is
defined against the stored per-seed deletion masks; no definition is taken from a
caption or a paragraph. Code locations are relative to the workspace root.

## 1. Representation and truth graph

- `S(d)`: set of distinct 5-gram shingles of document `d` after the shared
  simplification (lower-case, numbers to `0`, punctuation removed, whitespace
  collapsed; non-ASCII diacritics stripped). Project tokenizer splits on spaces;
  datatrove 0.6.0 uses `SpaCyTokenizer` (`utils/word_tokenizers.py`,
  `assets/tokenizer_assignment.csv` row `eng,en,...,SpaCyTokenizer,en`).
- `J(d,e) = |S(d) ∩ S(e)| / |S(d) ∪ S(e)|`.
- `E_τ = {(d,e) : J(d,e) ≥ τ}`; `G_τ = (D, E_τ)`.
- `N(G_τ) = {d : deg_τ(d) ≥ 1}` — documents with at least one true neighbour.
- `K_exact = |K_exact| = removed_set(uf_components(E_τ))` — the exact-join
  deletion set. Because `uf_components` links every component to its
  **smaller document id** and `removed_set(root) = root != arange`
  (`.work/vldb2027-ai-db-ideas/experiment/src/C03/lsh.py:117-136`), exactly one
  document per non-trivial component is kept: the minimum id. Therefore
  `|K_exact| = |{d : deg_τ(d) ≥ 1}| − #components(≥2)`, **not** the number of
  non-isolated vertices. C4, τ=0.8: 5,123 − 1,213 = 3,910.
- `K_raw = raw LSH deletion set`: union-find over the candidate edges emitted by
  the 112-permutation, 14×8 banding (`main_compare.py:299-392`), again min-id
  representative.
- `K_ver = verified LSH deletion set`: union-find over candidate edges whose
  exact `J ≥ τ − 1e-12` (`main_compare.py:333-336`).
- `K_wit = witness deletion set`: complement of a greedy maximal independent set
  over the **verified** edges, visited in increasing document id
  (`main_compare.py:174-204`). A document is deleted only if it has an adjacent
  kept document in the verified graph. Isolated documents are kept.

Stored masks: `.work/dedupjoin/main/{corpus}_s{seed}_t{tau}_sets.npz`
(`rem_raw`, `rem_ver`, `rem_wit`, `rem_exact`, bit-packed uint8, length `n`);
truth edges: `{corpus}_truth_t{tau}.npz` (`i`, `j`, `jac`, `root`, `removed`).

## 2. The four deletion classes

All counts below are per corpus × configuration × threshold × seed, over the
`n` documents of that corpus. "Population" names the denominator population,
not the denominator of the count itself (the count is a count).

| Name | Formula | Population | Definition in code |
|---|---|---|---|
| `unsupported` | `\|K_raw ∩ ¬N(G_τ)\|` | all `n` documents; reported as a count, and as a share of `\|K_raw\|` or of `n` | `main_compare.py:340` (`wrong_raw = rem_raw & ~has_nb`); `banding_sweep.py:98` |
| `hijacked` | `\|K_raw ∩ N(G_τ) ∩ ¬K_exact\|` | all `n` documents | `main_compare.py:341` (`hijack = rem_raw & has_nb & ~rem_ex`); `banding_sweep.py:99` |
| `raw_missed` | `\|K_exact ∩ ¬K_raw\|` | all `n` documents | `main_compare.py:359`; `banding_sweep.py:101` (there named `missed`) |
| `verified_missed` | `\|K_exact ∩ ¬K_ver\|` | all `n` documents | `main_compare.py:362` (`missed_ver`); synonymous with `lshv_missed_removals` (`rerun_seeds.py:100`, identity `lshv_removed + lshv_missed_removals == exact_removed`) |
| `witness_missed` | `\|K_exact ∩ ¬K_wit\|` | all `n` documents | `main_compare.py:363` (`wit_missed`) |

Notes that matter when reading Draft 2:

1. `K_ver ⊆ K_raw` and `K_ver ⊆ K_exact`, and `K_wit ⊆ K_ver`. The inclusions are
   constructive (each pipeline unions a subset of the previous pipeline's
   edges under the same min-id rule). Verified empirically in all 80 main
   cells (`subset_check.json`: 0 violations) and in every stored mask set.
   Consequently `verified_missed ≥ raw_missed` per seed, and
   `verified_missed − raw_missed = |K_exact ∩ K_raw ∩ ¬K_ver|`, i.e. exact
   deletions that raw deletes but verification keeps.
2. `unsupported` counts documents deleted by **raw** LSH only. Verification
   cannot produce an unsupported deletion: every verified edge satisfies
   `J ≥ τ`, so every non-kept endpoint has a true neighbour in its verified
   component. `ver_wrong` is numerically 0 in every cell
   (`main_compare.py:364`); that is a theorem, not a measurement result.
3. `hijacked` is defined **relative to the exact join's min-id representative**.
   It means: raw LSH deletes the document, the document has a true neighbour,
   and the exact join keeps it. It is not "merged under a lower-id document"
   in general; that phrase describes the common mechanism, not the definition.
   A different representative rule (datatrove's union-by-size) changes which
   document is kept and therefore changes this count.

## 3. Derived ratios and other named quantities

| Name | Numerator / denominator | Source |
|---|---|---|
| `unsupported share of deletions` | `unsupported / \|K_raw\|` | `main_compare.py` aggregate; `banding_sweep.py:114` |
| `unsupported share of input` | `unsupported / n` | `banding_sweep.py:115` |
| `deletion precision` (method m) | `1 − unsupported(m) / \|K_m\|`; only defined when the method has a deletion policy. For `+verify` and `witness` it is 1 by construction | `make_tables_step.py:93,100` |
| `pair recall` | `\|E_τ ∩ candidate edges\| / \|E_τ\|`; denominator `\|E_τ\|` (all true pairs), not `\|K_exact\|` | `main_compare.py`; `banding_sweep.py` |
| `candidate precision` | `\|E_τ ∩ candidate edges\| / \|candidate edges\|` | `main_compare.py` |
| `witness coverage` | share of `K_wit` whose every document has a kept neighbour in the verified graph. Greedy MIS construction gives 1.000 by construction; `witness_share.wit = 1.0` in all cells | `main_compare.py`; `aggregate_main.py` |
| `residual duplicate pairs` (method m) | `#{(d,e) ∈ E_τ : d ∉ K_m and e ∉ K_m}` — true pairs with both endpoints retained | `main_compare.py:344-346` |
| `retained tokens/content` | tokens (or unique shingles) of `D \ K_m` divided by those of `D` | `main_compare.py` |
| `overdispersion` | observed seed-to-seed sd of pair recall / binomial SE `sqrt(p(1−p)/\|E_τ\|)` | `seed_variance.py`; `variance_decomposition.py` |

`witness coverage` is **not** "share of deletions with a true neighbour"; that
quantity is `1 − unsupported/|K_raw|` for raw LSH and `1.0` for verification.

## 4. Words that may not be used loosely

- **error**: only relative to an explicitly named deletion policy. This audit
  uses "unsupported", "hijacked", "raw_missed", "verified_missed"; it does not
  say "wrong deletion" without naming the policy.
- **missed**: always qualified (`raw_missed` or `verified_missed`). The bare
  word `missed` appears in three incompatible senses in Draft 2:
  `banding_sweep.py` / `tab2_banding.tex` (raw_missed), `make_tables_step.py`
  tab6 (per-method `exact_removed × (1 − deletion_recall)`, i.e. missed
  removals), and the tab7 caption (verified_missed). Do not reuse it unqualified.
- **residual**: in tab6/tab7 it means missed exact deletions
  (`verified_missed`); in the residual-pairs column it means true pairs left in
  the retained set. Keep the two apart: `verified_missed` versus
  `residual duplicate pairs`.

## 5. Values frozen for later steps (C4, 14×8, τ=0.8, five main seeds)

- `|K_raw| = 10,397.0 ± 129.8`
- `unsupported = 6,311.8 ± 129.0` (60.71% of deletions, 0.902% of input)
- `hijacked = 188.0 ± 4.0`
- `|K_exact| = 3,910` (seed-independent)
- `verified_missed = 22.8 ± 3.3` (5 seeds)
- `raw_missed = 12.8 ± 2.6` (same 5 seeds; 100-seed mean 11.11 ± 3.32)
- `witness_missed = 235.8 ± 4.4`
- `residual pairs`: raw 10.2, verified 18.6, witness 43.6, exact 0

These are the numbers the 23-versus-13 question is about: **13 is the rounded
raw_missed (12.8), 23 is the rounded verified_missed (22.8)**, and the gap is
the exact deletions raw deletes but verification keeps (10.0 per seed on
average, 12 on seed 1).
