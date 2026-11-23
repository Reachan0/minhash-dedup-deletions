# Experiment E operators

`stream_b.py` implements Path B as a bucket stream. It sorts one band at a time,
passes one bucket's document ids to a numba nested-pair kernel, and keeps only
the union-find state. A pair whose endpoints already have the same root skips
Jaccard. Roots are chosen by minimum document id, so the deletion mask is the
non-minimum member of every final component. Repeated pairs across bands are
safe and are counted when encountered. The implementation uses a star prefix
for a bucket and stops once all of its documents are already in one verified
component. This is the same skip induction: after every emitted true edge the
current roots are exactly the components of the verified prefix. A skipped
pair lies inside one such component and cannot change the final equivalence
relation. Every unskipped pair joining different roots is exact-checked, so the
final components equal B's graph components.

`exact_w.py` builds a bucket-membership CSR and processes documents in id order.
For each document it scans earlier members of its buckets, deduplicating a pair
across bands, and stops at the first earlier kept document whose exact Jaccard
passes. It does not use the B root skip. Empty documents have no candidates and
are retained.

`stream_b_cap` limits exact checks to `q` per bucket in deterministic pair order.
Every emitted edge is exact, hence the result is sound. If `U` is the reported
number of unchecked candidates, the document-level deviation from B obeys
`extra retained documents vs B <= true edges among unchecked pairs <= U`.

`collapse.py` hashes sorted shingle arrays for indexing, compares arrays within
each digest group, and splits every collision. Empty sets are singleton groups.
`lift_b` and `lift_c` expand minimum-id component results. `lift_w` treats each
equal-set group as a block at its minimum id. If the minimum is kept, later
copies are deleted. If it is deleted, the whole block has an earlier witness.

`exactjoin64.py` keeps int64 offsets and selects uint32 ranks while the distinct
rank universe fits, otherwise int64. The vendor verification kernel is reused
without its 2^31 rank assertion. `identical_edge_set` compares the edge sets on
an input under the vendor cap.

The worst case remains quadratic in a bucket of distinct near-duplicates. A
stream does not materialise the quadratic pair array, but it cannot generally
avoid exact tests that distinguish two possible component graphs. Collapse
helps exact copies and does not remove this distinct-near-duplicate lower bound.

## Commands

Run the synthetic checks with the supplied environment.

```bash
cd /path/to/home/dedupjoin-postreveal
env/bin/python - <<'PY'
import sys
sys.path.insert(0, 'dedupjoin-newsnap-selection/code/operator')
import tests.test_operator as t
for name in ('test_w_three_document_counterexample',
             'test_distinct_near_duplicate_bucket_and_cap_bound',
             'test_exact_copy_groups_and_lifting',
             'test_exactjoin64_matches_vendor'):
    getattr(t, name)()
    print(name, 'ok')
PY
```

After `logs/e123.log` contains `E3_DONE`, run the no-collapse 531,597-document
campaign. It checks the existing `ver_s{seed}_t{tau}.npz` masks for all five
seeds and both thresholds and appends one JSON row per run to
`results_post_reveal/operator_E.jsonl` in the workspace. The rows used in the paper are in
`results/snapshot2025_post_reveal/operator_E.jsonl`.

```bash
env/bin/python dedupjoin-newsnap-selection/code/operator/tests/run_snapshot.py
```

The snapshot harness records signature and streaming wall time, peak RSS,
enumerated, skipped, verified, accepted, and unchecked counters.
