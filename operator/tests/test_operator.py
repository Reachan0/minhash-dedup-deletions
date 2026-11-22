import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from collapse import collapse, lift_b, lift_w  # noqa: E402
from exact_w import exact_w  # noqa: E402
from stream_b import stream_b, stream_b_cap  # noqa: E402


def _keys(n, bands=1):
    return np.zeros((bands, n), dtype=np.uint64)


def test_w_three_document_counterexample():
    # A--B and B--C pass at .8, A--C does not.
    sets = [np.arange(10), np.arange(1, 11), np.arange(2, 12)]
    flat = np.concatenate(sets).astype(np.uint64)
    off = np.array([0, 10, 20, 30], dtype=np.int64)
    w = exact_w(off, flat, _keys(3), 0.8)
    assert w["deleted"].tolist() == [False, True, False]
    b = stream_b(off, flat, _keys(3), 0.8)
    assert b["deleted"].tolist() == [False, True, True]
    assert b["skipped_pairs"] >= 0


def test_distinct_near_duplicate_bucket_and_cap_bound():
    n, m = 13, 40
    sets = [np.arange(m) + i for i in range(n)]
    flat = np.concatenate(sets).astype(np.uint64)
    off = np.arange(0, (n + 1) * m, m, dtype=np.int64)
    out = stream_b(off, flat, _keys(n), 0.8)
    assert out["enumerated_pairs"] == n * (n - 1) // 2
    capped = stream_b_cap(off, flat, _keys(n), 0.8, 3)
    # two star passes are exempt from the cap
    assert capped["verified_pairs"] <= 2 * (n - 1) + 3
    assert capped["unchecked_pairs"] > 0
    assert 0 < capped["extra_retained_bound"] <= capped["unchecked_pairs"]
    assert np.all(~capped["deleted"] | (off[1:] > off[:-1]))


def test_exact_copy_groups_and_lifting():
    a = np.array([1, 2, 4], dtype=np.uint64)
    b = np.array([8], dtype=np.uint64)
    flat = np.concatenate([a, a, b, np.zeros(0, np.uint64)])
    off = np.array([0, 3, 6, 7, 7], dtype=np.int64)
    z = collapse(off, flat)
    assert z["n_groups"] == 3
    assert z["group_size"].tolist() == [2, 1, 1]
    root = np.arange(3, dtype=np.int64)
    root[1] = 0
    assert lift_b(root, z["group_of_doc"], z["group_rep"]).tolist() == [False, True, True, False]
    kept = np.array([True, True, True])
    assert lift_w(kept, z["group_of_doc"], z["group_rep"]).tolist() == [False, True, False, False]


def test_exactjoin64_matches_vendor():
    from exactjoin64 import identical_edge_set
    sets = [np.array([1, 2, 3, 4], np.uint64), np.array([1, 2, 3, 5], np.uint64),
            np.array([8, 9], np.uint64)]
    flat = np.concatenate(sets)
    off = np.array([0, 4, 8, 10], dtype=np.int64)
    same, ours, theirs = identical_edge_set(off, flat, tau=0.5, threads=1)
    assert same
    assert set(zip(ours["i"], ours["j"])) == {(0, 1)}


def test_snapshot_masks_if_available():
    """The full snapshot test is also exposed as run_snapshot.py."""
    if os.environ.get("RUN_OPERATOR_SNAPSHOT") != "1":
        return
    import snap_common
    from stream_b import candidate_keys
    off, flat, _ = snap_common.load_snapshot()
    mask_dir = os.path.join(os.path.dirname(ROOT), "..", "newsnap", "cache", "masks")
    for seed in range(1, 6):
        keys = candidate_keys(off, flat, seed)
        for tau in (0.8, 0.75):
            got = stream_b(off, flat, keys, tau)["deleted"]
            expected = np.asarray(np.load(os.path.join(mask_dir, f"ver_s{seed}_t{tau}.npz"))["rem"])
            assert np.array_equal(got, expected), (seed, tau, int(np.sum(got != expected)))


def _literal_b(off, flat, keys, tau):
    n = len(off) - 1
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for band in range(keys.shape[0]):
        for k in np.unique(keys[band]):
            docs = np.flatnonzero(keys[band] == k)
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    a, b = docs[i], docs[j]
                    sa, sb = set(flat[off[a]:off[a + 1]].tolist()), set(flat[off[b]:off[b + 1]].tolist())
                    if sa | sb and len(sa & sb) / len(sa | sb) >= tau:
                        ra, rb = find(a), find(b)
                        parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(x) for x in range(n)])
    return (root != np.arange(n)) & (off[1:] > off[:-1])


def test_random_buckets_star_variants_match_literal_b():
    rng = np.random.default_rng(3)
    for trial in range(30):
        n = int(rng.integers(5, 40))
        bases = [rng.choice(200, 20, replace=False) for _ in range(3)]
        sets = []
        for d in range(n):
            s = bases[int(rng.integers(0, 3))].copy()
            s[rng.choice(20, int(rng.integers(0, 5)), replace=False)] = rng.integers(1000, 5000, 4)[:1]
            sets.append(np.unique(s))
        flat = np.concatenate(sets).astype(np.uint64)
        off = np.r_[0, np.cumsum([len(s) for s in sets])].astype(np.int64)
        keys = rng.integers(0, 3, size=(3, n)).astype(np.uint64)
        ref = _literal_b(off, flat, keys, 0.8)
        for stars in (1, 2):
            assert (stream_b(off, flat, keys, 0.8, stars=stars)["deleted"] == ref).all()
        for q in (0, 1, 3):
            c = stream_b_cap(off, flat, keys, 0.8, q)
            assert (c["deleted"] <= ref).all()
            assert ref.sum() - c["deleted"].sum() <= c["extra_retained_bound"] <= c["unchecked_pairs"]


def test_unrelated_first_member_does_not_split_capped_bucket():
    m = 300
    sets = [np.arange(10000, 10040)] + [np.r_[np.arange(40), 5000 + d] for d in range(1, m)]
    flat = np.concatenate(sets).astype(np.uint64)
    off = np.r_[0, np.cumsum([len(s) for s in sets])].astype(np.int64)
    keys = _keys(m)
    b = stream_b(off, flat, keys, 0.8)
    assert b["deleted"].sum() == m - 2
    one = stream_b_cap(off, flat, keys, 0.8, 10, stars=1)
    two = stream_b_cap(off, flat, keys, 0.8, 10)
    assert one["deleted"].sum() < m - 2
    assert two["deleted"].sum() == m - 2
    assert two["verified_pairs"] <= 2 * (m - 1)


def test_capped_counting_matches_enumeration():
    # the reference is the enumerating implementation the paper's earlier tables used
    import importlib.util
    spec = importlib.util.spec_from_file_location("ref", os.path.join(os.path.dirname(__file__), "stream_b_enum_ref.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rng = np.random.default_rng(5)
    keys_names = ("enumerated_pairs", "skipped_pairs", "verified_pairs", "accepted_edges", "unchecked_pairs")
    for trial in range(40):
        n = int(rng.integers(5, 60))
        bases = [rng.choice(300, 20, replace=False) for _ in range(4)]
        sets = []
        for d in range(n):
            if rng.random() < 0.15:
                s = rng.choice(np.arange(10000, 20000), 20, replace=False)
            else:
                s = bases[int(rng.integers(0, 4))].copy()
                s[rng.choice(20, int(rng.integers(0, 5)), replace=False)] = rng.integers(1000, 5000)
            sets.append(np.unique(s))
        flat = np.concatenate(sets).astype(np.uint64)
        off = np.r_[0, np.cumsum([len(s) for s in sets])].astype(np.int64)
        keys = rng.integers(0, 2, size=(3, n)).astype(np.uint64)
        for stars in (1, 2):
            for q in (0, 1, 3, 10):
                for prefix in (False, True):
                    a = stream_b_cap(off, flat, keys, 0.8, q, cap_prefix=prefix, stars=stars)
                    b = ref.stream_b_cap(off, flat, keys, 0.8, q, cap_prefix=prefix, stars=stars)
                    assert (a["deleted"] == b["deleted"]).all()
                    full = stream_b(off, flat, keys, 0.8)["deleted"]
                    assert full.sum() - a["deleted"].sum() <= a["extra_retained_bound"] <= a["unchecked_pairs"]
                    for k in keys_names:
                        assert a[k] == b[k], (trial, stars, q, prefix, k, a[k], b[k])


def test_two_groups_component_bound_is_one():
    # two groups of near-duplicates in one bucket, no duplicate across them: once the cap stops the
    # bucket, its members sit in two components, so at most one more document could be deleted
    m = 400
    sets = [np.r_[np.arange(40), 5000 + d] for d in range(m // 2)] + \
           [np.r_[np.arange(100, 140), 6000 + d] for d in range(m // 2)]
    order = np.random.default_rng(1).permutation(m)
    sets = [sets[i] for i in order]
    flat = np.concatenate(sets).astype(np.uint64)
    off = np.r_[0, np.cumsum([len(s) for s in sets])].astype(np.int64)
    keys = _keys(m)
    c = stream_b_cap(off, flat, keys, 0.8, 10)
    assert c["unchecked_pairs"] > 0
    assert c["extra_retained_bound"] == 1
    assert stream_b(off, flat, keys, 0.8)["deleted"].sum() - c["deleted"].sum() <= 1
