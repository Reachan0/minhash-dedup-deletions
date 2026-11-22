#!/usr/bin/env python3
"""Exact greedy retained-witness operator W."""
import os
import sys
import time

import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "..", "core")
sys.path.insert(0, CORE)


@nb.njit(cache=True, inline="always")
def _jaccard(off, flat, a, b):
    ia, ib = off[a], off[b]
    ea, eb = off[a + 1], off[b + 1]
    common = 0
    while ia < ea and ib < eb:
        xa, xb = flat[ia], flat[ib]
        if xa == xb:
            common += 1
            ia += 1
            ib += 1
        elif xa < xb:
            ia += 1
        else:
            ib += 1
    den = (ea - off[a]) + (eb - off[b]) - common
    return 0.0 if den == 0 else common / den


@nb.njit(cache=True)
def _run_w(off, flat, bucket_offsets, bucket_docs, memberships, membership_offsets, tau):
    n = off.shape[0] - 1
    kept = np.zeros(n, np.uint8)
    seen = np.zeros(n, np.int64)
    stamp = 0
    enumerated = verified = accepted = 0
    for d in range(n):
        if off[d + 1] == off[d]:
            kept[d] = 1
            continue
        stamp += 1
        found = 0
        for mi in range(membership_offsets[d], membership_offsets[d + 1]):
            bid = memberships[mi]
            for p in range(bucket_offsets[bid], bucket_offsets[bid + 1]):
                u = bucket_docs[p]
                if u >= d or seen[u] == stamp:
                    continue
                seen[u] = stamp
                enumerated += 1
                verified += 1
                if kept[u] != 0 and _jaccard(off, flat, u, d) + 1e-15 >= tau:
                    accepted += 1
                    found = 1
                    break
            if found:
                break
        kept[d] = 0 if found else 1
    return kept, enumerated, verified, accepted


def build_bucket_membership_csr(keys, valid=None):
    """Return bucket CSR and a per-document CSR of bucket memberships.

    Bucket order is band order, then key order, with document ids increasing.
    """
    keys = np.asarray(keys)
    n = keys.shape[1]
    if valid is None:
        valid = np.ones(n, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    bucket_docs = []
    bucket_offsets = [0]
    doc_members = [[] for _ in range(n)]
    for band in range(keys.shape[0]):
        ids = np.flatnonzero(valid)
        order = np.argsort(keys[band, ids], kind="stable")
        docs = ids[order]
        vals = keys[band, docs]
        starts = np.r_[0, np.flatnonzero(vals[1:] != vals[:-1]) + 1]
        ends = np.r_[starts[1:], len(docs)]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            bid = len(bucket_offsets) - 1
            group = docs[s:e].astype(np.int64)
            bucket_docs.extend(group.tolist())
            bucket_offsets.append(len(bucket_docs))
            for d in group:
                doc_members[int(d)].append(bid)
    membership_offsets = np.zeros(n + 1, dtype=np.int64)
    for d in range(n):
        membership_offsets[d + 1] = membership_offsets[d] + len(doc_members[d])
    memberships = np.empty(int(membership_offsets[-1]), dtype=np.int64)
    k = 0
    for x in doc_members:
        memberships[k:k + len(x)] = x
        k += len(x)
    return (np.asarray(bucket_offsets, dtype=np.int64),
            np.asarray(bucket_docs, dtype=np.int64), memberships, membership_offsets)


def exact_w(off, flat, keys, tau, valid=None):
    """Compute the exact W mask without the B union-find skip rule."""
    off = np.asarray(off, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.uint64)
    n = len(off) - 1
    if valid is None:
        valid = off[1:] > off[:-1]
    valid = np.asarray(valid, dtype=bool)
    t0 = time.perf_counter()
    bo, bd, mem, mo = build_bucket_membership_csr(keys, valid)
    t_csr = time.perf_counter() - t0
    t1 = time.perf_counter()
    kept, enum, ver, acc = _run_w(off, flat, bo, bd, mem, mo, float(tau))
    t_verify = time.perf_counter() - t1
    wall = time.perf_counter() - t0
    kept = kept.astype(bool)
    kept[~valid] = True
    deleted = ~kept
    return dict(kept=kept, deleted=deleted, bucket_offsets=bo, bucket_docs=bd,
                memberships=mem, membership_offsets=mo,
                enumerated_pairs=int(enum), skipped_pairs=0,
                verified_pairs=int(ver), accepted_edges=int(acc), wall_s=wall,
                wall_total_s=wall, wall_csr_s=t_csr, wall_verify_s=t_verify)


if __name__ == "__main__":
    raise SystemExit("use exact_w() from Python")
