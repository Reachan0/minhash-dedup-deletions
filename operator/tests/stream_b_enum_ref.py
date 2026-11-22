#!/usr/bin/env python3
"""Streaming exact Path B and its sound capped variant."""
import os
import sys
import time

import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "..", "core")
sys.path.insert(0, CORE)


@nb.njit(cache=True, inline="always")
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@nb.njit(cache=True, inline="always")
def _jaccard(off, flat, a, b):
    i, j = off[a], off[b]
    ia, ib = i, j
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
    den = (ea - i) + (eb - j) - common
    return 0.0 if den == 0 else common / den


@nb.njit(cache=True)
def _process_bucket(docs, off, flat, parent, tau, cap, cap_prefix, stars):
    """Process one sorted bucket. cap < 0 means uncapped. The star prefix (m - 1 tests per
    star pass) counts against the cap only if cap_prefix is set. With stars == 2, a second star
    pass runs from the first member that the first pass left outside the anchor's component, so
    that an anchor unrelated to the rest of its bucket does not leave the bucket split."""
    enumerated = skipped = verified = accepted = unchecked = 0
    # A star prefix makes exact-copy buckets cheap. Once every document in the
    # bucket is in one verified component, all remaining pairs are known skips.
    anchor = docs[0]
    for y in range(1, docs.shape[0]):
        a = anchor
        b = docs[y]
        enumerated += 1
        ra = _find(parent, a)
        rb = _find(parent, b)
        if ra == rb:
            skipped += 1
            continue
        if cap_prefix and cap >= 0 and verified >= cap:
            unchecked += 1
            continue
        verified += 1
        if _jaccard(off, flat, a, b) + 1e-15 >= tau:
            accepted += 1
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    prefix_verified = verified
    anchor_root = _find(parent, anchor)
    all_same = True
    for z in range(docs.shape[0]):
        if _find(parent, docs[z]) != anchor_root:
            all_same = False
            break
    if all_same:
        # The omitted suffix is logically enumerated as known skips. Counting it
        # keeps the candidate counter equal to the full within-bucket stream.
        total = docs.shape[0] * (docs.shape[0] - 1) // 2
        skipped += total - enumerated
        enumerated = total
        return enumerated, skipped, verified, accepted, unchecked
    second = -1
    if stars >= 2:
        for z in range(1, docs.shape[0]):
            if _find(parent, docs[z]) != anchor_root:
                second = z
                break
        a = docs[second]
        for y in range(1, docs.shape[0]):
            if y == second:
                continue
            b = docs[y]
            enumerated += 1
            ra = _find(parent, a)
            rb = _find(parent, b)
            if ra == rb:
                skipped += 1
                continue
            if cap_prefix and cap >= 0 and verified >= cap:
                unchecked += 1
                continue
            verified += 1
            if _jaccard(off, flat, a, b) + 1e-15 >= tau:
                accepted += 1
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb
        prefix_verified = verified
        # Every pair with an anchor is decided. If the other members form one component,
        # every remaining pair is a known skip.
        r0 = -1
        rest_same = True
        for z in range(1, docs.shape[0]):
            if z == second:
                continue
            rz = _find(parent, docs[z])
            if r0 < 0:
                r0 = rz
            elif rz != r0:
                rest_same = False
                break
        if rest_same:
            total = docs.shape[0] * (docs.shape[0] - 1) // 2
            skipped += total - enumerated
            enumerated = total
            return enumerated, skipped, verified, accepted, unchecked
    for x in range(1, docs.shape[0] - 1):
        if x == second:
            continue
        a = docs[x]
        for y in range(x + 1, docs.shape[0]):
            if y == second:
                continue
            b = docs[y]
            enumerated += 1
            ra = _find(parent, a)
            rb = _find(parent, b)
            if ra == rb:
                skipped += 1
                continue
            if cap >= 0 and verified - (0 if cap_prefix else prefix_verified) >= cap:
                unchecked += 1
                continue
            verified += 1
            if _jaccard(off, flat, a, b) + 1e-15 >= tau:
                accepted += 1
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb
    return enumerated, skipped, verified, accepted, unchecked


def _groups_for_band(keys, valid, band):
    ids = np.flatnonzero(valid).astype(np.int64)
    if ids.size < 2:
        return
    order = np.argsort(keys[band, ids], kind="stable")
    docs = ids[order]
    vals = keys[band, docs]
    starts = np.r_[0, np.flatnonzero(vals[1:] != vals[:-1]) + 1]
    ends = np.r_[starts[1:], len(docs)]
    for s, e in zip(starts, ends):
        if e - s >= 2:
            yield docs[s:e]


def stream_b(off, flat, keys, tau, valid=None, cap=None, cap_prefix=False, stars=1):
    """Run B while holding one bucket's document ids at a time.

    Returns a minimum-id deletion mask and counters. Candidate pairs repeated in
    multiple bands are harmless and are counted each time they are encountered.
    """
    off = np.asarray(off, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.uint64)
    keys = np.asarray(keys)
    n = len(off) - 1
    if valid is None:
        valid = (off[1:] > off[:-1])
    valid = np.asarray(valid, dtype=np.bool_)
    parent = np.arange(n, dtype=np.int64)
    totals = np.zeros(5, dtype=np.int64)
    t0 = time.perf_counter()
    t_kernel = 0.0
    for band in range(keys.shape[0]):
        for docs in _groups_for_band(keys, valid, band):
            tk = time.perf_counter()
            totals += np.asarray(_process_bucket(docs, off, flat, parent, float(tau),
                                                  -1 if cap is None else int(cap), bool(cap_prefix), int(stars)),
                                 dtype=np.int64)
            t_kernel += time.perf_counter() - tk
    t_stream = time.perf_counter() - t0
    roots = np.empty(n, dtype=np.int64)
    for d in range(n):
        roots[d] = _find(parent, d)
    mins = np.full(n, n, dtype=np.int64)
    np.minimum.at(mins, roots, np.arange(n, dtype=np.int64))
    deleted = valid & (np.arange(n) != mins[roots])
    e, s, v, a, u = [int(x) for x in totals]
    return dict(deleted=deleted, roots=roots,
                enumerated_pairs=e, skipped_pairs=s, verified_pairs=v,
                accepted_edges=a, unchecked_pairs=u,
                extra_retained_bound=u, true_edges_among_unchecked_bound=u,
                wall_s=t_stream, wall_total_s=t_stream,
                wall_verify_s=t_kernel, wall_bucket_generation_s=t_stream - t_kernel)


def stream_b_cap(off, flat, keys, tau, q, valid=None, cap_prefix=False, stars=2):
    """Capped B: up to two star passes, then at most q tests per bucket (the star passes count
    against the cap only if cap_prefix). The returned bound is sound either way."""
    out = stream_b(off, flat, keys, tau, valid=valid, cap=q, cap_prefix=cap_prefix, stars=stars)
    out["cap_per_bucket"] = int(q)
    out["bound_statement"] = (
        "extra retained documents vs B <= true edges among unchecked pairs <= U")
    return out


def candidate_keys(off, flat, seed, H=112, bands=14, rows=8):
    """Construct frozen signatures and band keys without importing the run harness."""
    vendor = os.path.join(CORE, "vendor")
    sys.path.insert(0, vendor)
    import lsh
    sig = lsh.signatures(np.asarray(off, np.int64), np.asarray(flat, np.uint64), H, seed, "dt")
    return lsh.band_keys(sig, bands, rows)


if __name__ == "__main__":
    raise SystemExit("use stream_b() or stream_b_cap() from Python")
