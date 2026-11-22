#!/usr/bin/env python3
"""Anchor-verified sparse baseline (in the style of NeMo Curator's bucket-anchor check).

In every bucket, each member is compared with the bucket's anchor (its smallest document id) by
exact Jaccard, and only passing pairs become edges; components are formed by union-find and the
minimum id of each component is kept. The result is sound (every edge is exact) but can miss
exact deletions whose edges never involve an anchor. A pair whose endpoints are already
connected is skipped, which does not change the components.
"""
import time

import numba as nb
import numpy as np

from stream_b import _find, _groups_for_band, _jaccard


@nb.njit(cache=True)
def _anchor_bucket(docs, off, flat, parent, tau):
    verified = accepted = 0
    a = docs[0]
    for y in range(1, docs.shape[0]):
        b = docs[y]
        ra = _find(parent, a)
        rb = _find(parent, b)
        if ra == rb:
            continue
        verified += 1
        if _jaccard(off, flat, a, b) >= tau - 1e-12:
            accepted += 1
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    return verified, accepted


def anchor_b(off, flat, keys, tau, valid=None):
    off = np.asarray(off, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.uint64)
    n = len(off) - 1
    if valid is None:
        valid = off[1:] > off[:-1]
    parent = np.arange(n, dtype=np.int64)
    verified = accepted = 0
    t0 = time.perf_counter()
    for band in range(keys.shape[0]):
        for docs in _groups_for_band(keys, valid, band):
            v, a = _anchor_bucket(docs, off, flat, parent, float(tau))
            verified += v
            accepted += a
    roots = np.array([_find(parent, d) for d in range(n)], dtype=np.int64)
    mins = np.full(n, n, dtype=np.int64)
    np.minimum.at(mins, roots, np.arange(n, dtype=np.int64))
    deleted = np.asarray(valid, bool) & (np.arange(n) != mins[roots])
    return dict(deleted=deleted, verified_pairs=verified, accepted_edges=accepted,
                wall_s=time.perf_counter() - t0)
