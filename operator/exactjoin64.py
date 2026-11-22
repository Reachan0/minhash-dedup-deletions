#!/usr/bin/env python3
"""64-bit-offset variant of the frozen vendor exact self-join.

The join kernel is reused unchanged. Rank construction removes the vendor's
2^31 assertion and chooses uint32 ranks whenever the rank universe fits.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "..", "core", "vendor")
sys.path.insert(0, VENDOR)
import exactjoin as _vendor  # noqa: E402


def _sort_segments(ranks, off):
    for d in range(len(off) - 1):
        a, b = int(off[d]), int(off[d + 1])
        if b - a > 1:
            ranks[a:b].sort()


def build_ranks(off, flat, chunk=8_000_000):
    """Build ranks with int64 offsets and no total-position 2^31 limit."""
    off = np.asarray(off, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.uint64)
    npos = int(len(flat))
    if npos == 0:
        return np.zeros(0, dtype=np.uint32), 0, 0
    ordered = np.sort(flat.copy())
    repeated = ordered[1:][ordered[1:] == ordered[:-1]]
    del ordered
    nsv, counts = np.unique(repeated, return_counts=True)
    del repeated
    rank_of = np.empty(len(nsv), dtype=np.int64)
    order = np.lexsort((nsv, counts + 1))
    rank_of[order] = np.arange(len(nsv), dtype=np.int64)
    singleton_count = npos
    V = npos + len(nsv)
    dtype = np.uint32 if V <= np.iinfo(np.uint32).max else np.int64
    ranks = np.empty(npos, dtype=dtype)
    for start in range(0, npos, chunk):
        x = flat[start:start + chunk]
        pos = np.searchsorted(nsv, x)
        hit = pos < len(nsv)
        safe = np.minimum(pos, max(0, len(nsv) - 1))
        if len(nsv):
            hit &= nsv[safe] == x
        values = np.arange(start, start + len(x), dtype=np.int64)
        if len(nsv):
            values[hit] = singleton_count + rank_of[safe[hit]]
        ranks[start:start + len(x)] = values
    _sort_segments(ranks, off)
    return ranks, singleton_count, V


def exact_join(off, ranks, singleton_count, V, tau, threads=12, **kwargs):
    """Run the unchanged vendor verification/index kernel on 64-bit-offset data."""
    off = np.asarray(off, dtype=np.int64)
    ranks = np.asarray(ranks)
    return _vendor.exact_join(off, ranks, int(singleton_count), int(V), tau,
                              int(threads), **kwargs)


def edge_set(result):
    return {(int(i), int(j)) for i, j in zip(result["i"], result["j"])}


def identical_edge_set(off, flat, tau=0.8, threads=1):
    """Compare this implementation and the vendor join on an input under its cap."""
    ranks, s, v = build_ranks(off, flat)
    ours = exact_join(off, ranks, s, v, tau, threads=threads)
    vr, vs, vv = _vendor.build_ranks(np.asarray(off, np.int64), np.asarray(flat, np.uint64))
    theirs = _vendor.exact_join(off, vr, vs, vv, tau, threads)
    return edge_set(ours) == edge_set(theirs), ours, theirs
