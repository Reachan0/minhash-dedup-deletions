#!/usr/bin/env python3
"""Exact shingle-set collapse and document-level lifting helpers."""
import hashlib
from collections import defaultdict

import numpy as np


def _digest(a):
    # The digest is an index only. Full arrays are compared before grouping.
    return hashlib.blake2b(np.asarray(a, dtype=np.uint64).tobytes(), digest_size=8).digest()


def collapse(off, flat):
    """Group equal sorted arrays, splitting digest collisions and empty sets.

    Group ids follow the first document occurrence, so each group's representative
    is its minimum document id.
    """
    off = np.asarray(off, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.uint64)
    n = len(off) - 1
    by_digest = defaultdict(list)
    for d in range(n):
        by_digest[_digest(flat[off[d]:off[d + 1]])].append(d)
    groups = []
    digest_collisions = 0
    group_of = np.empty(n, dtype=np.int64)
    # Empty sets are explicitly singleton groups.
    for docs in by_digest.values():
        exact = []
        saw_empty = False
        for d in docs:
            arr = flat[off[d]:off[d + 1]]
            if arr.size == 0:
                saw_empty = True
                group_of[d] = len(groups)
                groups.append([d])
                continue
            match = None
            for gi, rep in exact:
                r = flat[off[rep]:off[rep + 1]]
                if np.array_equal(arr, r):
                    match = gi
                    break
            if match is None:
                match = len(groups)
                groups.append([d])
                exact.append((match, d))
            else:
                groups[match].append(d)
            group_of[d] = match
        digest_collisions += max(0, len(exact) - 1)
        if saw_empty and exact:
            digest_collisions += 1
    # Digest dictionary order is arbitrary. Renumber by minimum document id.
    order = np.argsort(np.asarray([g[0] for g in groups]), kind="stable")
    remap = np.empty(len(groups), dtype=np.int64)
    remap[order] = np.arange(len(groups), dtype=np.int64)
    group_of = remap[group_of]
    reps = np.asarray([min(groups[int(g)]) for g in order], dtype=np.int64)
    sizes = np.bincount(group_of, minlength=len(groups)).astype(np.int64)
    group_n = off[reps + 1] - off[reps]
    uoff = np.zeros(len(groups) + 1, dtype=np.int64)
    uoff[1:] = np.cumsum(group_n)
    uflat = np.empty(int(uoff[-1]), dtype=np.uint64)
    for g, d in enumerate(reps):
        uflat[uoff[g]:uoff[g + 1]] = flat[off[d]:off[d + 1]]
    return dict(group_of_doc=group_of, group_rep=reps, group_size=sizes,
                group_nshingle=group_n, off_u=uoff, flat_u=uflat,
                n_docs=n, n_groups=len(groups), digest_collision_groups=digest_collisions)


def lift_b(group_root, group_of_doc, group_rep, valid=None):
    """Lift group connected-component roots to the minimum-id B deletion mask."""
    root = np.asarray(group_root)
    group = np.asarray(group_of_doc, dtype=np.int64)
    reps = np.asarray(group_rep, dtype=np.int64)
    n = len(group)
    mins = np.full(len(root), n, dtype=np.int64)
    np.minimum.at(mins, root, reps)
    out = np.arange(n, dtype=np.int64) != mins[root[group]]
    if valid is not None:
        out &= np.asarray(valid, dtype=bool)
    return out


def lift_c(group_root, group_of_doc, group_rep, valid=None):
    """Lift exact component results for C, which uses the same min-id components."""
    return lift_b(group_root, group_of_doc, group_rep, valid=valid)


def lift_w(group_kept, group_of_doc, group_rep, valid=None):
    """Lift W's group-block result.

    If a group's minimum id is kept, it witnesses every later exact copy. If it
    is deleted, an earlier kept witness exists and the whole block is deleted.
    """
    kept = np.asarray(group_kept, dtype=bool)
    group = np.asarray(group_of_doc, dtype=np.int64)
    reps = np.asarray(group_rep, dtype=np.int64)
    d = np.arange(len(group), dtype=np.int64)
    out = (~kept[group]) | (d != reps[group])
    if valid is not None:
        out &= np.asarray(valid, dtype=bool)
    return out


def lift_masks(group_root=None, group_kept=None, collapsed=None, valid=None):
    """Convenience return of B/W/C masks from a collapse result."""
    if collapsed is None:
        raise ValueError("collapsed is required")
    kw = dict(group_of_doc=collapsed["group_of_doc"], group_rep=collapsed["group_rep"])
    out = {}
    if group_root is not None:
        out["B"] = lift_b(group_root, valid=valid, **kw)
        out["C"] = lift_c(group_root, valid=valid, **kw)
    if group_kept is not None:
        out["W"] = lift_w(group_kept, valid=valid, **kw)
    return out
