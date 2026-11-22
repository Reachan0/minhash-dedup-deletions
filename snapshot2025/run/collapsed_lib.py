#!/usr/bin/env python3
"""Exact-duplicate-collapsed evaluation for the CC-MAIN-2025-30 reveal.

Motivation (measured, not assumed): 86,771 of 531,597 documents have exactly the
same 8-shingle set, so the exact Jaccard graph contains a clique of 86,771
documents (3.76 billion edges) and the literal all-pairs paths B/W/C cannot be
materialised in 32 GiB. The document-level relation is nevertheless fully
determined by

  * the exact graph between *distinct* shingle sets (431,116 groups), and
  * the multiplicity m_g of each set (m_g docs are pairwise J = 1).

This module computes the exact document-level graph implicitly with those
multiplicities. It does not change any method definition: A's chain edges are
still built on the 531,597 documents; B/W's verified graph and C's exact graph
are lifted from the group level, where a group of m identical docs contributes
the clique C(m,2) and cross-group edges contribute m_u * m_v document pairs.

Empty documents (0 shingles) have no Jaccard edges; each is its own group.
"""
import json
import os

import numpy as np
import xxhash

import env_paths

S = env_paths.prepared()
K = S.K
EJ = K.EJ

CACHE = S.CACHE
COLLAPSED_NPZ = os.path.join(CACHE, "collapsed_sets.npz")
RANKS_U = os.path.join(CACHE, "collapsed_ranks.npz")


def truth_u_path(tau):
    return os.path.join(CACHE, f"truth_u_t{tau}.npz")


def build_collapsed():
    off, flat, nw = S.load_snapshot()
    n = len(off) - 1
    sizes = (off[1:] - off[:-1]).astype(np.int64)
    h = np.empty(n, np.uint64)
    for i in range(n):
        a, b = int(off[i]), int(off[i + 1])
        h[i] = xxhash.xxh64_intdigest(flat[a:b].tobytes()) if b > a else np.uint64(0)
    uniq, first, inverse, counts = np.unique(h, return_index=True,
                                            return_inverse=True, return_counts=True)
    # groups ordered by first occurrence (== minimum document index)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    group = rank[inverse].astype(np.int32)
    # empty documents: one group each, no edges
    empty = sizes == 0
    ne = int(empty.sum())
    if ne:
        base = int(group.max()) + 1
        group[empty] = np.arange(base, base + ne, dtype=np.int32)
    G = int(group.max()) + 1
    g_first = np.full(G, n, np.int64)
    np.minimum.at(g_first, group, np.arange(n))
    g_size = np.bincount(group, minlength=G).astype(np.int32)
    g_nsh = np.zeros(G, np.int64)
    g_nsh[group] = sizes
    rep = g_first
    off_u = np.zeros(G + 1, np.int64)
    off_u[1:] = np.cumsum(g_nsh)
    # flat_u = slices of the representative documents
    flat_u = np.empty(int(off_u[-1]), np.uint64)
    for g in range(G):
        if g_nsh[g] == 0:           # the original all-empty group left unused by the remap
            continue
        d = rep[g]
        flat_u[off_u[g]:off_u[g + 1]] = flat[int(off[d]):int(off[d + 1])]
    np.savez(COLLAPSED_NPZ, off_u=off_u, flat_u=flat_u, group_of_doc=group,
             group_size=g_size, group_rep=rep, group_nshingle=g_nsh)
    return dict(n_docs=n, n_groups=G, n_empty_docs=ne,
                n_unique_sets=int(len(uniq)), flat_u=int(off_u[-1]))


def load_collapsed():
    z = np.load(COLLAPSED_NPZ)
    out = {k: np.asarray(z[k]) for k in
           ("off_u", "flat_u", "group_of_doc", "group_size", "group_rep", "group_nshingle")}
    z.close()
    return out


def build_ranks_u():
    z = load_collapsed()
    off_u, flat_u = z["off_u"], z["flat_u"]
    import time
    t0 = time.time()
    R, Sv, V = EJ.build_ranks(off_u, flat_u)
    dt = time.time() - t0
    np.savez(RANKS_U, R=R, S=np.int64(Sv), V=np.int64(V), t_build_s=np.float64(dt))
    return R, Sv, V, dt


def ranks_u():
    z = np.load(RANKS_U)
    R, Sv, V = np.asarray(z["R"]), int(z["S"]), int(z["V"])
    z.close()
    return R, Sv, V


def truth_u(tau, threads=K.THREADS):
    p = truth_u_path(tau)
    if os.path.exists(p):
        z = np.load(p)
        out = (np.asarray(z["i"]), np.asarray(z["j"]), np.asarray(z["jac"]))
        z.close()
        return out
    dz = load_collapsed()
    off_u, flat_u = dz["off_u"], dz["flat_u"]
    R, Sv, V = ranks_u()
    res = EJ.exact_join(off_u, R, Sv, V, tau, threads)
    i = res["i"].astype(np.int64)
    j = res["j"].astype(np.int64)
    jac = res["jac"].astype(np.float64)
    tmp = p + ".tmp.npz"
    np.savez(tmp, i=i, j=j, jac=jac)
    os.replace(tmp, p)
    return i, j, jac


# ---------------------------------------------------------------- lifting
def group_degree(G, gi, gj):
    d = np.bincount(gi, minlength=G) + np.bincount(gj, minlength=G)
    return d.astype(np.int64)


def exact_state(z, ti, tj):
    """Exact components and per-document exact-deleted mask (min doc id per component)."""
    G = len(z["group_size"])
    m = z["group_size"]
    rep = z["group_rep"]
    group = z["group_of_doc"]
    n = len(group)
    valid = z["group_nshingle"][group] > 0
    root = K.uf_minid(G, ti, tj)
    comp_docs = np.bincount(root, weights=m, minlength=G)
    comp_min = np.full(G, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(comp_min, root, rep)
    doc_comp_min = comp_min[root[group]]
    rem = valid & (np.arange(n) != doc_comp_min)
    deg = group_degree(G, ti, tj)
    has_nb_group = (m >= 2) | (deg > 0)
    has_nb_doc = valid & has_nb_group[group]
    return dict(root=root, comp_docs=comp_docs, comp_min=comp_min,
                rem=rem, has_nb_doc=has_nb_doc, deg=deg, G=G)


def lift_metrics(rem, z, gi, gj, has_nb_doc, n, valid):
    """Document-level metrics of a deletion mask, using group multiplicities."""
    rem = rem & valid
    G = len(z["group_size"])
    m = z["group_size"]
    group = z["group_of_doc"]
    r = np.bincount(group[~rem & valid], minlength=G).astype(np.int64)
    uns = rem & ~has_nb_doc
    within = int((r * (r - 1) // 2).sum())
    across = int((r[gi].astype(np.int64) * r[gj].astype(np.int64)).sum())
    nb_ret = np.zeros(G, bool)
    np.logical_or.at(nb_ret, gi, r[gj] > 0)
    np.logical_or.at(nb_ret, gj, r[gi] > 0)
    cov_group = ((r >= 1) & (m >= 2)) | ((r == 0) & nb_ret)
    covered = rem & cov_group[group]
    del_count = int(rem.sum())
    return dict(
        deleted=del_count,
        unsupported=int(uns.sum()),
        unsupported_share_del=float(uns.sum() / del_count) if del_count else None,
        unsupported_share_input=float(uns.sum() / n),
        witness_coverage=float(covered.sum() / del_count) if del_count else 1.0,
        witness_covered=int(covered.sum()),
        residual_duplicate_pairs=within + across,
        residual_within_groups=within,
        residual_across_groups=across,
    )


def group_band_keys(z, seed):
    """Seed-s signatures on the collapsed corpus -> (14, G) band keys."""
    sig = K.L.signatures(z["off_u"], z["flat_u"], K.H, seed, "dt")
    Kb = K.L.band_keys(sig, K.B, K.R)
    del sig
    return Kb


def full_band_keys(Kb_g, z):
    """Expand group band keys to the 531,597 documents (identical sets share a signature)."""
    n = len(z["group_of_doc"])
    Kb = np.empty((Kb_g.shape[0], n), Kb_g.dtype)
    Kb[:, :] = Kb_g[:, z["group_of_doc"]]
    return Kb
