#!/usr/bin/env python3
"""Soundness audit by one exact scan, replacing the 256-permutation search of f_estimators.py.

Draws the same stratified sample of path A's deletions as f_estimators.py (seed 1, min-id, same
strata and rng), then decides for each sampled deletion whether some other document has J >= 0.8:
  index   the sampled documents' prefix shingles (prefix length |x| - ceil(0.8|x|) + 1 under
          ascending hash order), in a sorted array with a 2^24-bit filter in front of it
  scan    every document of the corpus once, uncollapsed; each of its prefix shingles probes the
          index, and a hit passing the length filter 0.8|x| <= |y| <= |x|/0.8 is checked by exact
          Jaccard. Two sets with J >= t share a prefix shingle, so the scan misses no neighbour.
Writes results/F_scan_<corpus>.jsonl: the estimate (equal to f_estimators.py's when both are
exact), agreement with the exact reference, index size in bytes, times for 12 threads and 1.
Usage: env/bin/python audit_scan.py <corpus name under runs/>
"""
import json
import math
import resource
import sys
import time

import numba as nb
import numpy as np

W = "/path/to/scratch/dedupjoin"
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run"]
import snap_common as S  # noqa: E402

TAU = 0.8
BITS = 24


def plen(n):
    return n - math.ceil(TAU * n - 1e-9) + 1 if n > 0 else 0


@nb.njit(cache=True, inline="always")
def _jac(off, flat, a, b):
    ia, ib = off[a], off[b]
    ea, eb = off[a + 1], off[b + 1]
    na, nb_ = ea - ia, eb - ib
    common = 0
    while ia < ea and ib < eb:
        if flat[ia] == flat[ib]:
            common += 1
            ia += 1
            ib += 1
        elif flat[ia] < flat[ib]:
            ia += 1
        else:
            ib += 1
    den = na + nb_ - common
    return 0.0 if den == 0 else common / den


@nb.njit(parallel=True, cache=True)
def _scan(off, flat, tok, owner, bitmap, sdoc, ssize, found, tau):
    n = off.shape[0] - 1
    mask = (1 << 24) - 1
    probes = np.zeros(n, np.int64)
    checks = np.zeros(n, np.int64)
    for y in nb.prange(n):
        ny = off[y + 1] - off[y]
        if ny == 0:
            continue
        p = ny - int(math.ceil(tau * ny - 1e-9)) + 1
        for k in range(off[y], off[y] + p):
            t = flat[k]
            h = np.int64(t & np.uint64(mask))
            if (bitmap[h >> 3] >> (h & 7)) & 1 == 0:
                continue
            probes[y] += 1
            lo = np.searchsorted(tok, t)
            while lo < tok.shape[0] and tok[lo] == t:
                x = owner[lo]
                lo += 1
                if found[x] or sdoc[x] == y:
                    continue
                nx = ssize[x]
                if ny < tau * nx - 1e-9 or tau * ny > nx + 1e-9:
                    continue
                checks[y] += 1
                if _jac(off, flat, sdoc[x], y) + 1e-12 >= tau:
                    found[x] = True
    return probes.sum(), checks.sum()


def sample_deletions(name):
    """The sample of f_estimators.py, reproduced step by step (same seeds and strata)."""
    d = f"{W}/runs/{name}"
    S.SHINGLE_NPZ, S.CACHE, S.SNAP = f"{d}/snapshot.npz", f"{d}/cache", d
    import collapsed_lib as CL
    K = CL.K
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    group, m, nsh = z["group_of_doc"], z["group_size"], z["group_nshingle"]
    n, G = len(group), len(m)
    valid = nsh[group] > 0
    kb_g = CL.group_band_keys(z, 1)
    ti, tj, _ = CL.truth_u(TAU)
    st = CL.exact_state(z, ti, tj)
    ei, ej, _ = K.chain_edges(kb_g[:, group], valid)
    rem = (K.uf_minid(n, ei, ej) != np.arange(n)) & valid
    del ei, ej
    D = np.flatnonzero(rem)
    y_true = ~st["has_nb_doc"][D]
    size = nsh[group[D]]
    q = np.searchsorted(np.quantile(size, [0.2, 0.4, 0.6, 0.8]), size, side="right")
    big = np.zeros(G, np.int64)
    for b in range(kb_g.shape[0]):
        _, inv, c = np.unique(kb_g[b], return_inverse=True, return_counts=True)
        docs_in_bucket = np.bincount(inv, weights=m, minlength=len(c)).astype(np.int64)
        big = np.maximum(big, docs_in_bucket[inv])
    bc = np.digitize(big[group[D]], [3, 11, 101])
    stratum = q * 4 + bc
    labels, N_h = np.unique(stratum, return_counts=True)
    total = 1000
    n_h = np.maximum(5, np.round(total * N_h / N_h.sum()).astype(int))
    n_h = np.minimum(n_h, N_h)
    srng = np.random.default_rng(11)
    sample = np.concatenate([srng.choice(np.flatnonzero(stratum == h), k, replace=False)
                             for h, k in zip(labels, n_h)])
    return D, y_true, stratum, labels, N_h, sample


def main():
    name = sys.argv[1]
    D, y_true, stratum, labels, N_h, sample = sample_deletions(name)
    with np.load(f"{W}/runs/{name}/snapshot.npz") as z:
        off, flat = z["off"].astype(np.int64), np.asarray(z["flat"], np.uint64)
    sdoc = D[sample].astype(np.int64)
    rows = []
    for threads in (12, 1):
        nb.set_num_threads(threads)
        t0 = time.perf_counter()
        ssize = (off[sdoc + 1] - off[sdoc]).astype(np.int64)
        toks, owners = [], []
        for x, dd in enumerate(sdoc):
            p = plen(int(ssize[x]))
            toks.append(flat[off[dd]:off[dd] + p])
            owners.append(np.full(p, x, np.int64))
        tok = np.concatenate(toks)
        owner = np.concatenate(owners)
        o = np.argsort(tok, kind="stable")
        tok, owner = tok[o], owner[o]
        h = (tok & np.uint64((1 << BITS) - 1)).astype(np.int64)
        bitmap = np.zeros(1 << (BITS - 3), np.uint8)
        np.bitwise_or.at(bitmap, h >> 3, (1 << (h & 7)).astype(np.uint8))
        set_bytes = int(ssize.sum()) * 8
        t_index = time.perf_counter() - t0
        found = np.zeros(len(sdoc), np.bool_)
        if threads == 12:
            _scan(off[:3], flat, tok, owner, bitmap, sdoc, ssize, np.ones(len(sdoc), np.bool_), TAU)  # compile
        t1 = time.perf_counter()
        probes, checks = _scan(off, flat, tok, owner, bitmap, sdoc, ssize, found, TAU)
        t_scan = time.perf_counter() - t1
        y = ~found
        est, var = 0.0, 0.0
        for hh, Nh in zip(labels, N_h):
            sel = stratum[sample] == hh
            nh = int(sel.sum())
            yh = y[sel].astype(float)
            est += Nh / len(D) * yh.mean()
            if nh > 1:
                var += (Nh / len(D)) ** 2 * (1 - nh / Nh) * yh.var(ddof=1) / nh
        se = var ** 0.5
        row = dict(kind="soundness_scan", corpus=name, tau=TAU, docs=int(len(off) - 1),
                   shingles=int(len(flat)), deletions=int(len(D)), sample=int(len(sample)),
                   strata=int(len(labels)), estimate=est, se=se, upper95=est + 1.645 * se,
                   true_share=float(y_true.mean()), disagreements_vs_truth=int((y != y_true[sample]).sum()),
                   threads=threads, t_index_s=round(t_index, 3), t_scan_s=round(t_scan, 2),
                   probes=int(probes), jaccard_checks=int(checks), sample_set_bytes=set_bytes,
                   index_entries=int(len(tok)),
                   peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(f"{W}/results/F_scan_{name}.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
