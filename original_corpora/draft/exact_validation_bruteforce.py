#!/usr/bin/env python3
"""Validate the exact reference (prefix filtering) against brute force.

Takes document samples that are enriched with exact near-duplicate components, computes
the complete Jaccard edge set with a dense sparse product, and compares it with the
prefix-filter join on the same subset. This checks the reference itself on real corpus
data rather than on synthetic sets.

Output: results/exact_validation_bruteforce.json
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
C03 = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, C03)
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")


def brute(off, flat, tau):
    uniq, inv = np.unique(flat, return_inverse=True)
    n = len(off) - 1
    rows = np.repeat(np.arange(n), np.diff(off))
    X = sp.csr_matrix((np.ones(len(flat), np.float32), (rows, inv)), shape=(n, len(uniq)))
    O = (X @ X.T).tocoo()
    sz = np.diff(off).astype(np.float64)
    m = O.row < O.col
    i, j, o = O.row[m], O.col[m], O.data[m].astype(np.float64)
    jac = o / (sz[i] + sz[j] - o)
    k = jac >= tau - 1e-12
    return set(zip(i[k].tolist(), j[k].tolist()))


def subset(off, flat, sel):
    """Renumber a document subset into a compact shingle representation."""
    sel = np.sort(sel)
    offs = off[sel]
    lens = np.diff(off)[sel]
    parts = [flat[offs[k]:offs[k] + lens[k]] for k in range(len(sel))]
    new_off = np.zeros(len(sel) + 1, np.int64)
    new_off[1:] = np.cumsum([len(p) for p in parts])
    new_flat = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return new_off, new_flat


def run(corpus, rng_seed=0):
    n = MC.CORPORA[corpus]
    off, flat, nw, meta = MC.load_shingles(corpus)
    q = np.load(os.path.join(WRK, f"{corpus}_truth_t0.8.npz"))
    root = np.asarray(q["root"])
    rng = np.random.default_rng(rng_seed)
    # base: 1,000 random documents
    sel = set(rng.choice(n, 1000, replace=False).tolist())
    # enrich with 30 random exact components (size 2..10) so the sample contains true edges
    counts = np.bincount(root, minlength=n)
    comps = np.zeros(n, bool)
    comps[root] = counts[root] >= 2
    comp_ids = np.unique(root[comps])
    sizes = counts[comp_ids]
    ok = comp_ids[(sizes >= 2) & (sizes <= 10)]
    for c in rng.choice(ok, size=min(30, len(ok)), replace=False):
        docs = np.flatnonzero(root == c)
        sel.update(docs.tolist())
    sel = np.array(sorted(sel))
    log = []
    for tau in (0.7, 0.8, 0.9):
        o2, f2 = subset(off, flat, sel)
        t0 = time.time()
        b = brute(o2, f2, tau)
        tb = time.time() - t0
        R, S, V = EJ.build_ranks(o2, f2)
        t0 = time.time()
        res = EJ.exact_join(o2, R, S, V, tau, 4)
        tj = time.time() - t0
        g = set(zip(res["i"].tolist(), res["j"].tolist()))
        row = dict(corpus=corpus, tau=tau, n_docs=int(len(sel)),
                   brute_pairs=len(b), join_pairs=len(g),
                   missing_from_join=len(b - g), extra_in_join=len(g - b),
                   identical=bool(b == g),
                   brute_sec=round(tb, 2), join_sec=round(tj, 2),
                   sec_bruteforce=round(tb, 2))
        log.append(row)
        print(row, flush=True)
    return log


def main():
    out = dict(version=1, provenance="independently rerun (this run)",
               design="1,000 random documents plus all members of 30 sampled exact components "
                      "of size 2-10, per corpus; complete edge set by sparse matrix product "
                      "compared with the prefix-filter exact join on the same subset",
               rows=[])
    for corpus in ("c4", "fw"):
        out["rows"] += run(corpus)
    with open(os.path.join(OUT, "exact_validation_bruteforce.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
