#!/usr/bin/env python3
"""Cross-validate paths B and C on a 1,000-document subset against brute force.

Also checks path A's pairing edges against the same truth (as a candidate set) and
records the LSH misses.  Output: core/crosscheck_bruteforce.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K

import numba as nb


@nb.njit(parallel=True, cache=True)
def jaccard_matrix(off, flat, n):
    J = np.zeros((n, n), np.float32)
    for i in nb.prange(n):
        li = off[i + 1] - off[i]
        for j in range(i + 1, n):
            lj = off[j + 1] - off[j]
            a, b, o = off[i], off[j], 0
            a1, b1 = off[i + 1], off[j + 1]
            while a < a1 and b < b1:
                if flat[a] == flat[b]:
                    o += 1
                    a += 1
                    b += 1
                elif flat[a] < flat[b]:
                    a += 1
                else:
                    b += 1
            u = li + lj - o
            v = o / u if u else 0.0
            J[i, j] = v
            J[j, i] = v
    return J


def subset_arrays(w, docs, off, flat):
    lens = off[docs + 1] - off[docs]
    so = np.zeros(len(docs) + 1, np.int64)
    np.cumsum(lens, out=so[1:])
    sf = np.empty(int(so[-1]), np.uint64)
    for k, d in enumerate(docs):
        sf[int(so[k]):int(so[k + 1])] = flat[off[d]:off[d + 1]]
    return so, sf


def main():
    t0 = time.time()
    w = "wet"
    tau = 0.8
    off, flat, _ = K.load_shingles(w)
    n_all = len(off) - 1
    ti, tj, jac = K.compute_truth(w, tau, off)
    # components of the truth graph, to include real duplicate clusters
    root = K.uf_minid(n_all, ti, tj)
    size = np.bincount(root, minlength=n_all)
    order = np.argsort(-size, kind="stable")
    chosen = []
    seen = set()
    for d in order:
        if size[d] < 2:
            break
        comp = np.flatnonzero(root == root[d])
        for c in comp:
            c = int(c)
            if c not in seen:
                seen.add(c)
                chosen.append(c)
        if len(chosen) >= 700:
            break
    rng = np.random.RandomState(20261119)
    while len(chosen) < 1000:
        c = int(rng.randint(n_all))
        if c not in seen:
            seen.add(c)
            chosen.append(c)
    docs = np.array(sorted(chosen), dtype=np.int64)
    so, sf = subset_arrays(w, docs, off, flat)
    n = len(docs)
    print("subset", n, "docs", int(so[-1]), "shingles, largest included component",
          int(size[order[0]]))

    J = jaccard_matrix(so, sf, n)
    ii, jj = np.triu_indices(n, 1)
    brute = {(int(a), int(b)) for a, b, v in zip(ii.tolist(), jj.tolist(), J[ii, jj].tolist())
             if v >= tau - 1e-12}
    print("brute-force edges", len(brute), "in", round(time.time() - t0, 1), "s")

    # ---------------- C: full exact self-join on the subset
    R, S, V = K.EJ.build_ranks(so, sf)
    res = K.EJ.exact_join(so, R, S, V, tau, K.THREADS)
    c_edges = {(int(a), int(b)) for a, b in zip(res["i"].tolist(), res["j"].tolist())}
    c_equal = c_edges == brute

    # ---------------- B: LSH candidates + exact verification
    sig = K.L.signatures(so, sf, K.H, 1, "dt")
    Kb = K.band_keys(sig)
    pi, pj, bstats = K.all_bucket_pairs(Kb, np.ones(n, bool), n)
    cand = {(int(a), int(b)) for a, b in zip(pi.tolist(), pj.tolist())}
    jv = K.L.jaccard_pairs(so, sf, pi, pj)
    b_edges = {(int(a), int(b)) for a, b, v in zip(pi.tolist(), pj.tolist(), jv.tolist())
               if v >= tau - 1e-12}
    b_equal = b_edges == (brute & cand)
    missed_by_lsh = brute - cand

    # ---------------- A: chain edges (for reference)
    ei, ej, astats = K.chain_edges(Kb, np.ones(n, bool))
    a_edges = {(int(a), int(b)) for a, b in zip(ei.tolist(), ej.tolist())}
    a_subset_of_cand = a_edges <= cand

    out = dict(
        evidence_tier="rerun", generated_utc=K.now(), workload=w, tau=tau,
        subset_n=n, subset_shingles=int(so[-1]), threads=K.THREADS, hash_seed=1,
        selection="docs of the largest truth components first, then random (seed 20261119) up to 1000",
        brute_edges=len(brute), brute_seconds=round(time.time() - t0, 2),
        C=dict(edges=len(c_edges), equals_bruteforce=c_equal,
               symmetric_difference=len(c_edges ^ brute)),
        B=dict(candidate_edges=len(cand), verified_edges=len(b_edges),
               equals_bruteforce_intersect_candidates=b_equal,
               symmetric_difference_vs_brute_and_cand=len(b_edges ^ (brute & cand)),
               lsh_missed_true_edges=len(missed_by_lsh),
               candidate_recall=len(brute & cand) / len(brute) if brute else None,
               **bstats),
        A=dict(chain_edges=len(a_edges), chain_subset_of_all_pairs=a_subset_of_cand,
               chain_true_edges=len(a_edges & brute), **astats),
        all_checks_pass=bool(c_equal and b_equal and a_subset_of_cand),
    )
    os.makedirs(K.CORE, exist_ok=True)
    p = os.path.join(K.CORE, "crosscheck_bruteforce.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
