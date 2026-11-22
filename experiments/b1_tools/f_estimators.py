#!/usr/bin/env python3
"""F: pre-deployment estimators, validated against the full computation on one corpus.

Cost: for sample sizes s in {1000, 2000, 5000} and 20 random starts each (plus the frozen
evenly spaced sample), compute 14x8 band buckets (seed 1) on the sample only and estimate
  sum_b C(m_b, 2) by sample_pairs * n(n-1) / (s(s-1)),  largest bucket by sample_max * n / s.
Soundness: a stratified sample of ~1,000 of path A's deletions (seed 1, min-id), strata =
shingle-count quintile x largest-bucket class. For each sampled deletion, look for a neighbour
with J >= 0.8 using a high-recall 64x4 banding (256 permutations, seed 101) plus exact Jaccard;
estimate the unsupported share with Horvitz-Thompson weights and a one-sided 95% upper bound;
calibrate against the exact truth (misses of the high-recall search).
Usage: env/bin/python f_estimators.py <corpus name under runs/>   (needs b1_rung.py output)
"""
import json
import sys

import numpy as np

W = "/path/to/scratch/dedupjoin"
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run"]
import snap_common as S  # noqa: E402

TAU = 0.8


def bucket_stats(kb):
    pairs, largest = 0, 0
    for b in range(kb.shape[0]):
        _, c = np.unique(kb[b], return_counts=True)
        c = c.astype(np.int64)
        pairs += int((c * (c - 1) // 2).sum())
        largest = max(largest, int(c.max()))
    return pairs, largest


def main():
    name = sys.argv[1]
    d = f"{W}/runs/{name}"
    S.SHINGLE_NPZ, S.CACHE, S.SNAP = f"{d}/snapshot.npz", f"{d}/cache", d
    import collapsed_lib as CL
    K, L = CL.K, CL.K.L
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    group, m, nsh = z["group_of_doc"], z["group_size"], z["group_nshingle"]
    off_u, flat_u = z["off_u"], z["flat_u"]
    n, G = len(group), len(m)
    valid = nsh[group] > 0
    out = open(f"{W}/results/F_{name}.jsonl", "a")

    # ---------------- cost estimator
    kb_g = CL.group_band_keys(z, 1)
    kb = kb_g[:, group]
    full_pairs, full_max = bucket_stats(kb[:, valid])
    rng = np.random.default_rng(7)
    vidx = np.flatnonzero(valid)
    nv = len(vidx)
    for s in (1000, 2000, 5000):
        designs = [("even", np.array([vidx[(k * nv) // s] for k in range(s)]))]
        designs += [(f"random{r}", np.sort(rng.choice(vidx, s, replace=False))) for r in range(20)]
        for label, idx in designs:
            sp, sm = bucket_stats(kb[:, idx])
            est_pairs = sp * nv * (nv - 1) / (s * (s - 1))
            est_max = sm * nv / s
            out.write(json.dumps(dict(kind="cost", corpus=name, s=s, design=label, sample_pairs=sp,
                                      sample_max=sm, est_pairs=est_pairs, full_pairs=full_pairs,
                                      rel_err_pairs=est_pairs / full_pairs - 1 if full_pairs else None,
                                      est_max=est_max, full_max=full_max, rel_err_max=est_max / full_max - 1,
                                      pairs_per_doc_sample=sp / s)) + "\n")
    del kb

    # ---------------- soundness estimator
    ti, tj, _ = CL.truth_u(TAU)
    st = CL.exact_state(z, ti, tj)
    ei, ej, _ = K.chain_edges(kb_g[:, group], valid)
    rem = (K.uf_minid(n, ei, ej) != np.arange(n)) & valid
    del ei, ej
    D = np.flatnonzero(rem)
    y_true = ~st["has_nb_doc"][D]
    # strata: shingle-count quintile x largest bucket containing the document (seed-1 bands)
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
    # high-recall search: 256 permutations, 64 bands x 4 rows, on groups
    sig = L.signatures(off_u, flat_u, 256, 101, "dt")
    hk = L.band_keys(sig, 64, 4)
    del sig
    order = [np.argsort(hk[b], kind="stable") for b in range(64)]
    sorted_keys = [hk[b][order[b]] for b in range(64)]
    found = np.zeros(len(sample), bool)
    for k, di in enumerate(sample):
        g = group[D[di]]
        if m[g] >= 2:
            found[k] = True
            continue
        cand = set()
        for b in range(64):
            lo = np.searchsorted(sorted_keys[b], hk[b, g], "left")
            hi = np.searchsorted(sorted_keys[b], hk[b, g], "right")
            cand.update(order[b][lo:hi].tolist())
        cand.discard(g)
        if cand:
            c = np.fromiter(cand, np.int64)
            jac = L.jaccard_pairs(off_u, flat_u, np.full(len(c), g, np.int64), c)
            found[k] = bool((jac >= TAU - 1e-12).any())
    y = ~found
    yt = y_true[sample]
    est, var = 0.0, 0.0
    for h, Nh in zip(labels, N_h):
        sel = stratum[sample] == h
        nh = int(sel.sum())
        yh = y[sel].astype(float)
        est += Nh / len(D) * yh.mean()
        if nh > 1:
            var += (Nh / len(D)) ** 2 * (1 - nh / Nh) * yh.var(ddof=1) / nh
    se = var ** 0.5
    out.write(json.dumps(dict(kind="soundness", corpus=name, tau=TAU, deletions=int(len(D)),
                              sample=int(len(sample)), strata=int(len(labels)), estimate=est,
                              se=se, upper95=est + 1.645 * se, true_share=float(y_true.mean()),
                              disagreements_vs_truth=int((y != yt).sum()),
                              false_unsupported=int((y & ~yt).sum()), false_supported=int((~y & yt).sum()))) + "\n")
    out.flush()
    print(f"F {name}: estimate {est:.4f} +- {se:.4f}, true {y_true.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
