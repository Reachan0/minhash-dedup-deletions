#!/usr/bin/env python3
"""Repeated-sampling coverage of the soundness audit of Section 10 (f_estimators.py design).

For a rung: A's deletions (seed 1, min-id, tau 0.8), the same strata (shingle-count quintile x class
of the largest seed-1 bucket containing the document), proportional allocation of 1,000 with at least
5 per stratum, simple random sampling without replacement in each stratum. Labels are the exact ones
(no neighbour at J >= 0.8), so the simulation measures sampling error only; the high-recall search
agreed with them on every audited document. The number of unsupported documents drawn from a stratum
is hypergeometric, so R repetitions are simulated exactly from the stratum counts.
Estimator: p = sum_h W_h ybar_h, var = sum_h W_h^2 (1 - n_h/N_h) s_h^2 / n_h, upper = p + 1.645 sqrt(var).
Writes results/coverage_sim.jsonl (stratum counts and coverage).
Usage: env/bin/python coverage_sim.py <rung N> [R]
"""
import json
import sys

import numpy as np

W = "/path/to/scratch/dedupjoin"
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run"]
import snap_common as S  # noqa: E402


def strata_counts(name):
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
    ei, ej, _ = K.chain_edges(kb_g[:, group], valid)
    rem = (K.uf_minid(n, ei, ej) != np.arange(n)) & valid
    del ei, ej
    ti, tj, _ = CL.truth_u(0.8)
    st = CL.exact_state(z, ti, tj)
    D = np.flatnonzero(rem)
    y = ~st["has_nb_doc"][D]
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
    K_h = np.array([int(y[stratum == h].sum()) for h in labels])
    n_h = np.minimum(np.maximum(5, np.round(1000 * N_h / N_h.sum()).astype(int)), N_h)
    return N_h, K_h, n_h, float(y.mean())


def simulate(N_h, K_h, n_h, R, seed=7):
    rng = np.random.default_rng(seed)
    Wh = N_h / N_h.sum()
    k = np.stack([rng.hypergeometric(K_h[i], N_h[i] - K_h[i], n_h[i], size=R) for i in range(len(N_h))], 1)
    ybar = k / n_h
    s2 = np.where(n_h > 1, n_h / np.maximum(n_h - 1, 1) * ybar * (1 - ybar), 0.0)
    p = (Wh * ybar).sum(1)
    var = (Wh ** 2 * (1 - n_h / N_h) * s2 / n_h).sum(1)
    return p, p + 1.645 * np.sqrt(var)


def main():
    N = int(sys.argv[1])
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    N_h, K_h, n_h, true = strata_counts(f"rung_{N}")
    p, up = simulate(N_h, K_h, n_h, R)
    row = dict(corpus=f"rung_{N}", R=R, true_share=true, deletions=int(N_h.sum()), sample=int(n_h.sum()),
               strata=int(len(N_h)), N_h=N_h.tolist(), K_h=K_h.tolist(), n_h=n_h.tolist(),
               mean_estimate=float(p.mean()), sd_estimate=float(p.std()), coverage_upper95=float((up >= true).mean()),
               median_upper=float(np.median(up)), share_estimate_below_true=float((p < true).mean()))
    with open(f"{W}/results/coverage_sim.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("N_h", "K_h", "n_h")}), flush=True)


if __name__ == "__main__":
    main()
