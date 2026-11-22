#!/usr/bin/env python3
"""Where does the seed-to-seed variance of pair recall come from?

The per-component caches `<corpus>_comp_s<seed>_t<tau>.npz` store, for every exact
connected component, the number of true edges (`ecount`) and the number of those
edges that the 14x8 banding made into candidates for that seed (`hits`).
With E = sum(ecount) and R_s = sum(hits_s) / E:

    Var(R) = sum_c (e_c/E)^2 Var_s(h_c/e_c)          (within-component terms)
           + sum_{c != c'} (e_c e_c'/E^2) Cov_s(p_c, p_c')   (between components)

We report the measured cross-seed variance, the sum of per-component terms, the
leftover covariance term, the largest component's share of the component terms,
the variance after removing the largest component, and the theoretical upper
bound sum_c (e_c/E)^2 pbar_c (1-pbar_c) that obtains when all edges inside a
component flip together.

Usage: python3 run_variance.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correctness_lib as K

TAUS = [0.7, 0.8, 0.9]
SEED_LIMIT = {"c4": 100, "fw": 20, "mix": 20, "wet": 20}


def comp_files(corpus, tau):
    out = []
    for f in os.listdir(K.WRK):
        pre = f"{corpus}_comp_s"
        if f.startswith(pre) and f.endswith(f"_t{tau}.npz"):
            seed = int(f[len(pre):].split("_")[0])
            out.append((seed, os.path.join(K.WRK, f)))
    return sorted(out)


def analyse(corpus, tau):
    files = comp_files(corpus, tau)
    if not files:
        return None
    e = None
    hits = []
    seeds = []
    for seed, path in files:
        z = np.load(path)
        h = np.asarray(z["hits"], np.float64)
        if e is None:
            e = np.asarray(z["ecount"], np.float64)
        else:
            assert len(h) == len(e)
        z.close()
        hits.append(h)
        seeds.append(seed)
    H = np.vstack(hits)                      # K x C
    E = float(e.sum())
    R = H.sum(axis=1) / E
    K_runs = len(seeds)
    var_total = float(R.var(ddof=1)) if K_runs > 1 else 0.0
    p = H / np.maximum(e, 1.0)               # K x C hit rate
    pbar = p.mean(axis=0)
    var_c = p.var(axis=0, ddof=1) if K_runs > 1 else np.zeros_like(pbar)
    w = (e / E) ** 2
    contrib = w * var_c
    s_comp = float(contrib.sum())
    cov_term = float(var_total - s_comp)
    imax = int(np.argmax(e))
    cmax = float(contrib[imax])
    share_max = float(cmax / s_comp) if s_comp > 0 else None
    share_max_total = float(cmax / var_total) if var_total > 0 else None
    cov_share = float(cov_term / var_total) if var_total > 0 else None
    # remove the largest component
    if E - e[imax] > 0:
        R2 = (H.sum(axis=1) - H[:, imax]) / (E - e[imax])
        var_wo = float(R2.var(ddof=1)) if K_runs > 1 else 0.0
    else:
        var_wo = 0.0
    bound = float((w * pbar * (1 - pbar)).sum())
    p_mean = float(R.mean())
    binom = p_mean * (1 - p_mean) / E if E > 0 else None
    return dict(
        corpus=corpus, tau=tau, n_seeds=K_runs, seeds=seeds,
        n_components=int(len(e)), true_edges=int(E),
        pair_recall=dict(mean=p_mean, sd=float(R.std(ddof=1)) if K_runs > 1 else 0.0,
                         min=float(R.min()), max=float(R.max())),
        variance=dict(
            measured_total=var_total,
            sum_component_terms=s_comp,
            component_share=s_comp / var_total if var_total > 0 else None,
            covariance_term=cov_term,
            covariance_share=cov_share,
            largest_component_edges=int(e[imax]),
            largest_component_terms=cmax,
            largest_component_share_of_component_terms=share_max,
            largest_component_share_of_measured_total=share_max_total,
            measured_total_without_largest=var_wo,
            measured_change_after_removal=var_total - var_wo,
            measured_change_after_removal_share=((var_total - var_wo) / var_total
                                                 if var_total > 0 else None),
            theoretical_upper_bound=bound,
            measured_over_bound=var_total / bound if bound > 0 else None,
            binomial_null_variance=binom,
            overdispersion=float(R.std(ddof=1) / np.sqrt(binom)) if binom else None,
        ),
        notes=dict(
            terms="(e_c/E)^2 * Var_s(hits_c/e_c), sample variance over seeds",
            covariance="measured_total - sum_component_terms = sum of between-component covariances",
            bound="sum_c (e_c/E)^2 pbar_c (1-pbar_c): the largest variance a component with marginal hit rate pbar_c can contribute; attained only when all its edges flip together (p_c in {0,1})",
            ratio_def="each component's term divided by the sum of component terms",
        ),
    )


def main():
    out = {}
    rows = []
    for corpus in K.CORPORA:
        for tau in TAUS:
            rec = analyse(corpus, tau)
            if rec is None:
                continue
            out[f"{corpus}|t{tau}"] = rec
            rows.append(rec)
            v = rec["variance"]
            def f(x):
                return "n/a" if x is None else f"{x:.4g}"
            print(f"[{corpus} t{tau}] K={rec['n_seeds']} sd={f(rec['pair_recall']['sd'])} "
                  f"comp_share={f(v['component_share'])} cov_share={f(v['covariance_share'])} "
                  f"max_share={f(v['largest_component_share_of_component_terms'])} "
                  f"over_bound={f(v['measured_over_bound'])}", flush=True)
    path = os.path.join(K.OUTDIR, "variance_decomposition.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(cells=out, generated_utc=K.now()), f, indent=1, ensure_ascii=False)
    with open(os.path.join(K.OUTDIR, "variance_decomposition_rows.jsonl"), "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
