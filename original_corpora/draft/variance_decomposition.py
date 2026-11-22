#!/usr/bin/env python3
"""Exact decomposition of the across-seed variance of pair recall.

For the 14x8 banding on one corpus and threshold, the per-seed hit counts per exact
component are saved by seed_variance.py.  With H (seeds x components) and edge counts
E_c, the variance of the total hit count decomposes exactly as

    Var(sum_c H_c) = sum_c Var(H_c) + sum_{c != c'} Cov(H_c, H_c'),

so the share of the variance carried by covariance between components measures how much
components flip together, as opposed to independent edge-level sampling.  Two reference
points are reported: the independent-edge binomial variance, p(1-p)/n, and the
all-or-nothing bound sum_c E_c^2 p_c(1-p_c) / (sum E)^2, which assumes every edge of a
component shares one hit.

Usage: python3 variance_decomposition.py [--tau 0.8]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03"))
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")


def one(corpus, tau):
    files = sorted(glob.glob(os.path.join(WRK, f"{corpus}_comp_s*_t{tau}.npz")))
    if len(files) < 3:
        return None
    H = np.array([np.load(f)["hits"] for f in files], np.float64)
    E = np.load(files[0])["ecount"].astype(np.float64)
    K = len(files)
    tot = E.sum()
    hits = H.sum(axis=1)
    var_total = float(np.var(hits, ddof=1) / tot ** 2)
    var_within = float(np.var(H, axis=0, ddof=1).sum() / tot ** 2)
    p = hits.sum() / (K * tot)
    binom = p * (1 - p) / tot
    pc = H.mean(axis=0) / np.maximum(E, 1)
    allornothing = float((E ** 2 * pc * (1 - pc)).sum() / tot ** 2)
    per_var = np.var(H, axis=0, ddof=1)  # per-component variance of hit counts
    order = np.argsort(-per_var)
    top = [dict(rank=int(i), edges=int(E[i]), mean_rate=float(pc[i]),
                var_hits=float(per_var[i]),
                var_share=float(per_var[i] / per_var.sum()) if per_var.sum() else None)
           for i in order[:10]]
    # pairwise correlation of component hit counts among the 20 largest components
    big = np.argsort(-E)[:20]
    with np.errstate(invalid="ignore", divide="ignore"):
        Cm = np.corrcoef(H[:, big].T)
    iu = np.triu_indices(len(big), 1)
    corr = Cm[iu]
    corr = corr[np.isfinite(corr)]
    return dict(
        corpus=corpus, tau=float(tau), n_seeds=K, n_components=int(len(E)),
        n_truth_pairs=int(tot),
        mean_recall=float(hits.mean() / tot),
        observed_sd=float(np.sqrt(var_total)),
        binomial_sd=float(np.sqrt(binom)),
        overdispersion=float(np.sqrt(var_total / binom)),
        within_component_var_share=float(var_within / var_total),
        covariance_var_share=float(1 - var_within / var_total),
        all_or_nothing_bound_sd=float(np.sqrt(allornothing)),
        bound_over_observed=float(np.sqrt(allornothing / var_total)),
        top_component_var_share=[t["var_share"] for t in top],
        top_components=top,
        mean_corr_top20=float(np.nanmean(corr)) if len(corr) else None,
        median_corr_top20=float(np.nanmedian(corr)) if len(corr) else None,
        n_eff_edges=float(tot ** 2 / (E ** 2).sum()),
        provenance="independently rerun (this run)",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="0.8")
    a = ap.parse_args()
    out = {"tau": float(a.tau), "provenance": "independently rerun (this run)",
           "cells": {}}
    for corpus in ("c4", "fw", "mix", "wet"):
        r = one(corpus, a.tau)
        if r:
            out["cells"][corpus] = r
            print(corpus, "sd %.5f binom %.5f od %.1fx within %.2f cov %.2f bound/obs %.2f corr %.3f"
                  % (r["observed_sd"], r["binomial_sd"], r["overdispersion"],
                     r["within_component_var_share"], r["covariance_var_share"],
                     r["bound_over_observed"], r["mean_corr_top20"] or float("nan")))
    with open(os.path.join(OUT, "variance_decomposition.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
