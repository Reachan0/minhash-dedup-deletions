#!/usr/bin/env python3
"""Pre-run features that predict whether a corpus is in the high-variance regime.

For each corpus and tau, compute from the exact truth alone (before any hash seed is
drawn): number of truth pairs and components, the largest component, the size-weighted
component, the share of pair mass in the ten largest components, the share of pairs
within 0.05 of the threshold, the effective number of independent edges
(sum E_c^2 / sum E_c), and the design-effect prediction of the recall standard
deviation.  Join them with the observed across-seed standard deviation from
results/seed_variance.json and report the correlation.

Usage: python3 regime_features.py [--tau 0.8]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03"))
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")


def features(corpus, tau):
    n = MC.CORPORA[corpus]
    t = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
    ti, tj, jac = np.asarray(t["i"]), np.asarray(t["j"]), np.asarray(t["jac"])
    root = np.asarray(t["root"])
    comp = np.bincount(root, minlength=n)
    docs_comp = comp[comp >= 2]
    E = np.bincount(root[ti])[comp >= 2] if len(ti) else np.zeros(0)
    pc = None
    # across-seed mean per-component hit rate, if the component files exist
    import glob
    files = sorted(glob.glob(os.path.join(WRK, f"{corpus}_comp_s*_t{tau}.npz")))
    if files:
        H = np.array([np.load(f)["hits"] for f in files], np.float64)
        Ec = np.load(files[0])["ecount"].astype(np.float64)
        pc = H.mean(axis=0) / np.maximum(Ec, 1)
        var_c = (Ec ** 2 * pc * (1 - pc)).sum() / max(Ec.sum() ** 2, 1)
        pred_std = float(np.sqrt(var_c))
    else:
        Ec, pred_std = np.zeros(0), None
    order = np.argsort(-Ec) if len(Ec) else np.zeros(0, int)
    near = (jac >= float(tau)) & (jac < float(tau) + 0.05)
    out = dict(
        n_docs=int(n), n_truth_pairs=int(len(ti)),
        n_components=int(len(docs_comp)), max_component=int(docs_comp.max()) if len(docs_comp) else 0,
        size_weighted_component=float((docs_comp.astype(np.float64) ** 2).sum() / docs_comp.sum())
        if len(docs_comp) else None,
        top10_pair_share=float(Ec[order[:10]].sum() / Ec.sum()) if len(Ec) else None,
        near_threshold_share=float(near.mean()) if len(jac) else None,
        n_eff_edges=float((Ec ** 2).sum() / Ec.sum()) if len(Ec) else None,
        design_effect_pred_std=pred_std,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="0.8")
    a = ap.parse_args()
    sv = json.load(open(os.path.join(OUT, "seed_variance.json")))["cells"]
    out = {"tau": float(a.tau), "provenance": "independently rerun (this run)",
           "corpora": {}, "note": ("features use the exact truth only; observed std from "
                                   "seed_variance.json")}
    for corpus in ("c4", "fw", "mix", "wet"):
        p = os.path.join(WRK, f"{corpus}_truth_t{a.tau}.npz")
        if not os.path.exists(p):
            continue
        f = features(corpus, a.tau)
        cell = sv.get(f"{corpus}|14x8|{a.tau}")
        if cell:
            f.update(observed_std=cell["pair_recall"]["std"],
                     observed_min=cell["pair_recall"]["min"],
                     binomial_se=cell["binomial_se"],
                     overdispersion=cell["overdispersion"],
                     design_effect_ratio=cell["design_effect_ratio"])
        out["corpora"][corpus] = f
    with open(os.path.join(OUT, "regime_features.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
