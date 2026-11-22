#!/usr/bin/env python3
"""Controlled synthetic corpora that isolate the seed-variance mechanism.

Documents are shingle sets, not text, so the exact Jaccard structure is fully controlled:
each planted cluster has a base set, and every copy keeps a fraction of it and adds fresh
shingles, which fixes the Jaccard value of the copy against the base.  Four regimes:

  uniform     2,000 clusters of 10 copies, J=0.85 (far above tau)
  threshold   all copies at J in [0.795, 0.805] (around tau=0.8)
  heavy       cluster sizes drawn from a Zipf law (2..500), J in [0.82, 0.92]
  far         all copies at J=0.95

For each regime and hash seed the datatrove 14x8 pipeline is run and pair-, document- and
cluster-weighted recall are recorded, together with the design-effect prediction computed
from the exact components.  This tests the claim that overdispersion comes from pair mass
concentrated in a few exact components, not from the hash family being biased.

Usage: python3 synthetic_variance.py --seeds 24
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
C03 = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, C03)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
TAU = 0.8


def build_corpus(regime, seed=0):
    """Every pair of documents inside a planted cluster has exactly the target Jaccard.

    A cluster is a common core of s shingles plus a private part of m-s shingles per
    document; two documents of size m sharing s have J = s/(2m-s), so the pairwise
    similarity is set directly by s.
    """
    rng = np.random.default_rng(20261118 + {"uniform": 1, "threshold": 2, "far": 3, "heavy": 4}[regime])
    sets = []
    cluster_sizes = []
    if regime == "uniform":
        sizes = [10] * 2000
        jacs = [0.85] * 2000
    elif regime == "threshold":
        sizes = [10] * 2000
        jacs = list(rng.uniform(0.795, 0.805, 2000))
    elif regime == "far":
        sizes = [10] * 2000
        jacs = [0.95] * 2000
    elif regime == "heavy":
        sizes, tot = [], 0
        while tot < 20000:
            s = int(np.clip(rng.zipf(1.7), 2, 500))
            sizes.append(s)
            tot += s
        jacs = list(rng.uniform(0.82, 0.92, len(sizes)))
    else:
        raise ValueError(regime)
    for s, jac in zip(sizes, jacs):
        m = 180 if regime != "heavy" else int(rng.integers(120, 260))
        core_n = int(round(jac * 2 * m / (1 + jac)))
        core_n = min(max(core_n, 1), m - 1)
        core = rng.integers(0, 2 ** 62, core_n, dtype=np.uint64)
        for _ in range(s):
            private = rng.integers(0, 2 ** 62, m - core_n, dtype=np.uint64)
            doc = np.unique(np.concatenate([core, private]))
            sets.append(doc)
        cluster_sizes.append((s, jac))
    # background documents with no duplicates
    for _ in range(5000):
        m = int(rng.integers(120, 260))
        sets.append(np.unique(rng.integers(0, 2 ** 62, m, dtype=np.uint64)))
    rng.shuffle(sets)
    off = np.zeros(len(sets) + 1, np.int64)
    for i, s in enumerate(sets):
        off[i + 1] = off[i] + len(s)
    flat = np.concatenate(sets)
    return off, flat, cluster_sizes


def run_regime(regime, seeds, threads):
    C.set_threads(threads)
    off, flat, cluster_sizes = build_corpus(regime)
    n = len(off) - 1
    R, S, V = EJ.build_ranks(off, flat)
    res = EJ.exact_join(off, R, S, V, TAU, threads)
    ti, tj = res["i"].astype(np.int64), res["j"].astype(np.int64)
    root = L.uf_components(n, ti, tj)
    rem = L.removed_set(root)
    comp_doc = np.bincount(root, minlength=n)
    edge_comp = root[ti]
    ecomp_u, ecomp_idx = np.unique(edge_comp, return_inverse=True)
    ecount = np.bincount(ecomp_idx)
    csize = comp_doc[ecomp_u]
    truth = dict(ti=ti, tj=tj, n=len(ti), root=root, rem=rem,
                 edge_comp=ecomp_idx, ecount=ecount, csize=csize, n_comp=len(ecomp_u),
                 max_component=int(comp_doc.max()),
                 size_weighted=float((comp_doc.astype(np.float64) ** 2).sum() / comp_doc.sum()))
    per_seed = []
    valid = np.diff(off) > 0
    for seed in seeds:
        sig = L.signatures(off, flat, 112, seed, "dt")
        rec = {}
        for (b, r) in [(14, 8), (20, 5)]:
            K = L.band_keys(sig, b, r)
            codes, ci, cj, _ = L.lsh_pairs(K, valid, 50_000_000)
            root_raw = L.uf_components(n, ci, cj)
            rrem = L.removed_set(root_raw) & valid
            hit = L.truth_hits(sig, ti, tj, b, r).astype(np.float64)
            if len(edge_comp):
                order = np.argsort(edge_comp, kind="stable")
                cc = edge_comp[order]
                hh = hit[order]
                brk = np.flatnonzero(np.r_[True, cc[1:] != cc[:-1]])
                per_comp = np.add.reduceat(hh, brk) / np.diff(np.r_[brk, len(cc)])
            else:
                per_comp = np.zeros(0)
            rec[f"{b}x{r}"] = dict(
                pair_recall=float(hit.mean()) if len(hit) else None,
                cluster_weighted=float(per_comp.mean()) if len(per_comp) else None,
                doc_recall=float((rrem & rem).sum() / rem.sum()) if rem.sum() else None,
                deleted=int(rrem.sum()))
        per_seed.append(rec)
    # design-effect prediction for 14x8
    out = dict(regime=regime, n_docs=int(n), n_truth_pairs=truth["n"],
               n_components=truth["n_comp"], max_component=truth["max_component"],
               size_weighted_docs=truth["size_weighted"], per_seed=per_seed)
    for cfg in ("14x8", "20x5"):
        pr = np.array([s[cfg]["pair_recall"] if s[cfg]["pair_recall"] is not None else np.nan
                       for s in per_seed])
        dr = np.array([s[cfg]["doc_recall"] if s[cfg]["doc_recall"] is not None else np.nan
                       for s in per_seed])
        cw = np.array([s[cfg]["cluster_weighted"] if s[cfg]["cluster_weighted"] is not None else np.nan
                       for s in per_seed])
        p = float(np.nanmean(pr)) if np.isfinite(pr).any() else None
        binom = float(np.sqrt(p * (1 - p) / truth["n"])) if p is not None and truth["n"] else None
        out[cfg] = dict(mean=p, std=float(np.nanstd(pr, ddof=1)) if p is not None else None,
                        min=float(np.nanmin(pr)) if p is not None else None,
                        binomial_se=binom,
                        overdispersion=float(np.nanstd(pr, ddof=1) / binom) if binom else None,
                        cluster_weighted_mean=float(np.nanmean(cw)),
                        cluster_weighted_std=float(np.nanstd(cw, ddof=1)),
                        doc_recall_mean=float(np.nanmean(dr)) if np.isfinite(dr).any() else None,
                        doc_recall_std=float(np.nanstd(dr, ddof=1)))
    # exact design-effect for 14x8
    hits_all = []
    for seed in seeds:
        sig = L.signatures(off, flat, 112, seed, "dt")
        h = L.truth_hits(sig, ti, tj, 14, 8).astype(np.float64)
        hits_all.append(np.bincount(truth["edge_comp"], weights=h,
                                    minlength=truth["n_comp"]))
    H = np.array(hits_all)
    pc = H.mean(axis=0) / np.maximum(ecount, 1)
    if ecount.sum():
        var_c = (ecount.astype(np.float64) ** 2 * pc * (1 - pc)).sum() / ecount.sum() ** 2
        out["design_effect_pred_std"] = float(np.sqrt(var_c))
    else:
        out["design_effect_pred_std"] = None
    out["provenance"] = "independently rerun (this run)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--threads", type=int, default=15)
    a = ap.parse_args()
    out = {"tau": TAU, "seeds": a.seeds, "regimes": {}, "provenance": "independently rerun (this run)"}
    for regime in ("uniform", "threshold", "heavy", "far"):
        t0 = time.time()
        r = run_regime(regime, list(range(1, a.seeds + 1)), a.threads)
        r["t_s"] = time.time() - t0
        out["regimes"][regime] = r
        print(json.dumps({regime: {k: r[k] for k in ("n_truth_pairs", "n_components", "14x8", "20x5", "design_effect_pred_std")}}), flush=True)
    with open(os.path.join(OUT, "synthetic_variance.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
