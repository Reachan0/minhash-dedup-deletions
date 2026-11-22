#!/usr/bin/env python3
"""Seed-to-seed variance of the LSH decision, and the mechanism behind it.

`run` recomputes, for each hash seed, the datatrove 14x8 banding (plus 20x5 and 10x10 as
configuration controls), the candidate pair set, the connected components and the deleted
set, and records pair-weighted, document-weighted and cluster-weighted recall together with
per-component / per-size-bin / per-domain edge-hit aggregates.

`analyze` turns those rows into results/seed_variance.json: overdispersion against the
binomial null, the design-effect prediction from exact-component structure, tail recall,
per-cluster contributions, keep/drop flips and their token / unique-content weight.

Usage:
  python3 seed_variance.py run --corpus c4 --seeds 1..100
  python3 seed_variance.py analyze
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
C03 = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, C03)
sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import lsh as L  # noqa: E402
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
ROWS = os.path.join(OUT, "seed_variance_rows.jsonl")
CONFIGS = [(14, 8), (20, 5), (10, 10)]
TAUS = [0.7, 0.8, 0.9]
SIZE_BINS = [(2, 2), (3, 4), (5, 15), (16, 99), (100, 10 ** 9)]
TOPK_COMP = 50
TOPK_DOM = 200


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def append(row):
    row = dict(row)
    row.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("provenance", "independently rerun (this run)")
    os.makedirs(OUT, exist_ok=True)
    with open(ROWS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(corpus, seeds, threads):
    n = MC.CORPORA[corpus]
    C.set_threads(threads)
    off, flat, nw, meta = MC.load_shingles(corpus)
    z = np.load(os.path.join(WRK, f"{corpus}_ranks.npz"))
    uniq = np.asarray(z["uniq"])
    valid = (np.diff(off) > 0)
    meta_t = pq.read_table(os.path.join(WRK, f"{corpus}_meta.parquet")).to_pydict()
    domain = np.array(meta_t["domain"])
    # inverse domain ids
    dom_u, dom_inv = np.unique(domain, return_inverse=True)

    truth = {}
    for tau in TAUS:
        q = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
        ti, tj, jac = np.asarray(q["i"]), np.asarray(q["j"]), np.asarray(q["jac"])
        root, rem = np.asarray(q["root"]), np.asarray(q["removed"])
        comp_doc = np.bincount(root, minlength=n)
        edge_comp = root[ti]
        ecomp_u, ecomp_idx = np.unique(edge_comp, return_inverse=True)
        ecount = np.bincount(ecomp_idx)
        csize = comp_doc[ecomp_u]
        # size bins over components (edge weighted)
        bin_id = np.zeros(len(ecomp_u), np.int32)
        for k, (lo, hi) in enumerate(SIZE_BINS):
            bin_id[(csize >= lo) & (csize <= hi)] = k
        edge_bin = bin_id[ecomp_idx]
        top = np.argsort(-ecount)[:TOPK_COMP]
        edge_top = np.full(len(ti), -1, np.int32)
        for k, c in enumerate(top):
            edge_top[ecomp_idx == c] = k
        dom_pairs = dom_inv[ti]
        d_u, d_cnt = np.unique(dom_pairs, return_counts=True)
        d_top = np.argsort(-d_cnt)[:TOPK_DOM]
        dom_map = {int(d): k for k, d in enumerate(d_top)}
        edge_dom = np.array([dom_map.get(int(d), -1) for d in dom_pairs], np.int32)
        near = (jac >= tau) & (jac < tau + 0.05)
        truth[tau] = dict(ti=ti, tj=tj, jac=jac, root=root, rem=rem,
                          edge_comp=ecomp_idx, ecount=ecount, csize=csize,
                          edge_bin=edge_bin, edge_top=edge_top, edge_dom=edge_dom,
                          near=near, n=len(ti),
                          n_eff=float(ecount.sum() ** 2
                                      / max(1.0, (ecount.astype(np.float64) ** 2).sum())),
                          size_weighted_component=float((ecount.astype(np.float64) ** 2).sum()
                                                        / max(1, ecount.sum())),
                          n_comp=len(ecomp_u))

    for seed in seeds:
        t0 = time.time()
        sig = L.signatures(off, flat, 112, seed, "dt")
        row = dict(kind="seed", corpus=corpus, seed=seed, configs={},
                   t_sig_s=time.time() - t0)
        for (b, r) in CONFIGS:
            K = L.band_keys(sig, b, r)
            codes, ci, cj, tot = L.lsh_pairs(K, valid, 400_000_000)
            del K
            root_raw = L.uf_components(n, ci, cj)
            rem = L.removed_set(root_raw) & valid
            cfg = f"{b}x{r}"
            cres = {}
            for tau in TAUS:
                T = truth[tau]
                hit = L.truth_hits(sig, T["ti"], T["tj"], b, r).astype(np.float64)
                tot_hit = hit.sum()
                rec = dict(
                    pair_recall=float(hit.mean()) if T["n"] else None,
                    doc_deletion_recall=float((rem & T["rem"]).sum() / T["rem"].sum())
                    if T["rem"].sum() else None,
                    n_exact_removed=int(T["rem"].sum()),
                    deleted=int(rem.sum()),
                    residual_pairs=int((~rem[T["ti"]] & ~rem[T["tj"]]).sum()),
                    tokens_kept=int(nw[~rem].sum()),
                    uniq_kept=int(uniq[~rem].sum()),
                    cluster_weighted=cluster_weighted(T["edge_comp"], hit),
                    near_pairs=int(T["near"].sum()),
                    near_hits=int(hit[T["near"]].sum()),
                )
                # size-bin edge hits
                bins = []
                for k in range(len(SIZE_BINS)):
                    m = T["edge_bin"] == k
                    bins.append([int(m.sum()), int(hit[m].sum())])
                rec["size_bins"] = bins
                # top components
                tt = T["edge_top"]
                mm = tt >= 0
                topt = np.bincount(tt[mm], minlength=TOPK_COMP)
                toph = np.bincount(tt[mm], weights=hit[mm], minlength=TOPK_COMP)
                rec["top_comp_edges"] = topt.astype(int).tolist()
                rec["top_comp_hits"] = toph.astype(int).tolist()
                dd = T["edge_dom"]
                mm = dd >= 0
                domn = np.bincount(dd[mm], minlength=TOPK_DOM)
                domh = np.bincount(dd[mm], weights=hit[mm], minlength=TOPK_DOM)
                rec["dom_edges"] = domn.astype(int).tolist()
                rec["dom_hits"] = domh.astype(int).tolist()
                rec["n_eff"] = T["n_eff"]
                rec["size_weighted_component"] = T["size_weighted_component"]
                rec["n_comp"] = T["n_comp"]
                cres[str(tau)] = rec
            row["configs"][cfg] = cres
            if (b, r) == (14, 8):
                for tau in (0.7, 0.8):
                    np.savez_compressed(os.path.join(WRK, f"{corpus}_var_s{seed}_t{tau}.npz"),
                                        rem=np.packbits(rem), n=n)
                # per-component edge counts and hits for the design-effect analysis
                for tau in TAUS:
                    T = truth[tau]
                    h = L.truth_hits(sig, T["ti"], T["tj"], b, r).astype(np.float64)
                    sums = np.bincount(T["edge_comp"], weights=h, minlength=T["n_comp"])
                    np.savez_compressed(os.path.join(WRK, f"{corpus}_comp_s{seed}_t{tau}.npz"),
                                        ecount=T["ecount"].astype(np.int32),
                                        hits=sums.astype(np.int32))
            del codes, ci, cj, root_raw, rem
        append(row)
        log(corpus, "seed", seed, "%.1fs" % (time.time() - t0))


def cluster_weighted(edge_comp, hit):
    if len(edge_comp) == 0:
        return None
    order = np.argsort(edge_comp, kind="stable")
    c = edge_comp[order]
    h = hit[order]
    brk = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    sums = np.add.reduceat(h, brk)
    cnts = np.diff(np.r_[brk, len(c)])
    return float((sums / cnts).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--corpus", default="c4")
    ap.add_argument("--seeds", default="1-100")
    ap.add_argument("--threads", type=int, default=15)
    a = ap.parse_args()
    if a.mode == "run":
        if "-" in a.seeds:
            x, y = a.seeds.split("-")
            seeds = list(range(int(x), int(y) + 1))
        else:
            seeds = [int(s) for s in a.seeds.split(",")]
        run(a.corpus, seeds, a.threads)
    else:
        analyze()


def analyze():
    import collections
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        for cfg, cres in r["configs"].items():
            for tau, m in cres.items():
                by[(r["corpus"], cfg, tau)].append((r["seed"], m))
    out = {}
    for (corpus, cfg, tau), lst in sorted(by.items()):
        lst.sort()
        def series(key):
            return np.array([m[key] for _, m in lst], dtype=np.float64)
        pr = series("pair_recall")
        dr = series("doc_deletion_recall")
        cw = series("cluster_weighted")
        p = float(np.mean(pr))
        n_pairs = float(np.mean([m["residual_pairs"] + 0 for _, m in lst]))  # unused
        n_truth = int(round(np.mean([sum(b[0] for b in m["size_bins"]) for _, m in lst])))
        binom = float(np.sqrt(p * (1 - p) / n_truth)) if n_truth else None
        n_eff = float(np.mean([m["n_eff"] for _, m in lst]))
        # design-effect prediction from exact components: with p_c the per-component edge
        # hit probability, var(recall) = sum_c n_c^2 p_c(1-p_c) / (sum_c n_c)^2.
        # p_c is estimated as the across-seed mean hit rate of that component (the
        # within-component correlation is what the binomial null ignores).
        pred_std = None
        if cfg == "14x8":
            files = sorted(glob.glob(os.path.join(WRK, f"{corpus}_comp_s*_t{tau}.npz")))
            if files:
                H = np.array([np.load(f)["hits"] for f in files], np.float64)
                E = np.load(files[0])["ecount"].astype(np.float64)
                pc = H.mean(axis=0) / np.maximum(E, 1)
                var_c = (E ** 2 * pc * (1 - pc)).sum() / E.sum() ** 2
                pred_std = float(np.sqrt(var_c))
        rec = dict(
            n_seeds=len(lst), n_truth_pairs=n_truth,
            pair_recall=dict(mean=float(pr.mean()), std=float(pr.std(ddof=1)),
                             min=float(pr.min()), p05=float(np.quantile(pr, 0.05)),
                             p95=float(np.quantile(pr, 0.95)), max=float(pr.max())),
            doc_deletion_recall=dict(mean=float(dr.mean()), std=float(dr.std(ddof=1)),
                                     min=float(dr.min())),
            cluster_weighted=dict(mean=float(cw.mean()), std=float(cw.std(ddof=1)),
                                  min=float(cw.min())),
            binomial_se=binom,
            overdispersion=float(pr.std(ddof=1) / binom) if binom else None,
            doc_overdispersion=(float(dr.std(ddof=1) / np.sqrt(np.mean(dr) * (1 - np.mean(dr))
                                                              / np.mean([m["n_exact_removed"] for _, m in lst])))
                                if np.mean(dr) not in (0, 1) else None),
            n_eff_edges=n_eff,
            size_weighted_component=float(np.mean([m["size_weighted_component"] for _, m in lst])),
            design_effect_pred_std=pred_std,
            design_effect_ratio=float(pred_std / binom) if (pred_std and binom) else None,
            deleted_mean=float(np.mean([m["deleted"] for _, m in lst])),
        )
        # size-bin aggregated hit rates and across-seed contribution to variance
        contrib = []
        for k, (lo, hi) in enumerate(SIZE_BINS):
            e = np.array([m["size_bins"][k][0] for _, m in lst], float)
            h = np.array([m["size_bins"][k][1] for _, m in lst], float)
            rate = h / np.maximum(e, 1)
            contrib.append(dict(bin=f"{lo}-{hi}", edges=int(e.mean()), mean_rate=float(rate.mean()),
                                sd_rate=float(rate.std(ddof=1)),
                                sd_contribution_pts=float(100 * e.mean() * rate.std(ddof=1) / n_truth)
                                if n_truth else None))
        rec["size_bins"] = contrib
        # top-10 component contribution to the across-seed spread
        e = np.array([m["top_comp_edges"] for _, m in lst], float)
        h = np.array([m["top_comp_hits"] for _, m in lst], float)
        rate = np.divide(h, np.maximum(e, 1))
        sdc = rate.std(axis=0, ddof=1)
        order = np.argsort(-e.mean(axis=0))
        rec["top_components"] = [
            dict(rank=int(j), edges=int(e.mean(axis=0)[j]), mean_rate=float(rate.mean(axis=0)[j]),
                 sd_rate=float(sdc[j]),
                 sd_contribution_pts=float(100 * e.mean(axis=0)[j] * sdc[j] / n_truth))
            for j in order[:10]]
        rec["top10_pair_share"] = float(e.mean(axis=0)[order[:10]].sum() / n_truth)
        out[f"{corpus}|{cfg}|{tau}"] = rec
    with open(os.path.join(OUT, "seed_variance.json"), "w") as f:
        json.dump({"analysis": "seed variance", "rows": len(rows),
                   "provenance": "independently rerun (this run)", "cells": out}, f, indent=1)
    log("wrote results/seed_variance.json", len(out), "cells")


if __name__ == "__main__":
    main()
