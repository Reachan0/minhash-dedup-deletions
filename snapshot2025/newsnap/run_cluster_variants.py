#!/usr/bin/env python3
"""Cluster-equal-weight / drop-largest / pair-vs-doc cells with the SAME code.

Cells:
  pool workloads c4(700k), fw(1M), mix(1M), wet(175k) on their existing frozen
  caches (core/cache signatures), and size-matched subsamples c4sub/fwsub of
  N = 531,597 documents (the snapshot size) to remove scale as a confounder.
  For a subsample the truth graph is the full-workload truth restricted to pairs
  with both endpoints < N (exactly the truth of the subset) and path A/B are
  recomputed on the truncated document set with the same band keys.

Rows append to newsnap/analysis/cells.jsonl; log-recomputed (existing per-seed
signatures + truth) with the snapshot cells recomputed here.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402
K = S.K

OUT_DIR = os.path.join(S.SNAP, "analysis")
CELLS = os.path.join(OUT_DIR, "cells.jsonl")
SNAP_N = 531597

# workload -> (taus, seeds) matching the core campaign
PLAN = {
    "c4": ([0.8, 0.75], {0.8: [1, 2, 3, 4, 5], 0.75: [1, 2, 3]}),
    "fw": ([0.8, 0.75], {0.8: [1, 2, 3, 4, 5], 0.75: [1, 2, 3]}),
    "mix": ([0.8, 0.75], {0.8: [1, 2, 3], 0.75: [1, 2, 3]}),
    "wet": ([0.8, 0.75], {0.8: [1, 2, 3], 0.75: [1, 2, 3]}),
    "c4sub": ([0.8, 0.75], {0.8: [1, 2, 3], 0.75: [1, 2, 3]}),
    "fwsub": ([0.8, 0.75], {0.8: [1, 2, 3], 0.75: [1, 2, 3]}),
}
SRC = {"c4sub": "c4", "fwsub": "fw"}


def cell_stats(n, root, rem, has_nb, rng):
    size = np.bincount(root, minlength=n)
    comp = size >= 2
    del_c = np.where(comp, size - 1, 0).astype(np.float64)
    uns = rem & ~has_nb
    uns_c = np.bincount(root[uns], minlength=n).astype(np.float64)
    shares = uns_c[comp] / del_c[comp]
    largest = int(np.argmax(size))
    keep = np.ones(n, bool)
    keep[root == largest] = False
    dl = uns[keep].sum() / max((rem & keep).sum(), 1)
    # template density: share of docs that sit in an LSH component of >= 10 docs
    big = size >= 10
    tpl = float(np.isin(root, np.flatnonzero(big)).sum() / max(int((size[root] > 1).sum()), 1))
    bs = []
    if len(shares):
        idx = rng.integers(0, len(shares), size=(1000, len(shares)))
        bs = shares[idx].mean(axis=1)
    return dict(
        n_components=int(comp.sum()),
        largest_component_docs=int(size[largest]),
        largest_component_share_deleted=float(del_c[largest] / max(del_c.sum(), 1)),
        cluster_equal_weight_unsupported_share=float(shares.mean()) if len(shares) else None,
        cluster_equal_weight_ci95=([float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
                                   if len(bs) else None),
        drop_largest_unsupported_share_del=float(dl),
        unsupported_share_del_docweighted=float(uns.sum() / max(rem.sum(), 1)),
        component_size_p50=float(np.median(size[comp])) if comp.any() else None,
        component_size_p90=float(np.quantile(size[comp], 0.9)) if comp.any() else None,
        component_size_max=int(size[largest]),
        template_density=float(tpl),
    )


def process(name):
    src = SRC.get(name, name)
    off, flat, nw = K.load_shingles(src)
    n_full = len(off) - 1
    n = SNAP_N if name.endswith("sub") else n_full
    assert n <= n_full
    off = off[:n + 1]
    flat = flat[:off[n]] if n < n_full else flat
    valid = (off[1:] - off[:-1]) > 0 if n < n_full else (np.diff(off) > 0)
    taus, seeds = PLAN[name]
    done = S.done_keys(CELLS, "key")
    # truth: full graph restricted to the subset (exact for a subset)
    truth = {}
    for tau in taus:
        ti, tj, jac = K.compute_truth(src, tau, off)
        if n < n_full:
            m = (ti < n) & (tj < n)
            ti, tj, jac = ti[m], tj[m], jac[m]
        truth[tau] = (ti, tj, K.has_true_neighbour(n, ti, tj))
        print(f"[{name}] truth t{tau}: {len(ti)} edges", flush=True)
    rng = np.random.default_rng(20261119)
    for seed in sorted({s for tau in taus for s in seeds[tau]}):
        taus_s = [t for t in taus if seed in seeds[t]]
        if all(f"{name}|t{t}|s{seed}" in done for t in taus_s):
            continue
        sig = np.load(K.sig_path(src, seed), mmap_mode="r")
        Kb = K.band_keys(np.asarray(sig))
        del sig
        ei, ej, bstats = K.chain_edges(Kb, valid)
        root = K.uf_minid(n, ei, ej)
        rem = (root != np.arange(n)) & valid
        pi, pj, pstats = K.all_bucket_pairs(Kb, valid, n)
        for tau in taus_s:
            key = f"{name}|t{tau}|s{seed}"
            if key in done:
                continue
            ti, tj, hnb = truth[tau]
            ver = K.L.jaccard_pairs(off, flat, pi, pj) >= tau - 1e-12
            vei, vej = pi[ver], pj[ver]
            root_b = K.uf_minid(n, vei, vej)
            rem_b = (root_b != np.arange(n)) & valid
            ma = K.method_metrics(n, rem, ti, tj, hnb)
            mb = K.method_metrics(n, rem_b, ti, tj, hnb)
            ea = K.edge_stats(ti, tj, ei, ej, n)
            eb = K.edge_stats(ti, tj, pi, pj, n)          # candidate recall
            ev = K.edge_stats(ti, tj, vei, vej, n)        # verified-edge recall
            ex_root = K.uf_minid(n, ti, tj)
            ex_rem = (ex_root != np.arange(n))
            row = dict(key=key, evidence_tier="log-recomputed", name=name, source=src, n=n,
                       tau=tau, seed=seed, truth_edges=int(len(ti)),
                       largest_bucket=int(bstats["largest_bucket"]),
                       bucket_pairs_sum=int(bstats["bucket_pairs_sum"]),
                       candidate_pairs=int(len(pi)),
                       pair_recall_A_chain=ea.get("pair_recall"),
                       pair_recall_B_candidates=eb.get("pair_recall"),
                       pair_recall_B_verified=ev.get("pair_recall"),
                       doc_recall_A=float((ex_rem & rem).sum() / max(ex_rem.sum(), 1)),
                       doc_recall_B=float((ex_rem & rem_b).sum() / max(ex_rem.sum(), 1)),
                       exact_deleted=int(ex_rem.sum()),
                       deleted_A=ma["deleted"], deleted_B=mb["deleted"],
                       unsupported_A=ma["unsupported"],
                       unsupported_share_del_A=ma["unsupported_share_del"],
                       unsupported_share_input_A=ma["unsupported_share_input"],
                       residual_A=ma["residual_duplicate_pairs"],
                       residual_B=mb["residual_duplicate_pairs"],
                       **cell_stats(n, root, rem, hnb, rng))
            S.append_row(CELLS, row)
            print(key, "pair_recallA", round(ea.get("pair_recall") or 0, 3),
                  "pair_recallBcand", round(eb.get("pair_recall") or 0, 3),
                  "docA", round(row["doc_recall_A"], 3), "docB", round(row["doc_recall_B"], 3),
                  "ew", round(row["cluster_equal_weight_unsupported_share"] or 0, 3), flush=True)
        del Kb, ei, ej, root, rem, pi, pj
    del off, flat


if __name__ == "__main__":
    names = sys.argv[1:] or list(PLAN)
    os.makedirs(OUT_DIR, exist_ok=True)
    for nm in names:
        process(nm)
