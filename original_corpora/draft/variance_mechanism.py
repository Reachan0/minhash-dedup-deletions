#!/usr/bin/env python3
"""Mechanism behind the seed-to-seed variance, on the C4 cell (tau=0.8, 14x8).

Reads the per-seed deleted masks and per-component edge-hit counts written by
seed_variance.py and produces results/variance_mechanism.json:

  flips        documents whose keep/drop status changes when only the hash seed changes,
               with their token and unique-content weight, and pairwise disagreement
  components   per-component contribution to the variance of pair recall; the largest
               clusters, their domains and their across-seed rates
  domains      domain-level contribution (template pages), top domains by variance
  shared       documents whose exact duplicates span more than one domain (syndicated
               content) and their share of the pair mass
  risk         document-level predictability of a flip from pre-run features (AUC, CV)
  regime       corpus-level features that predict overdispersion before running

Usage: python3 variance_mechanism.py [--corpus c4] [--tau 0.8]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03"))
import lsh as L  # noqa: E402
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")


def load_rem(corpus, tau, k_max):
    files = []
    for s in range(1, k_max + 1):
        p = os.path.join(WRK, f"{corpus}_var_s{s}_t{tau}.npz")
        if os.path.exists(p):
            files.append((s, p))
    masks = []
    for s, p in files:
        z = np.load(p)
        n = int(z["n"])
        masks.append(np.unpackbits(np.asarray(z["rem"]))[:n].astype(bool))
    return [s for s, _ in files], np.array(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="c4")
    ap.add_argument("--tau", default="0.8")
    a = ap.parse_args()
    corpus, tau = a.corpus, a.tau
    n = MC.CORPORA[corpus]
    meta = pq.read_table(os.path.join(WRK, f"{corpus}_meta.parquet")).to_pydict()
    nw = np.asarray(meta["nwords"], np.int64)
    uniq = np.asarray(meta["n_unique"], np.int64)
    dom = np.array(meta["domain"])
    ids = meta["id"]

    seeds, M = load_rem(corpus, tau, 200)
    K = len(seeds)
    count = M.sum(axis=0)
    at_risk = count > 0
    truth = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
    ti, tj, jac = np.asarray(truth["i"]), np.asarray(truth["j"]), np.asarray(truth["jac"])
    root, rem_ex = np.asarray(truth["root"]), np.asarray(truth["removed"])
    comp_doc = np.bincount(root, minlength=n)

    flips = (count > 0) & (count < K)
    always = count == K
    deg0 = np.bincount(np.r_[ti, tj], minlength=n)
    truth_at_risk = deg0 > 0
    out = {"corpus": corpus, "tau": float(tau), "n_seeds": K, "n": int(n),
           "provenance": "independently rerun (this run)"}
    out["flips"] = dict(
        n_always_kept_at_risk=int((truth_at_risk & (count == 0)).sum()),
        n_always_deleted=int(always.sum()), n_flip=int(flips.sum()),
        flip_share_of_at_risk=float(flips[truth_at_risk].mean()) if truth_at_risk.any() else None,
        flip_tokens=int(nw[flips].sum()), flip_unique_items=int(uniq[flips].sum()),
        flip_tokens_share=float(nw[flips].sum() / nw.sum()) if nw.sum() else None,
        all_deleted_tokens=int(nw[always].sum()),
        exact_removed=int(rem_ex.sum()), at_risk_docs=int(truth_at_risk.sum()),
        mean_deleted_per_seed=float(M.sum(axis=1).mean()),
    )
    # pairwise disagreement across the first 10 seeds
    kk = min(10, K)
    dis, uni = [], []
    for i in range(kk):
        for j in range(i + 1, kk):
            dis.append(int((M[i] ^ M[j]).sum()))
            uni.append(int((M[i] | M[j]).sum()))
    out["flips"]["pairwise_disagreement_mean"] = float(np.mean(dis)) if dis else None
    out["flips"]["pairwise_disagreement_share"] = float(np.mean(np.array(dis) / np.maximum(uni, 1))) if dis else None
    out["flips"]["pairwise_disagreement_max"] = int(np.max(dis)) if dis else None

    # ---- component-level variance decomposition ----
    comp_files = sorted(glob.glob(os.path.join(WRK, f"{corpus}_comp_s*_t{tau}.npz")))
    E = np.load(comp_files[0])["ecount"].astype(np.float64)
    H = np.array([np.load(f)["hits"] for f in comp_files], np.float64)
    pc = H.mean(axis=0) / np.maximum(E, 1)
    tot_e = E.sum()
    var_c = E ** 2 * pc * (1 - pc) / tot_e ** 2
    sd_parts = np.sqrt(var_c)
    order = np.argsort(-var_c)
    # map edge-component id -> docs
    ecomp_u, ecomp_idx = np.unique(root[ti], return_inverse=True)
    # modal domain and size for each edge component
    # component representative doc list via root
    comp_members = {}
    for d in np.flatnonzero(comp_doc[root] >= 2):
        r = int(root[d])
        comp_members.setdefault(r, []).append(d)
    comp_root_of_ec = [int(ecomp_u[k]) for k in range(len(ecomp_u))]
    top = []
    for k in order[:15]:
        r = comp_root_of_ec[k]
        mem = comp_members.get(r, [])
        dcount = {}
        for d in mem:
            dd = dom[d]
            dcount[dd] = dcount.get(dd, 0) + 1
        top_dom = max(dcount.items(), key=lambda x: x[1])[0] if dcount else ""
        top.append(dict(rank=int(k), docs=int(len(mem)), edges=int(E[k]),
                        mean_hit_rate=float(pc[k]),
                        sd_hit_rate=float(np.sqrt(max(pc[k] * (1 - pc[k]), 0))),
                        across_seed_rate_sd=float(H[:, k].std(ddof=1) / max(E[k], 1)),
                        var_pts=float(var_c[k] * 100 ** 2),
                        sd_pts=float(sd_parts[k] * 100),
                        modal_domain=top_dom,
                        example_id=str(ids[mem[0]]) if mem else ""))
    out["components"] = dict(n_components=int(len(E)),
                             top10_pair_share=float(E[order[:10]].sum() / tot_e),
                             implied_sd_pts=float(np.sqrt(var_c.sum()) * 100),
                             top=top)
    # ---- domain-level variance contribution ----
    dcomp = {}
    dom_u, dom_inv = np.unique(dom, return_inverse=True)
    for k in range(len(E)):
        r = comp_root_of_ec[k]
        mem = comp_members.get(r, [])
        if not mem:
            continue
        vals, cnts = np.unique(dom_inv[mem], return_counts=True)
        dd = int(vals[np.argmax(cnts)])
        dcomp[dd] = dcomp.get(dd, 0) + var_c[k]
    dorder = sorted(dcomp.items(), key=lambda x: -x[1])[:15]
    dom_total = {}
    for d in np.flatnonzero(comp_doc[root] >= 2):
        dom_total[dom[d]] = dom_total.get(dom[d], 0) + 1
    out["domains"] = dict(top=[dict(domain=str(dom_u[dd]), var_pts=float(v * 100 ** 2),
                                    sd_pts=float(np.sqrt(v) * 100),
                                    docs_in_components=int(dom_total.get(str(dom_u[dd]), 0)))
                              for dd, v in dorder],
                          n_domains_with_duplicates=len(dcomp))
    # ---- shared documents (duplicates span >1 domain) ----
    deg = deg0
    # neighbour domain diversity: sweep pairs, accumulate min/max domain id per doc
    nd_lo = np.full(n, 10 ** 9, np.int64)
    nd_hi = np.full(n, -1, np.int64)
    di = dom_inv[ti]
    dj = dom_inv[tj]
    np.minimum.at(nd_lo, ti, dj)
    np.maximum.at(nd_hi, ti, dj)
    np.minimum.at(nd_lo, tj, di)
    np.maximum.at(nd_hi, tj, di)
    shared = (deg > 0) & (nd_hi > nd_lo)
    shared_edges = shared[ti] | shared[tj]
    # variance contribution of components containing shared docs
    shared_roots = set(int(root[d]) for d in np.flatnonzero(shared))
    var_shared = sum(var_c[k] for k in range(len(E)) if comp_root_of_ec[k] in shared_roots)
    out["shared"] = dict(n_docs=int(shared.sum()),
                         token_share=float(nw[shared].sum() / max(1, nw.sum())),
                         pair_mass_share=float(shared_edges.mean()),
                         var_pts=float(var_shared * 100 ** 2),
                         var_share=float(var_shared / var_c.sum()) if var_c.sum() else None)

    # ---- document-level risk model ----
    border = np.zeros(n, np.int64)
    mean_j = np.zeros(n, np.float64)
    for t in (0.6, 0.7):
        q = np.load(os.path.join(WRK, f"{corpus}_truth_t{t}.npz"))
        bi, bj, bj_jac = np.asarray(q["i"]), np.asarray(q["j"]), np.asarray(q["jac"])
        m = (bj_jac >= 0.6) & (bj_jac < 0.8)
        border += np.bincount(np.r_[bi[m], bj[m]], minlength=n)
    np.add.at(mean_j, ti, jac)
    np.add.at(mean_j, tj, jac)
    mean_j = np.divide(mean_j, deg, out=np.zeros(n), where=deg > 0)
    dom_docs = np.bincount(dom_inv, minlength=len(dom_u))
    X = np.column_stack([
        np.log1p(comp_doc[root]), np.log1p(deg), np.log1p(border), np.log1p(nw),
        np.log1p(np.asarray(meta["nshingle"], np.int64)), uniq / np.maximum(1, np.asarray(meta["nshingle"], np.int64)),
        np.log1p(dom_docs[dom_inv]), mean_j,
    ])
    names = ["log_comp_size", "log_truth_degree", "log_border_degree", "log_nwords",
             "log_nshingle", "unique_frac", "log_domain_docs", "mean_edge_j"]
    mask = deg > 0
    y = flips[mask].astype(int)
    Xm = X[mask]
    risk = dict(n_at_risk=int(mask.sum()), n_positive=int(y.sum()), features=names)
    if y.sum() >= 20 and y.sum() < len(y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        for nm, model in [("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                          ("gbm", HistGradientBoostingClassifier(max_iter=200, random_state=42))]:
            prob = cross_val_predict(model, Xm, y, cv=cv, method="predict_proba")[:, 1]
            risk[nm] = dict(auc=float(roc_auc_score(y, prob)),
                            ap=float(average_precision_score(y, prob)))
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xm, y)
        coef = model[-1].coef_[0]
        risk["logistic_coef"] = {n_: float(c) for n_, c in zip(names, coef)}
        # risk deciles using the GBM probabilities
        prob = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=42),
                                 Xm, y, cv=cv, method="predict_proba")[:, 1]
        qs = np.quantile(prob, np.linspace(0, 1, 11))
        dec = []
        for i in range(10):
            lo, hi = qs[i], qs[i + 1]
            m = (prob >= lo) & (prob <= hi) if i == 9 else (prob >= lo) & (prob < hi)
            dec.append(dict(decile=i + 1, n=int(m.sum()), flip_rate=float(y[m].mean()) if m.any() else None))
        risk["deciles"] = dec
    out["risk"] = risk

    # ---- pre-run regime features ----
    z = np.load(os.path.join(WRK, f"{corpus}_ranks.npz"))
    out["regime"] = dict(
        n_truth_pairs=int(len(ti)), n_components=int(len(E)),
        max_component=int(comp_doc.max()),
        size_weighted_component=float((comp_doc[comp_doc >= 2].astype(np.float64) ** 2).sum()
                                      / comp_doc[comp_doc >= 2].sum()),
        pair_weighted_mean_j=float(jac.mean()),
        near_threshold_share=float(((jac >= float(tau)) & (jac < float(tau) + 0.05)).mean()),
        top10_pair_share=float(E[order[:10]].sum() / tot_e),
        implied_sd_pts=float(np.sqrt(var_c.sum()) * 100),
    )
    with open(os.path.join(OUT, f"variance_mechanism_{corpus}_t{tau}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("flips", "regime")}, indent=1)[:2500])
    print("components:", out["components"]["implied_sd_pts"], "top10 share", out["components"]["top10_pair_share"])
    print("shared:", out["shared"])
    print("risk:", {k: out["risk"].get(k) for k in ("logistic", "gbm")})


if __name__ == "__main__":
    main()
