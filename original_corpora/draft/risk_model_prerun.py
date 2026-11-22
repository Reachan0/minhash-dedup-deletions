#!/usr/bin/env python3
"""Document-level flip-risk model with genuinely pre-run features.

Draft 1's risk model used exact-truth features (component size, truth degree, mean edge
Jaccard, near-threshold degree), which are unavailable before the audit runs. This
script refits with pre-run features only: document length, shingle count, unique-shingle
fraction and domain frequency. Cross-validation is grouped by domain so a document is
never scored by a model that saw another document from the same site.

Two populations are reported: all documents (no truth selection at all) and documents
with a true neighbour at tau=0.8 (the at-risk set, which is only identifiable after the
exact join). The truth-feature model is refit on the same folds to quantify the inflation.

Output: results/risk_model_prerun.json
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
CORPUS = "c4"
TAU = 0.8


def load():
    n = MC.CORPORA[CORPUS]
    z = np.load(os.path.join(WRK, f"{CORPUS}_ranks.npz"))
    uniq = np.asarray(z["uniq"], np.float64)
    md = pq.read_table(os.path.join(WRK, f"{CORPUS}_meta.parquet")).to_pydict()
    nw = np.asarray(md["nwords"], np.float64)
    ns = np.asarray(md["nshingle"], np.float64)
    dom = np.array([d.lower().removeprefix("www.") if d else "" for d in md["domain"]])
    du, di = np.unique(dom, return_inverse=True)
    count = np.zeros(n, np.int64)
    for f in sorted(os.listdir(WRK)):
        if f.startswith(f"{CORPUS}_var_s") and f.endswith(f"_t{TAU}.npz"):
            q = np.load(os.path.join(WRK, f))
            count += np.unpackbits(q["rem"], bitorder="big")[:n]
    q = np.load(os.path.join(WRK, f"{CORPUS}_truth_t{TAU}.npz"))
    ti, tj = np.asarray(q["i"]), np.asarray(q["j"])
    deg = np.zeros(n, np.int64)
    np.add.at(deg, ti, 1)
    np.add.at(deg, tj, 1)
    np.add.at(deg, ti, 0)
    return dict(n=n, uniq=uniq, nw=nw, ns=ns, dom_inv=di, dom_docs=np.bincount(di),
                count=count, deg=deg, ti=ti, tj=tj, q=q)


def cv_scores(X, y, groups, name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = GroupKFold(n_splits=5)
    out = {}
    for nm, model in [("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                      ("gbm", HistGradientBoostingClassifier(max_iter=200, random_state=42))]:
        p = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        out[nm] = dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)))
        out["prob_" + nm] = p
    out["n"] = int(len(y))
    out["positives"] = int(y.sum())
    out["base_rate"] = float(y.mean())
    return out


def deciles(p, y):
    qs = np.quantile(p, np.linspace(0, 1, 11))
    rows = []
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        m = (p >= lo) & (p <= hi) if i == 9 else (p >= lo) & (p < hi)
        rows.append(dict(decile=i + 1, n=int(m.sum()), flip_rate=float(y[m].mean()) if m.any() else None))
    return rows


def main():
    D = load()
    n = D["n"]
    Xpre = np.column_stack([
        np.log1p(D["nw"]), np.log1p(D["ns"]),
        D["uniq"] / np.maximum(1, D["ns"]),
        np.log1p(D["dom_docs"][D["dom_inv"]]),
    ])
    pre_names = ["log_nwords", "log_nshingle", "unique_frac", "log_domain_docs"]
    # truth features used in Draft 1 (for the inflation comparison only)
    root = np.asarray(D["q"]["root"])
    comp_size = np.bincount(root)[root]
    mean_j = np.zeros(n)
    jac = np.asarray(D["q"]["jac"], np.float64)
    np.add.at(mean_j, D["ti"], jac)
    np.add.at(mean_j, D["tj"], jac)
    mean_j = np.divide(mean_j, D["deg"], out=np.zeros(n), where=D["deg"] > 0)
    Xfull = np.column_stack([Xpre, np.log1p(comp_size), np.log1p(D["deg"]), mean_j])
    full_names = pre_names + ["log_comp_size", "log_truth_degree", "mean_edge_j"]

    flips = (D["count"] > 0) & (D["count"] < 100)
    res = dict(corpus=CORPUS, tau=TAU, n_seeds=100, provenance="independently rerun (this run)",
               features_pre=pre_names, features_truth=full_names,
               cv="GroupKFold(5) grouped by domain", populations={})
    for pop_name, mask in [("all_docs", np.ones(n, bool)), ("at_risk", D["deg"] > 0)]:
        y = flips[mask].astype(int)
        g = D["dom_inv"][mask]
        pre = cv_scores(Xpre[mask], y, g, "pre")
        full = cv_scores(Xfull[mask], y, g, "full")
        res["populations"][pop_name] = dict(
            n=int(mask.sum()), positives=int(y.sum()), base_rate=float(y.mean()),
            pre_run_features={k: pre[k] for k in ("logistic", "gbm", "n", "positives", "base_rate")},
            truth_features={k: full[k] for k in ("logistic", "gbm", "n", "positives", "base_rate")},
            deciles_pre=deciles(pre["prob_gbm"], y),
            deciles_truth=deciles(full["prob_gbm"], y),
        )
        print(pop_name, "pre", res["populations"][pop_name]["pre_run_features"],
              "truth", res["populations"][pop_name]["truth_features"], flush=True)
    with open(os.path.join(OUT, "risk_model_prerun.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
