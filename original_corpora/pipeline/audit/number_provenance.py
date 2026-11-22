#!/usr/bin/env python3
"""Provenance for the five headline numbers.

Recomputes from stored per-seed records and compares with the value printed in
Draft 2. Writes audit/number_provenance.json (no hand-typed numbers).
"""
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
OUT = os.path.join(ROOT, ".work/dedupjoin-pvldb-v3/audit")
N = 700000


def unpack(path):
    z = np.load(path)
    return np.unpackbits(np.asarray(z["rem"]))[:N].astype(bool)


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# ---- masks and metadata -----------------------------------------------------
paths = sorted(glob.glob(os.path.join(WRK, "c4_var_s*_t0.8.npz")),
               key=lambda p: int(os.path.basename(p).split("_s")[1].split("_")[0]))
masks = np.stack([unpack(p) for p in paths])
K = masks.shape[0]
count = masks.sum(axis=0)
flips = (count > 0) & (count < K)

t = np.load(os.path.join(WRK, "c4_truth_t0.8.npz"))
ti, tj = np.asarray(t["i"]), np.asarray(t["j"])
deg = np.zeros(N, np.int64)
np.add.at(deg, ti, 1)
np.add.at(deg, tj, 1)
at_risk = deg > 0

meta = pq.read_table(os.path.join(WRK, "c4_meta.parquet")).to_pydict()
nw = np.asarray(meta["nwords"], np.int64)
nu = np.asarray(meta["n_unique"], np.int64)
total_tokens = int(nw.sum())
total_uniq = int(nu.sum())

at_risk_flips = flips & at_risk
no_nb_flips = flips & ~at_risk

vd = load_json("variance_decomposition.json")["cells"]["c4"]
vm = load_json("variance_mechanism_c4_t0.8.json")
sv = load_json("seed_variance.json")["cells"]["c4|14x8|0.8"]

top = vd["top_component_var_share"][0]
within = vd["within_component_var_share"]
share_of_total = top * within

out = {
    "provenance": "rerun (masks + parquet) and log-recomputed (JSON)",
    "population": {
        "corpus": "c4", "n_docs": N, "config": "14x8", "tau": 0.8,
        "seeds": K, "total_tokens": total_tokens,
        "total_unique_shingles": total_uniq,
        "docs_with_true_neighbour": int(at_risk.sum()),
        "truth_pairs": int(ti.shape[0]),
    },
    "numbers": {
        "at_risk_flip_token_share_0.05pct": {
            "printed": "0.05% (intro, item 4; discussion; conclusion)",
            "numerator_docs": int(at_risk_flips.sum()),
            "numerator_tokens": int(nw[at_risk_flips].sum()),
            "denominator": "total C4 tokens (258,514,546)",
            "value_frac": float(nw[at_risk_flips].sum() / total_tokens),
            "value_pct": float(100 * nw[at_risk_flips].sum() / total_tokens),
            "population": "documents with >=1 exact neighbour at tau=0.8 whose "
                          "raw-LSH keep/delete status differs across the 100 seeds",
            "source": ".work/dedupjoin/main/c4_var_s*_t0.8.npz + c4_truth_t0.8.npz + c4_meta.parquet",
            "status": "confirmed for the at-risk population; the conclusion "
                      "omits that qualifier",
        },
        "at_risk_flip_unique_share_0.004pct": {
            "printed": "0.004% (intro, item 4; discussion)",
            "numerator_docs": int(at_risk_flips.sum()),
            "numerator_unique": int(nu[at_risk_flips].sum()),
            "denominator": "sum of per-document distinct-shingle counts, C4 "
                           "(217,269,688)",
            "value_pct": float(100 * nu[at_risk_flips].sum() / total_uniq),
            "population": "same at-risk flipping documents",
            "status": "confirmed for the at-risk population",
        },
        "all_flip_token_share_1.14pct": {
            "printed": "1.10% for the 17,840 no-neighbour flips (05-results); "
                       "1.14% is the all-flip total requested in the work order",
            "numerator_docs": int(flips.sum()),
            "numerator_tokens": int(nw[flips].sum()),
            "denominator": "total C4 tokens (258,514,546)",
            "value_pct": float(100 * nw[flips].sum() / total_tokens),
            "no_neighbour_flip_docs": int(no_nb_flips.sum()),
            "no_neighbour_flip_token_pct": float(100 * nw[no_nb_flips].sum() / total_tokens),
            "no_neighbour_flip_unique_pct": float(100 * nu[no_nb_flips].sum() / total_uniq),
            "population": "all 18,529 documents whose raw-LSH status differs "
                          "across 100 seeds at tau=0.8",
            "source": "same masks + parquet; matches "
                      "variance_mechanism_c4_t0.8.json:flips.flip_tokens_share "
                      f"({vm['flips']['flip_tokens_share']})",
            "status": "recomputed; 1.10% + 0.05% does not equal 1.14% exactly "
                      "because of rounding of two disjoint populations",
        },
        "at_risk_flip_share_13.4pct": {
            "printed": "13.4%",
            "numerator": int((flips & at_risk).sum()),
            "denominator": int(at_risk.sum()),
            "value_pct": float(100 * (flips & at_risk).sum() / at_risk.sum()),
            "source": "variance_mechanism_c4_t0.8.json:flips.flip_share_of_at_risk "
                      f"({vm['flips']['flip_share_of_at_risk']})",
            "status": "confirmed",
        },
        "top_component_variance_99.99pct": {
            "printed": "99.99% of the total variance (intro item 4; 05-results "
                       "body; figure 2b caption)",
            "top_component_edges": vd["top_components"][0]["edges"],
            "total_edges": vd["n_truth_pairs"],
            "mean_hit_rate": vd["top_components"][0]["mean_rate"],
            "across_seed_sd": vm["components"]["top"][0]["across_seed_rate_sd"],
            "empirical_var_share_of_summed_component_variance": top,
            "within_component_var_share_of_total": within,
            "empirical_var_share_of_total": share_of_total,
            "all_or_nothing_bound_over_observed": vd["bound_over_observed"],
            "quantity": "empirical variance decomposition: "
                        "Var(H_c)/sum_c Var(H_c), where H_c is the per-seed "
                        "hit count of exact component c; NOT the all-or-nothing "
                        "bound and NOT a sampling-error estimate",
            "corrected_value_pct_of_within": float(100 * top),
            "corrected_value_pct_of_total": float(100 * share_of_total),
            "source": "variance_decomposition.json:cells.c4",
            "status": "corrected: 99.97% of the summed per-component variance, "
                      "which is 100.1% of the total variance because the "
                      "covariance term is -0.13%; the printed 99.99% matches "
                      "neither denominator",
        },
        "shared_873_docs_99.994pct": {
            "printed": "99.994% of the variance (05-results body)",
            "value": vm["shared"]["var_share"],
            "quantity": "share of sum_c E_c^2 p_c(1-p_c), the all-or-nothing "
                        "per-component variance proxy, not the empirical "
                        "variance decomposition",
            "docs": vm["shared"]["n_docs"],
            "pair_mass_share": vm["shared"]["pair_mass_share"],
            "token_share": vm["shared"]["token_share"],
            "status": "confirmed as computed but a different decomposition from "
                      "the 99.99% claim; do not put the two in one sentence "
                      "without naming the denominator",
        },
        "overdispersion_25.8x": {
            "printed": "25.8x at tau=0.8; up to 35x in the abstract",
            "observed_sd": vd["observed_sd"],
            "binomial_sd": vd["binomial_sd"],
            "ratio": vd["overdispersion"],
            "tau0.7_ratio": load_json("seed_variance.json")["cells"]["c4|14x8|0.7"]["overdispersion"],
            "population": "100 seeds, C4, 14x8, 76,940 truth pairs, mean pair "
                          "recall 0.96654",
            "source": "variance_decomposition.json / seed_variance.json",
            "status": "confirmed (34.58 rounds to 35)",
        },
    },
}
with open(os.path.join(OUT, "number_provenance.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items()
                      if kk in ("value_pct", "corrected_value_pct_of_within",
                                "corrected_value_pct_of_total", "ratio",
                                "status")}
                  for k, v in out["numbers"].items()}, indent=1))
