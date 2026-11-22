#!/usr/bin/env python3
"""Aggregate the scale ladder into results/scale_ladder_summary.json.

Consumes results/scale_ladder_rows.jsonl (one row per workload, written by
scale_ladder.py) and results/main_compare_rows.jsonl (the earlier five-seed run at
700k C4 / 1M FineWeb) for a same-input cross-check at the 1M rung.

Definitions used everywhere downstream (paper text copies them from the summary):
  lsh_raw_s      signature + banding + candidate generation + bucket expansion + raw clustering
  lsh_verify_s   lsh_raw_s + exact verification of candidate edges + verified clustering (tau=0.8)
  exact_s        global rank construction + exact index + exact probe + exact clustering
  io_s           shard-cache read and composition (shared input, reported separately)
Every stage is scored at the same 16 threads and tau in {0.6,0.7,0.8,0.9}.
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
ROWS = os.path.join(RES, "scale_ladder_rows.jsonl")
MAIN = os.path.join(RES, "main_compare_rows.jsonl")
OUT = os.path.join(RES, "scale_ladder_summary.json")
TAUS = ["0.6", "0.7", "0.8", "0.9"]


def load_rows():
    out = []
    with open(ROWS) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stage_sum(stages, names):
    return round(sum(stages.get(n, 0.0) for n in names), 3)


def method_costs(r):
    st = r["stages"]
    lsh = stage_sum(st, ["signature", "banding", "cand_sort", "bucket_expand",
                         "cand_dedup", "cluster_raw"])
    ver = stage_sum(st, ["verify", "cluster_verify_t0.8"])
    ex = stage_sum(st, ["rank_build", "exact_index_t0.8", "exact_probe_t0.8",
                        "cluster_exact_t0.8"])
    wit = stage_sum(st, ["witness_t0.8"])
    return {"io_s": st.get("io_shards", 0.0), "lsh_raw_s": lsh,
            "lsh_verify_s": round(lsh + ver, 3), "witness_add_s": wit,
            "exact_s": ex, "rank_s": st.get("rank_build", 0.0)}


def peak_of(r, names):
    p = [r["peak_rss_mb"].get(n) or 0 for n in names]
    return max(p) if p else None


def crosscheck(rows):
    """The 1M prefix of the pool is exactly the 1M FineWeb cell of the earlier run."""
    r1 = next((r for r in rows if r["workload_id"] == "prefix:1000000"), None)
    if r1 is None:
        return {"status": "missing"}
    ref = None
    with open(MAIN) as f:
        for line in f:
            d = json.loads(line)
            if d["corpus"] == "fw" and d["seed"] == 1:
                ref = d
                break
    if ref is None:
        return {"status": "reference missing"}
    m = ref["per_tau"]["0.8"]
    got = r1["per_tau"]["0.8"]
    checks = {
        "raw_removed": [m["raw_removed"], got["raw_removed"]],
        "raw_unsupported": [m["raw_wrong"], got["raw_unsupported"]],
        "exact_removed": [m["exact_removed"], got["exact_removed"]],
        "ver_removed": [m["ver_removed"], got["ver_removed"]],
        "wit_removed": [m["wit_removed"], got["wit_removed"]],
        "n_cand_pairs": [ref["n_cand_pairs"], got["n_cand_pairs"]],
        "n_truth_pairs": [m["n_truth_pairs"], got["n_truth_pairs"]],
    }
    ok = all(a == b for a, b in checks.values())
    return {"status": "identical" if ok else "different", "checks": checks,
            "reference": "main_compare_rows.jsonl corpus=fw seed=1 tau=0.8",
            "note": "same shingles (C03 protocol), same seed-1 draw, same banding"}


def extrapolate(rows):
    """Linear fit of peak RSS and exact-join time against shingle count (rungs >= 300k)."""
    # fit only the rungs where the regime is stable: below 1M the fixed per-run overhead
    # (process, block boundaries) dominates the intercept and biases a 64 GiB projection
    rs = [r for r in rows if r["kind"] == "prefix" and r["n_docs"] >= 1000000]
    if len(rs) < 2:
        return {"status": "insufficient rungs"}
    x = np.array([r["n_shingles"] for r in rs], dtype=np.float64) / 1e9
    y = np.array([r["total_peak_rss_mb"] for r in rs], dtype=np.float64) / 1024.0  # GiB
    A = np.vstack([x, np.ones_like(x)]).T
    slope, icept = np.linalg.lstsq(A, y, rcond=None)[0]
    res = {"status": "extrapolated", "fit": "peak_GiB = %.3f * shingles_1e9 + %.2f"
                                          % (slope, icept),
           "slope_gib_per_1e9_shingles": round(float(slope), 3),
           "intercept_gib": round(float(icept), 3),
           "points": [{"n_docs": r["n_docs"], "n_shingles": r["n_shingles"],
                       "peak_gib": round(r["total_peak_rss_mb"] / 1024, 2)} for r in rs]}
    for limit in (58.0, 64.0):
        sh = (limit - icept) / slope if slope > 0 else float("inf")
        docs = sh * 1e9 / (x[-1] * 1e9 / rs[-1]["n_docs"]) if rs else None
        res[f"projected_shingles_at_{int(limit)}GiB"] = round(float(sh), 3)
        res[f"projected_docs_at_{int(limit)}GiB"] = int(docs) if docs else None
    # time fit for the exact reference
    t = np.array([r["stages"].get("rank_build", 0) + r["stages"].get("exact_index_t0.8", 0)
                  + r["stages"].get("exact_probe_t0.8", 0)
                  + r["stages"].get("cluster_exact_t0.8", 0) for r in rs])
    at, bt = np.linalg.lstsq(A, t, rcond=None)[0]
    res["exact_time_fit_s"] = "%.1f * shingles_1e9 + %.1f" % (at, bt)
    # where the 31-bit rank index fails: V = S + non-singletons, asserted V < 2^31-1
    big = max(rs, key=lambda r: r["n_docs"])
    v_per_shingle = big["V"] / big["n_shingles"]
    sh_per_doc = big["n_shingles"] / big["n_docs"]
    res["rank_index_limit_shingles"] = int((2 ** 31 - 1) / v_per_shingle)
    res["rank_index_limit_docs"] = int(res["rank_index_limit_shingles"] / sh_per_doc)
    res["rank_index_v_per_shingle"] = round(v_per_shingle, 5)
    res["largest_rung"] = dict(n_docs=big["n_docs"], n_shingles=big["n_shingles"],
                               V=big["V"])
    return res


def main():
    rows = load_rows()
    order = {"prefix": 0, "strata": 1}
    rows.sort(key=lambda r: (order.get(r["kind"], 9), r["n_docs"], r["label"]))
    ladder, strata = [], []
    for r in rows:
        # run high-water mark over every stage (older rows stored only the final stage's HWM)
        r["total_peak_rss_mb"] = max([r["total_peak_rss_mb"]]
                                     + [v for v in r["peak_rss_mb"].values() if v])
        rec = dict(workload_id=r["workload_id"], kind=r["kind"], label=r["label"],
                   n_docs=r["n_docs"], n_valid=r["n_valid"], n_shingles=r["n_shingles"],
                   nwords_mean=round(r["nwords_mean"], 1),
                   dup_density_14x8=round(r["measured_dup_density_14x8"], 5),
                   total_peak_rss_mb=r["total_peak_rss_mb"], wall_s=r["wall_s"],
                   costs=method_costs(r), stages=r["stages"], bucket_stats=r["bucket_stats"],
                   exact_peak_mb=peak_of(r, ["rank_build", "exact_index_t0.8",
                                             "exact_probe_t0.8", "cluster_exact_t0.8"]),
                   lsh_peak_mb=peak_of(r, ["signature", "banding", "cand_sort",
                                           "bucket_expand", "cand_dedup", "cluster_raw"]),
                   per_tau={t: r["per_tau"][t] for t in TAUS if t in r["per_tau"]})
        (ladder if r["kind"] == "prefix" else strata).append(rec)
    # derived shares at tau=0.8
    for rec in ladder + strata:
        d = rec["per_tau"].get("0.8")
        if not d:
            continue
        rec["at_0.8"] = dict(
            n_truth_pairs=d["n_truth_pairs"], n_cand_pairs=d["n_cand_pairs"],
            raw_removed=d["raw_removed"], raw_unsupported=d["raw_unsupported"],
            raw_unsupported_share=d["raw_unsupported_share_of_deleted"],
            exact_removed=d["exact_removed"], ver_removed=d["ver_removed"],
            ver_missed=d["ver_missed"], ver_unsupported=d["ver_unsupported"],
            wit_removed=d["wit_removed"], wit_missed=d["wit_missed"],
            wit_unsupported=d["wit_unsupported"],
            ver_residual_pairs=d["ver_residual_pairs"],
            exact_residual_pairs=d["exact_residual_pairs"],
            symdiff_raw_exact=d["symdiff_raw_exact"],
            symdiff_ver_exact=d["symdiff_ver_exact"],
            # witness deletions are a subset of verified deletions, so the two directions
            # against the exact join are exactly wit_unsupported + wit_missed
            symdiff_wit_exact=d["wit_unsupported"] + d["wit_missed"],
            raw_cost_per_100k=round(100000 * rec["costs"]["lsh_raw_s"] / rec["n_docs"], 2),
            exact_cost_per_100k=round(100000 * rec["costs"]["exact_s"] / rec["n_docs"], 2),
        )
    summary = dict(version=1,
                   generated_at=__import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                   provenance="independently rerun (this run)",
                   protocol=dict(threads=16, tau=TAUS, banding="14x8", shingles="5-gram word, xxh64",
                                 signature="datatrove permutations, seed 1",
                                 input_order="pool order: FineWeb, C4, raw WET"),
                   ladder=ladder, strata=strata,
                   crosscheck=crosscheck(rows),
                   extrapolation=extrapolate(rows))
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", OUT)
    for rec in ladder + strata:
        c = rec["costs"]
        a = rec.get("at_0.8", {})
        print(f"{rec['workload_id']:<22} n={rec['n_docs']:>8,} sh={rec['n_shingles']/1e6:8.1f}M "
              f"raw={c['lsh_raw_s']:7.1f}s exact={c['exact_s']:8.1f}s "
              f"peak={rec['total_peak_rss_mb']/1024:5.2f}GiB "
              f"unsup={100*(a.get('raw_unsupported_share') or 0):5.1f}% "
              f"res={a.get('ver_residual_pairs')}")
    print("crosscheck", summary["crosscheck"]["status"], summary["crosscheck"].get("checks"))
    print("extrapolation", {k: v for k, v in summary["extrapolation"].items()
                            if k != "points"})


if __name__ == "__main__":
    main()
