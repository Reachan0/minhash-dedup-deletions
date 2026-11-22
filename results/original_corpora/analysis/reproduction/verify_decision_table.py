#!/usr/bin/env python3
"""Recompute the main decision table from per-run records and compare every field.

Inputs (copied from the published experiment tree):
  runs.jsonl                 one row per (workload, tau, method, representative, seed)
  timing.jsonl               one row per timed configuration
  main_decision_table.csv    the published table
Evidence tier of the result: log-recomputed.
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "inputs")
runs = [json.loads(l) for l in open(os.path.join(IN, "runs.jsonl"))]
trows = [json.loads(l) for l in open(os.path.join(IN, "timing.jsonl"))]
pub = list(csv.DictReader(open(os.path.join(IN, "main_decision_table.csv"))))
STAGE_ALGO = ("chain_edges", "union_find", "bucket_pairs", "verify", "greedy_mis", "exact_join")
STAGE_PREPROC = ("read_shingle", "ranks", "signature", "banding")

def stage(r, names):
    return float(sum(r.get(n + "_s", 0.0) for n in names))

def med(v):
    return float(np.median(v)) if v else None

def timing_cells(w, tau):
    cell = {}
    for m in "ABWC":
        inc = [r for r in trows if r["workload"] == w and abs(r["tau"]-tau) < 1e-9
               and r["method"] == m and r["mode"] == "incremental"
               and r["threads"] == 12 and r.get("tag") == "main"]
        cm = "B" if m == "W" else m
        cold = [r for r in trows if r["workload"] == w and abs(r["tau"]-0.8) < 1e-9
                and r["method"] == cm and r["mode"] == "cold"
                and r["threads"] == 12 and r.get("tag") == "main"]
        inc1 = [r for r in trows if r["workload"] == w and abs(r["tau"]-tau) < 1e-9
                and r["method"] == m and r["mode"] == "incremental"
                and r["threads"] == 1 and r.get("tag") == "main"]
        c = {}
        if inc:
            c["incremental_median_s"] = med([r["wall_s"] for r in inc])
            c["incremental_peak_rss_mb"] = med([r["peak_rss_mb"] for r in inc])
        if inc1:
            c["incremental_1thread_median_s"] = med([r["wall_s"] for r in inc1])
        if cold:
            c["cold_peak_rss_mb"] = med([r["peak_rss_mb"] for r in cold])
            c["shared_preproc_median_s"] = med([stage(r, STAGE_PREPROC) for r in cold])
            if abs(tau-0.8) < 1e-9 and m != "W":
                c["cold_median_s"] = med([r["wall_s"] for r in cold])
            elif inc:
                c["cold_composed_median_s"] = c["shared_preproc_median_s"] + med(
                    [stage(r, STAGE_ALGO) for r in inc])
        cell[m] = c
    return cell

def close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a)-float(b)) <= tol * max(1.0, abs(float(b)))

checked = failed = 0
for row in pub:
    w, tau, m, rep = row["workload"], float(row["tau"]), row["method"], row["representative"]
    sub = [r for r in runs if r["workload"] == w and abs(r["tau"]-tau) < 1e-9
           and r["method"] == m and r["representative"] == rep]
    if not sub:
        print("MISSING runs", w, tau, m, rep); failed += 1; continue
    dele = np.array([r["deleted"] for r in sub], float)
    uns = np.array([r["unsupported"] for r in sub], float)
    exp = {
        "deleted_mean": dele.mean(),
        "unsupported_mean": uns.mean(),
        "unsupported_share_del_mean": float(np.mean([r["unsupported_share_del"] for r in sub])),
        "unsupported_share_input_mean": float(np.mean([r["unsupported_share_input"] for r in sub])),
        "witness_coverage_mean": float(np.mean([r["witness_coverage"] for r in sub])),
        "residual_duplicate_pairs_mean": float(np.mean([r["residual_duplicate_pairs"] for r in sub])),
    }
    exp.update(timing_cells(w, tau)[m])
    for k, v in exp.items():
        got = row.get(k)
        checked += 1
        if not close(v, got if got not in ("", None) else None, 1e-6):
            failed += 1
            print("MISMATCH", w, tau, m, rep, k, "recomputed", v, "published", got)
print(json.dumps({"fields_checked": checked, "failures": failed,
                  "rows": len(pub), "runs": len(runs), "timing_rows": len(trows)}))
sys.exit(1 if failed else 0)
