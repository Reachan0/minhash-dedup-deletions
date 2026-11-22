#!/usr/bin/env python3
"""Nearest-neighbour distribution for the *audited* (shared-117) C4 cell.

The exact pair list at floor 0.3 is draw-independent, so the same complete-input
neighbourhood is valid for any banding. This script takes the audited design's
raw-deletion and unsupported-deletion document IDs from the earlier run's dumps and
reports their exact nearest-neighbour distribution and retained-set maximum.

Writes results/nn_audited.json.
"""
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")
OLD_DUMP = os.path.join(ROOT, ".work/data/C03/r2/cache/lsh_out")
BINS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0001]


def main():
    n = 700000
    p = np.load(os.path.join(WRK, f"pairs_c4_{n}_0.3.npz"))
    pi, pj, jac = p["i"].astype(np.int64), p["j"].astype(np.int64), p["jac"]
    nsh = np.asarray(pq.read_table(os.path.join(WRK, f"c4_{n}_meta.parquet"),
                                  columns=["nshingle"]).column("nshingle"))
    maxj = np.full(n, -1.0)
    np.maximum.at(maxj, pi, jac)
    np.maximum.at(maxj, pj, jac)

    out = {}
    for tau in (0.8, 0.7):
        st = np.load(os.path.join(WRK, f"sets_c4_{n}_s1_14x8_t{tau}.npz"))
        keep = ~st["rem_exact"].astype(bool)
        rmax = np.full(n, -1.0)
        m1 = keep[pj]
        np.maximum.at(rmax, pi[m1], jac[m1])
        m2 = keep[pi]
        np.maximum.at(rmax, pj[m2], jac[m2])
        z = np.load(os.path.join(OLD_DUMP, f"c4_{n}_{tau}_dt1_14x8.npz"))
        wrong = z["wrong"].astype(np.int64)
        v = maxj[wrong]
        vk = rmax[wrong]
        short = nsh[wrong] < 100
        hist = {f"[{lo},{hi})": int(((v >= lo) & (v < hi)).sum())
                for lo, hi in zip(BINS[:-1], BINS[1:])}
        out[str(tau)] = dict(
            design="audited shared-117 14x8 cell, seed 1",
            n_wrongly_deleted=int(len(wrong)),
            n_no_pair_at_floor=int((v < 0).sum()),
            hist=hist,
            frac_lt_0_5=float((v < 0.5).mean()), frac_lt_0_6=float((v < 0.6).mean()),
            coverage_at_0_5=float((v >= 0.5).mean()), coverage_at_0_6=float((v >= 0.6).mean()),
            coverage_at_0_7=float((v >= 0.7).mean()) if tau == 0.8 else None,
            median_ge_floor=float(np.median(v[v >= 0])),
            p10_ge_floor=float(np.percentile(v[v >= 0], 10)),
            short_docs=dict(n_short=int(short.sum()), frac_short=float(short.mean()),
                            corpus_frac_short=float((nsh < 100).mean()),
                            median_nshingle=float(np.median(nsh[wrong])),
                            frac_lt_0_6_short=float((v[short] < 0.6).mean()),
                            frac_lt_0_6_long=float((v[~short] < 0.6).mean()),
                            median_short=float(np.median(v[short][v[short] >= 0])),
                            median_long=float(np.median(v[~short][v[~short] >= 0]))),
            retained_set=dict(frac_none_below_floor=float((vk < 0).mean()),
                              coverage_at_0_5=float((vk >= 0.5).mean()),
                              coverage_at_0_6=float((vk >= 0.6).mean())))
    # Bucket-limited comparison, recomputed from the earlier run's analyze log.
    old_log = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/logs/C03/round2/analyze_c4_700000.log")
    bucket = None
    with open(old_log) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "analysis" and r.get("tau") == 0.8 and r.get("n") == n:
                b = r["wrong_maxjac_bins"]
                den = r["wrongly_deleted"]
                below6 = b["<0.3"] + b["0.3-0.5"] + b["0.5-0.6"]
                below5 = b["<0.3"] + b["0.3-0.5"]
                bucket = dict(source="earlier run analyze.py, log-recomputed",
                              key=r["key"], n_wrongly_deleted=den,
                              max_over="co-bucketed candidate partners only",
                              frac_lt_0_6=below6 / den, frac_lt_0_5=below5 / den,
                              at_or_above_0_6=(b["0.6-0.7"] + b["0.7-tau"]) / den,
                              bins=b,
                              exact_frac_lt_0_6=out["0.8"]["frac_lt_0_6"],
                              exact_frac_lt_0_5=out["0.8"]["frac_lt_0_5"])
    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run): pair list; earlier run: deletion ids",
               cell=dict(corpus="c4", n=n, tau=[0.8, 0.7], seed=1, floor=0.3,
                         design="audited shared 117-hash draw, first 112 columns, 14x8"),
               method="exact Jaccard over every pair with J>=0.3 in the complete input; "
                      "per-document maximum over incident pairs; identical pair list for any draw",
               n_pairs_floor=int(len(pi)),
               bucket_limited=bucket,
               per_tau=out)
    with open(os.path.join(OUT, "nn_audited.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
