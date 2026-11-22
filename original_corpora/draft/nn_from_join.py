#!/usr/bin/env python3
"""Correct the earlier nearest-neighbour binning without a bucket-limited query.

The earlier analysis (experiment/src/C03/analyze.py) stored, for every wrongly
deleted document, the maximum Jaccard similarity among pairs that shared an LSH bucket
with it (and -1 when no bucket partner existed).  That quantity is a property of the
banding, not of the data: "22% of wrongful deletions have their nearest neighbour below
0.6" was computed from bucket-mates only.

Here the complete input is queried exactly: one exact self-join at a low floor
tau_floor returns every pair with exact Jaccard >= tau_floor, over all documents.
For each wrongly deleted document the maximum over incident pairs gives its exact
nearest-neighbour similarity whenever that maximum is >= tau_floor; a document with no
incident pair is reported as `<tau_floor`.  The retained-set maximum restricts the
partner to documents the exact join keeps at the query threshold.

Query thresholds: C4 0.8/0.7/0.6, FineWeb 0.8/0.7.  Floor 0.3 by default.

Writes results/nn_binning.json (merged per corpus when run with --corpus all) and
results/nn_docs_{corpus}.jsonl.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")
BINS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0001]


def one(corpus, n, taus, floor):
    C.set_threads(8)
    tag = f"{corpus}_{n}"
    z = np.load(os.path.join(WRK, f"{tag}.npz"), mmap_mode="r")
    off = np.asarray(z["off"])
    flat = z["flat"]
    sizes = np.diff(off)
    nsh = np.asarray(pq.read_table(os.path.join(WRK, f"{tag}_meta.parquet"),
                                   columns=["nshingle"]).column("nshingle"))
    t0 = time.time()
    pp = os.path.join(WRK, f"pairs_{tag}_{floor}.npz")
    if os.path.exists(pp):
        q = np.load(pp)
        pi, pj, pj_jac = q["i"].astype(np.int64), q["j"].astype(np.int64), q["jac"]
        t_floor = None
        floor_from_cache = True
    else:
        R, S, V = EJ.build_ranks(off, flat)
        del flat
        res = EJ.exact_join(off, R, S, V, floor, 8, nchunks=256)
        pi, pj, pj_jac = res["i"].astype(np.int64), res["j"].astype(np.int64), res["jac"]
        np.savez(pp, i=pi, j=pj, jac=pj_jac)
        t_floor = time.time() - t0
        floor_from_cache = False
        del R
    # per-document maximum over all incident pairs at >= floor
    maxj = np.full(n, -1.0)
    np.maximum.at(maxj, pi, pj_jac)
    np.maximum.at(maxj, pj, pj_jac)

    rows = []
    per_tau = {}
    for tau in taus:
        st = np.load(os.path.join(WRK, f"sets_{tag}_s1_14x8_t{tau}.npz"))
        wrong = st["rem_lsh"].astype(bool) & ~st["has_nb"].astype(bool)
        keep = ~st["rem_exact"].astype(bool)
        qs = np.nonzero(wrong)[0]
        # retained-set maximum: for pair (i,j), i gets the value only if j is kept and
        # j gets it only if i is kept (the partner must survive the exact join)
        rmax = np.full(n, -1.0)
        m1 = keep[pj]
        np.maximum.at(rmax, pi[m1], pj_jac[m1])
        m2 = keep[pi]
        np.maximum.at(rmax, pj[m2], pj_jac[m2])
        v = maxj[qs]
        vk = rmax[qs]
        for k, q in enumerate(qs):
            rows.append(dict(corpus=corpus, n=n, tau=tau, doc=int(q), nshingle=int(nsh[q]),
                             max_j=float(v[k]) if v[k] >= 0 else None,
                             max_j_retained=float(vk[k]) if vk[k] >= 0 else None))
        vv = v[v >= 0]
        vvk = vk[vk >= 0]
        short = nsh[qs] < 100
        per_tau[str(tau)] = dict(
            n_wrongly_deleted=int(len(qs)),
            n_with_no_pair_at_floor=int((v < 0).sum()),
            floor=floor,
            complete_input=dict(
                n=int(len(vv)),
                hist={f"[{lo},{hi})": int(((vv >= lo) & (vv < hi)).sum())
                      for lo, hi in zip(BINS[:-1], BINS[1:])},
                frac_lt_0_3=float((v < 0.3).mean()) if len(v) else None,
                frac_lt_0_5=float((v < 0.5).mean()) if len(v) else None,
                frac_lt_0_6=float((v < 0.6).mean()) if len(v) else None,
                coverage_at_0_5=float((v >= 0.5).mean()) if len(v) else None,
                coverage_at_0_6=float((v >= 0.6).mean()) if len(v) else None,
                coverage_at_0_7=float((v >= 0.7).mean()) if len(v) else None,
                median_ge_floor=float(np.median(vv)) if len(vv) else None,
                p10_ge_floor=float(np.percentile(vv, 10)) if len(vv) else None),
            retained_set=dict(
                n=int(len(vvk)),
                frac_none_below_floor=float((vk < 0).mean()) if len(vk) else None,
                coverage_at_0_5=float((vk >= 0.5).mean()) if len(vk) else None,
                coverage_at_0_6=float((vk >= 0.6).mean()) if len(vk) else None,
                coverage_at_0_7=float((vk >= 0.7).mean()) if len(vk) else None),
            short_docs=dict(n_short=int(short.sum()), frac_short=float(short.mean()),
                            n_long=int((~short).sum()),
                            median_nshingle=float(np.median(nsh[qs])) if len(qs) else None,
                            frac_with_pair_at_floor_short=float((v[short] >= 0).mean()) if short.any() else None,
                            frac_with_pair_at_floor_long=float((v[~short] >= 0).mean()) if (~short).any() else None,
                            median_max_j_short=float(np.median(v[short])) if short.any() else None,
                            median_max_j_long=float(np.median(v[~short])) if (~short).any() else None,
                            frac_below_0_5_short=float((v[short] < 0.5).mean()) if short.any() else None,
                            frac_below_0_5_long=float((v[~short] < 0.5).mean()) if (~short).any() else None),
            n_pairs_floor=int(len(pi)), floor_from_cache=floor_from_cache, t_floor_s=t_floor)
    with open(os.path.join(OUT, f"nn_docs_{corpus}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    doc = dict(version=2, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)",
               cell=dict(corpus=corpus, n=n, seed=1, config="dt 14x8 standalone 112-hash draw",
                         query_taus=taus, floor=floor),
               method="exact self-join at floor tau over the complete input; per-document maximum "
                      "over incident pairs. A query with no incident pair has true maximum < floor "
                      "and is counted as such (never as 0 similarity). The retained-set maximum "
                      "restricts the partner to exact-join kept documents.",
               correction=("The earlier binning used only same-bucket candidates. Bucket "
                           "co-membership is a property of the LSH; a document with no bucket "
                           "partner scored -1 and was counted as 'no neighbour', which inflates "
                           "the far-away share. Values below are exact over the complete input."),
               per_tau=per_tau)
    print(json.dumps(doc, indent=1), flush=True)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="c4")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--taus", default="0.8,0.7,0.6")
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        cells = {}
        for corpus, n, taus in (("c4", 700000, "0.8,0.7,0.6"), ("fw", 1000000, "0.8,0.7")):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--corpus", corpus,
                                "--n", str(n), "--taus", taus, "--floor", str(a.floor)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
                raise SystemExit(f"corpus {corpus} failed")
            m = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
            cells[corpus] = m
            print(corpus, "done", flush=True)
        doc = dict(version=2, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   provenance="independently rerun (this run)",
                   cells=cells)
        with open(os.path.join(OUT, "nn_binning.json"), "w") as f:
            json.dump(doc, f, indent=2)
        print("wrote results/nn_binning.json")
        return
    one(a.corpus, a.n, [float(x) for x in a.taus.split(",")], a.floor)


if __name__ == "__main__":
    main()
