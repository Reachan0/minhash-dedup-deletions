#!/usr/bin/env python3
"""Fix the nearest-neighbour binning of the earlier analysis.

The old analyze.py searched only same-bucket candidates when it asked "how close is a
wrongly deleted document to its nearest neighbour".  bucket membership is a property of
the LSH, not of the data, so that measure is circular.  Here every wrongly deleted
document is queried against the COMPLETE input (an inverted index over all 5-gram
shingles, exact Jaccard over every document that shares at least one shingle) and
against the FINAL RETAINED SET of the exact join at the same tau.

For each tau in {0.8, 0.7, 0.6} (FineWeb: {0.8, 0.7}) it reports the exact max-Jaccard
distribution, the coverage at 0.5/0.6/0.7, and the short-document question.

Writes results/nn_binning.json and results/nn_docs_{corpus}.jsonl (one row per query).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")
BINS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0001]


def build_index(off, flat):
    t0 = time.time()
    uniq = np.unique(flat)
    inv = np.searchsorted(uniq, flat)
    n = len(off) - 1
    sizes = np.diff(off)
    rows = np.repeat(np.arange(n, dtype=np.int64), sizes)
    order = np.argsort(inv, kind="stable")
    postings = rows[order].astype(np.int32)
    counts = np.bincount(inv, minlength=len(uniq))
    del rows, order, inv
    indptr = np.zeros(len(uniq) + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    del counts
    return dict(uniq=uniq, indptr=indptr, postings=postings, build_s=time.time() - t0)


def query(idx, off, flat, sizes, q, n, keep):
    sid = np.searchsorted(idx["uniq"], flat[off[q]:off[q + 1]])
    if len(sid) == 0:
        return 0.0, -1, 0.0, -1, 0
    segs = [idx["postings"][idx["indptr"][s]:idx["indptr"][s + 1]] for s in sid]
    cand = np.concatenate(segs) if segs else np.zeros(0, np.int32)
    if cand.size == 0:
        return 0.0, -1, 0.0, -1, 0
    cnt = np.bincount(cand, minlength=n)
    nonz = np.nonzero(cnt)[0]
    o = cnt[nonz].astype(np.float64)
    Lq = float(len(sid))
    jac = o / (Lq + sizes[nonz] - o)
    jac[nonz == q] = -1.0
    k = int(np.argmax(jac))
    best, best_id, n_cand = float(jac[k]), int(nonz[k]), int(len(nonz))
    jk = jac.copy()
    jk[~keep[nonz]] = -1.0
    kk = int(np.argmax(jk))
    best_keep, best_keep_id = float(jk[kk]), int(nonz[kk])
    if best_keep < 0:
        best_keep, best_keep_id = 0.0, -1
    return best, best_id, best_keep, best_keep_id, n_cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--taus", default="0.8,0.7,0.6")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    C.set_threads(8)
    tag = f"{a.corpus}_{a.n}"
    taus = [float(x) for x in a.taus.split(",")]

    z = np.load(os.path.join(WRK, f"{tag}.npz"), mmap_mode="r")
    off = np.asarray(z["off"])
    flat = z["flat"]
    sizes = np.diff(off).astype(np.float64)
    n = a.n
    sets = {tau: dict(np.load(os.path.join(WRK, f"sets_{tag}_s1_14x8_t{tau}.npz"))) for tau in taus}

    idx = build_index(off, flat)
    print("index built", idx["build_s"], "unique shingles", len(idx["uniq"]), flush=True)

    # validation hook: exact join at tau=0.5 (C4) should agree with the index for J>=0.5
    pairmax = {}
    p50 = os.path.join(WRK, f"pairs_{tag}_0.5.npz")
    if os.path.exists(p50):
        p = np.load(p50)
        for i, j, s in zip(p["i"].astype(np.int64), p["j"].astype(np.int64), p["jac"]):
            if s >= 0.5:
                pairmax[i] = max(pairmax.get(i, 0.0), float(s))
                pairmax[j] = max(pairmax.get(j, 0.0), float(s))

    rows = []
    for tau in taus:
        st = sets[tau]
        wrong = st["rem_lsh"].astype(bool) & ~st["has_nb"].astype(bool)
        qs = np.nonzero(wrong)[0]
        if a.limit:
            qs = qs[:a.limit]
        keep = ~st["rem_exact"].astype(bool)
        t0 = time.time()
        for zk, q in enumerate(qs):
            b, bid, bk, bkid, ncand = query(idx, off, flat, sizes, int(q), n, keep)
            rows.append(dict(corpus=a.corpus, n=n, tau=tau, doc=int(q),
                             nshingle=int(sizes[q]), max_j=b, max_j_doc=bid,
                             max_j_retained=bk, max_j_retained_doc=bkid, n_candidates=ncand,
                             pairmax_ge_0_5=pairmax.get(int(q))))
            if zk % 1000 == 0 and zk:
                print(f"tau {tau}: {zk}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
        print(f"tau {tau}: queried {len(qs)} in {time.time() - t0:.1f}s", flush=True)

    with open(os.path.join(OUT, f"nn_docs_{a.corpus}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def dist(vals):
        v = np.asarray(vals, dtype=np.float64)
        hist = {}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            hist[f"[{lo},{hi})"] = int(((v >= lo) & (v < hi)).sum())
        return dict(n=int(len(v)), hist=hist,
                    frac_lt_0_3=float((v < 0.3).mean()) if len(v) else None,
                    frac_lt_0_5=float((v < 0.5).mean()) if len(v) else None,
                    frac_lt_0_6=float((v < 0.6).mean()) if len(v) else None,
                    frac_lt_0_7=float((v < 0.7).mean()) if len(v) else None,
                    median=float(np.median(v)) if len(v) else None,
                    p10=float(np.percentile(v, 10)) if len(v) else None,
                    p90=float(np.percentile(v, 90)) if len(v) else None)

    per_tau = {}
    for tau in taus:
        rs = [r for r in rows if r["tau"] == tau]
        v = [r["max_j"] for r in rs]
        vk = [r["max_j_retained"] for r in rs]
        short = np.array([r["nshingle"] < 100 for r in rs])
        varr = np.asarray(v)
        per_tau[str(tau)] = dict(
            n_wrongly_deleted=len(rs),
            complete_input=dist(v),
            retained_set=dist(vk),
            coverage_at_0_5=float(np.mean(varr >= 0.5)) if len(v) else None,
            coverage_at_0_6=float(np.mean(varr >= 0.6)) if len(v) else None,
            coverage_at_0_7=float(np.mean(varr >= 0.7)) if len(v) else None,
            median_nshingle=float(np.median([r["nshingle"] for r in rs])) if rs else None,
            frac_nshingle_lt100=float(short.mean()) if len(short) else None,
            max_j_median_short=float(np.median(varr[short])) if short.any() else None,
            max_j_median_long=float(np.median(varr[~short])) if (~short).any() else None)
        if pairmax:
            chk = [r for r in rs if r["pairmax_ge_0_5"] is not None]
            per_tau[str(tau)]["validation_pairs_ge_0_5"] = dict(
                n=len(chk),
                agree=sum(abs(r["max_j"] - r["pairmax_ge_0_5"]) <= 1e-9 for r in chk),
                max_abs_diff=max((abs(r["max_j"] - r["pairmax_ge_0_5"]) for r in chk), default=0.0))

    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)",
               cell=dict(corpus=a.corpus, n=n, seed=1, config="dt 14x8",
                         taus=taus, query_set="documents removed by raw LSH with no exact neighbour at tau"),
               method="exact Jaccard against every document sharing >=1 shingle (inverted index over the full "
                      "input); retained-set maximum restricts candidates to the exact-join kept set",
               index_build_s=idx["build_s"], n_unique_shingles=int(len(idx["uniq"])),
               per_tau=per_tau)
    with open(os.path.join(OUT, "nn_binning.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(per_tau, indent=1))


if __name__ == "__main__":
    main()
