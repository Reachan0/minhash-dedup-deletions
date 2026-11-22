#!/usr/bin/env python3
"""Time ONE (method, tau, mode) on the snapshot in a fresh OS process.

Modes (same boundary as the core benchmark):
  cold         fresh process re-shingles the snapshot from the parquet shards
               (spacy tokenizer), then signatures+bands (A/B/W) or ranks+join (C),
               then the method.  Expensive (~10 min); composed cold in the report
               = this measured common shingle stage + the measured path stage.
  pathcold     fresh process, cached shingles, signatures/bands or ranks rebuilt,
               then the method (the per-path part of the cold run).
  incremental  cached shingles + signatures/ranks; only load + method timed.

Rows append to newsnap/timing.jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402
K = S.K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["A", "B", "W", "C"])
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--mode", required=True, choices=["cold", "pathcold", "incremental"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=0)
    args = ap.parse_args()
    K.set_threads(K.THREADS)
    stages = {}
    t_start = time.perf_counter()

    def stage(name, fn):
        t = time.perf_counter()
        out = fn()
        stages[name + "_s"] = round(time.perf_counter() - t, 4)
        return out

    if args.mode == "cold":
        off, flat, nw, meta = S.shingle_all()   # re-shingle from the parquet shards
        n = len(off) - 1
        valid = np.diff(off) > 0
    else:
        off, flat, nw = stage("load_shingles", S.load_snapshot)
        n = len(off) - 1
        valid = np.diff(off) > 0

    need_lsh = args.method in ("A", "B", "W")
    if need_lsh:
        if args.mode == "incremental":
            sig = stage("load_signatures", lambda: np.load(K.sig_path("snap", args.seed)))
        else:
            sig = stage("signature", lambda: K.L.signatures(off, flat, K.H, args.seed, "dt"))
        Kb = stage("banding", lambda: K.band_keys(sig))
        del sig
    else:
        if args.mode == "incremental":
            z = stage("load_ranks", lambda: np.load(S.RANKS))
            RV = (np.asarray(z["R"]), int(z["S"]), int(z["V"]))
            z.close()
        else:
            RV = stage("ranks", lambda: S.build_ranks())
            RV = (RV[0], RV[1], RV[2])

    if args.method in ("A", "B", "W"):
        if args.method == "A":
            ei, ej, bstats = stage("chain_edges", lambda: K.chain_edges(Kb, valid))
            root = stage("union_find", lambda: K.uf_minid(n, ei, ej))
            removed = (root != np.arange(n)) & valid
        else:
            pi, pj, bstats = stage("bucket_pairs", lambda: K.all_bucket_pairs(Kb, valid, n))
            jac = stage("verify", lambda: K.L.jaccard_pairs(off, flat, pi, pj))
            ver = jac >= args.tau - 1e-12
            ei, ej = pi[ver], pj[ver]
            del jac, pi, pj
            if args.method == "B":
                root = stage("union_find", lambda: K.uf_minid(n, ei, ej))
                removed = (root != np.arange(n)) & valid
            else:
                kept = stage("greedy_mis", lambda: K.greedy_mis(n, ei, ej))
                removed = (~kept) & valid
        edges = int(len(ei))
    else:
        res = stage("exact_join", lambda: K.EJ.exact_join(off, RV[0], RV[1], RV[2],
                                                          args.tau, K.THREADS))
        ei, ej = res["i"].astype(np.int64), res["j"].astype(np.int64)
        root = stage("union_find", lambda: K.uf_minid(n, ei, ej))
        removed = (root != np.arange(n)) & valid
        edges = int(len(ei))
        bstats = dict(index_entries=int(res.get("index_entries", -1)),
                      candidates=int(res.get("n_candidates", -1)))

    total = time.perf_counter() - t_start
    row = dict(key=f"snap|t{args.tau}|{args.method}|{args.mode}|thr{K.THREADS}|r{args.repeat}",
               workload="snap", tau=args.tau, method=args.method, mode=args.mode,
               threads=K.THREADS, hash_seed=(args.seed if need_lsh else None),
               repeat=args.repeat, n=n, valid_docs=int(valid.sum()),
               deleted=int(removed.sum()), edges=edges,
               **{k: v for k, v in bstats.items() if isinstance(v, (int, float))},
               **stages, wall_s=round(total, 3),
               peak_rss_mb=round(K.peak_rss_mb(), 1),
               evidence_tier="rerun", hw=K.hw_facts(), ts_utc=S.now())
    S.append_row(os.path.join(S.SNAP, "timing.jsonl"), row)
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
