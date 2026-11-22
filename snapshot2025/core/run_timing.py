#!/usr/bin/env python3
"""Time ONE configuration (workload, tau, method, mode, threads) in a fresh OS process.

Modes
  cold         read text from the frozen pool -> tokenise -> 5-gram shingles ->
               signatures+bands (A/B/W) or exact-join ranks (C) -> method
  incremental  shared preprocessing reused from disk (shingles + signatures or ranks);
               only loading plus the method itself is timed

Every run appends one row to core/timing.jsonl with per-stage seconds, peak RSS,
bytes actually read from storage (page-cache probe) and the hardware facts.

Called by timing_campaign.py; do not run two configurations in one process.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--method", required=True, choices=["A", "B", "W", "C"])
    ap.add_argument("--mode", required=True, choices=["cold", "incremental"])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--rlimit-gib", type=float, default=0.0)
    args = ap.parse_args()
    if args.rlimit_gib:
        import resource
        lim = int(args.rlimit_gib * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    K.set_threads(args.threads)
    w, tau, method = args.workload, args.tau, args.method
    stages = {}
    rb0 = K.read_bytes()
    t_start = time.perf_counter()
    files_read = {}

    def stage(name, fn):
        t = time.perf_counter()
        r0 = K.read_bytes()
        out = fn()
        stages[name + "_s"] = round(time.perf_counter() - t, 4)
        r1 = K.read_bytes()
        if r0 is not None and r1 is not None:
            files_read[name + "_read_bytes"] = int(r1 - r0)
        return out

    need_lsh = method in ("A", "B", "W")
    if args.mode == "cold":
        off, flat, nw, pre = stage(
            "read_shingle", lambda: K.shingle_from_pool(w, workers=args.threads))
        n = len(off) - 1
        valid = np.diff(off) > 0
        if need_lsh:
            sig = stage("signature", lambda: K.L.signatures(off, flat, K.H, args.seed, "dt"))
            Kb = stage("banding", lambda: K.band_keys(sig))
            del sig
        else:
            RV = stage("ranks", lambda: K.EJ.build_ranks(off, flat))
    else:
        off, flat, nw = stage("load_shingles", lambda: K.load_shingles(w))
        n = len(off) - 1
        valid = np.diff(off) > 0
        if need_lsh:
            sig = stage("load_signatures",
                        lambda: np.load(K.sig_path(w, args.seed)))
            Kb = stage("banding", lambda: K.band_keys(sig))
            del sig
        else:
            z = stage("load_ranks", lambda: np.load(K.RANKS_CACHE[w]))
            RV = (np.asarray(z["R"]), int(z["S"]), int(z["V"]))
            z.close()

    offv, flatv = off, flat  # kept for B's verification; no I/O inside the method
    if method in ("A", "B", "W"):
        if method == "A":
            ei, ej, bstats = stage("chain_edges", lambda: K.chain_edges(Kb, valid))
            root = stage("union_find", lambda: K.uf_minid(n, ei, ej))
            removed = (root != np.arange(n)) & valid
        else:
            pi, pj, bstats = stage("bucket_pairs",
                                   lambda: K.all_bucket_pairs(Kb, valid, n))
            jac = stage("verify", lambda: K.L.jaccard_pairs(offv, flatv, pi, pj))
            ver = jac >= tau - 1e-12
            ei, ej = pi[ver], pj[ver]
            del jac, pi, pj
            if method == "B":
                root = stage("union_find", lambda: K.uf_minid(n, ei, ej))
                removed = (root != np.arange(n)) & valid
            else:
                kept = stage("greedy_mis", lambda: K.greedy_mis(n, ei, ej))
                removed = (~kept) & valid
        edges = int(len(ei))
        bstats = bstats
    else:
        res = stage("exact_join", lambda: K.EJ.exact_join(
            offv, RV[0], RV[1], RV[2], tau, args.threads))
        ei = res["i"].astype(np.int64)
        ej = res["j"].astype(np.int64)
        root = stage("union_find", lambda: K.uf_minid(n, ei, ej))
        removed = (root != np.arange(n)) & valid
        edges = int(len(ei))
        bstats = dict(index_entries=int(res.get("index_entries", -1)),
                      candidates=int(res.get("n_candidates", -1)))

    total = time.perf_counter() - t_start
    rb1 = K.read_bytes()
    row = dict(
        key=f"{args.tag}|{w}|t{tau}|{method}|{args.mode}|thr{args.threads}|r{args.repeat}",
        tag=args.tag, workload=w, tau=tau, method=method, mode=args.mode,
        threads=args.threads, hash_seed=(args.seed if need_lsh else None),
        repeat=args.repeat, n=n, valid_docs=int(valid.sum()),
        deleted=int(removed.sum()), edges=edges,
        **{k: v for k, v in bstats.items() if isinstance(v, (int, float))},
        **stages,
        wall_s=round(total, 3),
        peak_rss_mb=round(K.peak_rss_mb(), 1),
        read_bytes_total=(int(rb1 - rb0) if rb0 is not None and rb1 is not None else None),
        **files_read,
        evidence_tier="rerun", hw=K.hw_facts(), ts_utc=K.now())
    if args.mode == "incremental":
        cache_size = os.path.getsize(K.SHINGLE_CACHE[w])
        row["shingle_cache_bytes"] = cache_size
        row["shingle_cache_read_ratio"] = (files_read.get("load_shingles_read_bytes", 0)
                                           / cache_size if cache_size else None)
    K.append_row(K.TIMING, row)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
