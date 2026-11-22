#!/usr/bin/env python3
"""Path C, core-benchmark cold boundary, with the exact edge set cached as truth.

This is run_timing.py's C pathcold branch executed verbatim in stage order
(load shingles -> build/reuse ranks -> exact join -> union-find), plus one
side effect that does not change the method: the exact (i, j, jac) edge arrays
are written to snap_common.truth_path(tau) so that run_snapshot_eval.py does
not pay for the same join a second time. A timing row with the same schema as
run_timing.py is appended to newsnap/timing.jsonl.

Usage: c_pathcold.py --tau 0.8 --repeat 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import env_paths

S = env_paths.prepared()
K = S.K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
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

    off, flat, nw = stage("load_shingles", S.load_snapshot)
    n = len(off) - 1
    valid = np.diff(off) > 0
    if os.path.exists(S.RANKS):
        z = stage("load_ranks", lambda: np.load(S.RANKS))
        R, Sv, V = np.asarray(z["R"]), int(z["S"]), int(z["V"])
        z.close()
    else:
        R, Sv, V, _ = stage("ranks", S.build_ranks)
    res = stage("exact_join", lambda: K.EJ.exact_join(off, R, Sv, V, args.tau, K.THREADS))
    ei = res["i"].astype(np.int64)
    ej = res["j"].astype(np.int64)
    jac = res["jac"].astype(np.float64)
    root = stage("union_find", lambda: K.uf_minid(n, ei, ej))
    removed = (root != np.arange(n)) & valid
    edges = int(len(ei))

    # cache the truth edge set (idempotent; no method change)
    p = S.truth_path(args.tau)
    if not os.path.exists(p):
        tmp = p + ".tmp.npz"
        np.savez(tmp, i=ei, j=ej, jac=jac)
        os.replace(tmp, p)

    total = time.perf_counter() - t_start
    row = dict(key=f"snap|t{args.tau}|C|pathcold|thr{K.THREADS}|r{args.repeat}",
               workload="snap", tau=args.tau, method="C", mode="pathcold",
               threads=K.THREADS, hash_seed=None, repeat=args.repeat, n=n,
               valid_docs=int(valid.sum()), deleted=int(removed.sum()), edges=edges,
               index_entries=int(res.get("index_entries", -1)),
               candidates=int(res.get("n_candidates", -1)),
               **stages, wall_s=round(total, 3),
               peak_rss_mb=round(K.peak_rss_mb(), 1),
               evidence_tier="rerun", hw=K.hw_facts(), ts_utc=S.now())
    S.append_row(os.path.join(S.SNAP, "timing.jsonl"), row)
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
