#!/usr/bin/env python3
"""Exact Jaccard graph on the collapsed corpus (one node per distinct shingle set).

This is the ground truth for the reveal. It is exact: two distinct sets are
joined iff their exact Jaccard >= tau, and identical documents within a group
are J = 1 by construction. A timing row (variant=collapsed_exact) is recorded
for transparency; it is not the frozen path-C cost (which is attempted
literally elsewhere and fails on the 3.76-billion-edge clique).
"""
import json
import os
import time

import numpy as np

import collapsed_lib as CL

S, K = CL.S, CL.K
TAUS = [0.8, 0.75]


def main():
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    G = len(z["group_size"])
    stages = {}
    t_start = time.perf_counter()
    if os.path.exists(CL.RANKS_U):
        t = time.perf_counter()
        R, Sv, V = CL.ranks_u()
        stages["load_ranks_s"] = round(time.perf_counter() - t, 3)
    else:
        t = time.perf_counter()
        R, Sv, V, dt = CL.build_ranks_u()
        stages["ranks_s"] = round(time.perf_counter() - t, 2)
    for tau in TAUS:
        t = time.perf_counter()
        ti, tj, jac = CL.truth_u(tau)
        stages[f"exact_join_t{tau}_s"] = round(time.perf_counter() - t, 2)
        print(f"truth tau={tau}: {len(ti)} group edges, {time.perf_counter()-t:.1f}s",
              flush=True)
    total = time.perf_counter() - t_start
    row = dict(key="snap|collapsed|C|exact_truth", workload="snap", tau=None,
               method="C", mode="collapsed_exact", variant="collapsed_exact",
               threads=K.THREADS, n_groups=G, edges_t08=None,
               **stages, wall_s=round(total, 3),
               peak_rss_mb=round(K.peak_rss_mb(), 1),
               evidence_tier="rerun", hw=K.hw_facts(), ts_utc=S.now())
    S.append_row(os.path.join(S.SNAP, "timing.jsonl"), row)
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
