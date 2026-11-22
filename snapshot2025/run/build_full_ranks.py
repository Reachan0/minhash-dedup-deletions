#!/usr/bin/env python3
"""Build the frozen C path's rank cache once, so that the literal C attempt can
start from the incremental boundary (cached shingles + cached ranks)."""
import json
import os
import time

import env_paths

S = env_paths.prepared()
K = S.K
K.set_threads(K.THREADS)

t0 = time.perf_counter()
R, Sv, V, dt = S.build_ranks()
wall = time.perf_counter() - t0
row = dict(key="snap|C|ranks_build", workload="snap", method="C",
           mode="ranks_build", tau=None, threads=K.THREADS,
           n=len(R), n_rank_space=int(V), build_s=round(dt, 3),
           wall_s=round(wall, 3), peak_rss_mb=round(K.peak_rss_mb(), 1),
           evidence_tier="rerun", hw=K.hw_facts(), ts_utc=S.now())
S.append_row(os.path.join(S.SNAP, "timing.jsonl"), row)
print(json.dumps(row, sort_keys=True), flush=True)
