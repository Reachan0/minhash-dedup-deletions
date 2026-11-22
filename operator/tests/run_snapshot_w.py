#!/usr/bin/env python3
"""Exact W (code/operator/exact_w.py) on the uncollapsed 531,597-document snapshot.

For seeds 1..5 and tau 0.8/0.75, compares the deletion mask with the frozen-pipeline W mask
(newsnap/cache/masks/wit_s<seed>_t<tau>.npz, computed on collapsed groups and lifted) and records
wall time and peak RSS. Appends rows to results_post_reveal/operator_E.jsonl.
"""
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.dirname(HERE)
PKG = os.path.abspath(os.path.join(OP, "..", ".."))
sys.path[:0] = [OP, os.path.join(PKG, "code", "newsnap"), os.path.join(PKG, "code", "run")]
import snap_common as S  # noqa: E402
from exact_w import exact_w  # noqa: E402
from stream_b import candidate_keys  # noqa: E402


def main():
    off, flat, _ = S.load_snapshot()
    valid = np.diff(off) > 0
    out = open(os.path.join(PKG, "results_post_reveal", "operator_E.jsonl"), "a")
    for seed in range(1, 6):
        t0 = time.time()
        keys = candidate_keys(off, flat, seed)
        t_keys = time.time() - t0
        for tau in (0.8, 0.75):
            t1 = time.time()
            res = exact_w(off, flat, keys, tau, valid=valid)
            rem = np.asarray(res["deleted"], bool)
            t_w = time.time() - t1
            with np.load(os.path.join(S.CACHE, "masks", f"wit_s{seed}_t{tau}.npz")) as f:
                ref = f["rem"]
            row = dict(experiment="E", method="W", mode="exact_w_no_collapse", seed=seed, tau=tau,
                       n_docs=int(len(rem)), equal_existing_mask=bool((rem == ref).all()),
                       differing_docs=int((rem != ref).sum()), key_wall_s=t_keys, w_wall_s=t_w,
                       peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            row.update({k: v for k, v in res.items() if isinstance(v, (int, float))})
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
    print("W_SNAPSHOT_DONE", flush=True)


if __name__ == "__main__":
    main()
