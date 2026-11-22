#!/usr/bin/env python3
"""Cost estimator (Section 10) on the filtered snapshot and its injected variants (B2).

The four ladder corpora share the bot-challenge template, so their estimator errors do not test
other bucket-size tails. The filtered snapshot has no template group, and each injected variant
adds one bucket of m copies or near-copies (m = 1e3, 1e4, 1e5) at fixed size.
For each corpus: seed-1 14x8 band keys on all documents, full within-bucket pair count and largest
bucket, then 20 uniform samples at s = 2000 and 5000 (rng 7, as in f_estimators.py), each scaled
by n(n-1)/(s(s-1)) and n/s.
Usage: env/bin/python f_est_b2.py   -> results/F_b2.jsonl
"""
import json
import sys

import numpy as np

W = "/path/to/scratch/dedupjoin"
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/operator", f"{W}/dedupjoin-newsnap-selection/code/core"]
from stream_b import candidate_keys  # noqa: E402

VARIANTS = ["base", "exact_1000", "exact_10000", "exact_100000", "near_1000", "near_10000", "near_100000"]


def bucket_stats(kb):
    pairs, largest = 0, 0
    for b in range(kb.shape[0]):
        _, c = np.unique(kb[b], return_counts=True)
        c = c.astype(np.int64)
        pairs += int((c * (c - 1) // 2).sum())
        largest = max(largest, int(c.max()))
    return pairs, largest


def main():
    out = open(f"{W}/results/F_b2.jsonl", "a")
    for v in VARIANTS:
        with np.load(f"{W}/b2/{v}.npz") as z:
            off, flat = z["off"].astype(np.int64), z["flat"]
        valid = np.diff(off) > 0
        kb = candidate_keys(off, flat, 1)
        del flat
        vidx = np.flatnonzero(valid)
        nv = len(vidx)
        full_pairs, full_max = bucket_stats(kb[:, vidx])
        rng = np.random.default_rng(7)
        for s in (2000, 5000):
            for r in range(20):
                idx = np.sort(rng.choice(vidx, s, replace=False))
                sp, sm = bucket_stats(kb[:, idx])
                est_pairs = sp * nv * (nv - 1) / (s * (s - 1))
                est_max = sm * nv / s
                out.write(json.dumps(dict(kind="cost", corpus=v, n=int(nv), s=s, design=f"random{r}",
                                          sample_pairs=sp, sample_max=sm, est_pairs=est_pairs,
                                          full_pairs=full_pairs, rel_err_pairs=est_pairs / full_pairs - 1,
                                          est_max=est_max, full_max=full_max,
                                          rel_err_max=est_max / full_max - 1)) + "\n")
        out.flush()
        print(v, nv, full_pairs, full_max, flush=True)
        del kb
    print("F_B2_DONE", flush=True)


if __name__ == "__main__":
    main()
