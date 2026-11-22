#!/usr/bin/env python3
"""Bandings whose design point reaches the audit threshold (none of the four swept bandings
had a design point >= 0.8). Path A (min-id) on C4 and WET-2019 with the pool caches and the exact
truth of the rebuilt pool; 14x8 is rerun as a check against the published sweep.
Writes results/banding_hi.jsonl (BANDING_OUT overrides), with C's deletions and A's missed exact deletions, one row per (corpus, banding, seed, tau).
Usage: env/bin/python banding_hi.py <core dir>
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
import core_lib as K  # noqa: E402

L = K.L
OUT = os.environ.get("BANDING_OUT", "/path/to/scratch/dedupjoin/results/banding_hi.jsonl")
CONFIGS = [(14, 8), (10, 11), (9, 12)]


def design(b, r):
    return float((1.0 - 0.5 ** (1.0 / b)) ** (1.0 / r))


def main():
    K.set_threads(12)
    for w in ("c4", "wet"):
        off, flat, _ = K.load_shingles(w)
        n = len(off) - 1
        valid = np.diff(off) > 0
        truth = {}
        for tau in (0.8, 0.75):
            ti, tj, _ = K.compute_truth(w, tau, off)
            truth[tau] = (ti, tj, K.has_true_neighbour(n, ti, tj),
                          (K.uf_minid(n, ti, tj) != np.arange(n)) & valid)  # C's deletions
        seeds = range(1, 6) if w == "c4" else range(1, 4)
        for b, r in CONFIGS:
            for seed in seeds:
                t0 = time.time()
                sig = L.signatures(off, flat, b * r, seed, "dt")
                kb = L.band_keys(sig, b, r)
                del sig
                ei, ej, _ = K.chain_edges(kb, valid)
                rem = (K.uf_minid(n, ei, ej) != np.arange(n)) & valid
                for tau, (ti, tj, hnb, remC) in truth.items():
                    m = K.method_metrics(n, rem, ti, tj, hnb)
                    m["exact_deletions"] = int(remC.sum())
                    m["missed_exact"] = int((remC & ~rem).sum())
                    m.update(K.edge_stats(ti, tj, ei, ej, n))
                    row = dict(corpus=w, bands=b, rows=r, perms=b * r, design=design(b, r), seed=seed, tau=tau,
                               n=n, t_s=round(time.time() - t0, 1),
                               **{k: v for k, v in m.items() if isinstance(v, (int, float)) or v is None})
                    with open(OUT, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(w, f"{b}x{r}", seed, tau, m["deleted"], m["unsupported"],
                          round(m["unsupported_share_del"], 4), m["exact_deletions"], m["missed_exact"], flush=True)
                del ei, ej, rem
    print("BANDING_HI_DONE", flush=True)


if __name__ == "__main__":
    main()
