#!/usr/bin/env python3
"""Band-bucket statistics for the frozen reveal (no pair materialisation).

Loads the cached shingles, builds seed-1 signatures (cached), then per band
reports the bucket-size distribution: max bucket, number of buckets >= 2,
sum of s(s-1)/2 over buckets (candidate pairs before cross-band dedup), and the
ten largest buckets. This sizes the B/W candidate expansion before committing
to the full run; peak RSS is recorded so the memory plan is measured, not
guessed.
"""
import json
import os
import sys
import time

import numpy as np

import env_paths

S = env_paths.prepared()
K = S.K


def main():
    K.set_threads(K.THREADS)
    out = {}
    t0 = time.perf_counter()
    off, flat, nw = S.load_snapshot()
    n = len(off) - 1
    valid = np.diff(off) > 0
    out["load_shingles_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    sig, cached = K.signatures_cached(off, flat, "snap", 1)
    out["signature_s"] = round(time.perf_counter() - t0, 2)
    out["signature_cached"] = bool(cached)
    t0 = time.perf_counter()
    Kb = K.band_keys(sig)
    del sig
    out["banding_s"] = round(time.perf_counter() - t0, 2)
    idx = np.nonzero(valid)[0]
    per_band = []
    tot_pairs = 0
    largest = 0
    largest_band = None
    top = []
    for b in range(Kb.shape[0]):
        k = np.asarray(Kb[b, idx])
        o = np.argsort(k, kind="stable")
        ks = k[o]
        brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[brk[1:], len(ks)]
        sz = ends - brk
        multi = sz[sz >= 2]
        pairs = int((multi.astype(np.int64) * (multi - 1) // 2).sum())
        tot_pairs += pairs
        if len(sz) and int(sz.max()) > largest:
            largest = int(sz.max())
            largest_band = b
        for s_ in np.sort(sz[sz >= 2])[-3:]:
            top.append((int(s_), b))
        per_band.append(dict(band=b, n_buckets=int(len(sz)),
                             n_multi=int(len(multi)),
                             max_bucket=int(sz.max()) if len(sz) else 0,
                             pairs=pairs))
    top.sort(reverse=True)
    out.update(n=int(n), valid_docs=int(valid.sum()),
               largest_bucket=int(largest), largest_bucket_band=largest_band,
               bucket_pairs_sum=int(tot_pairs),
               pairs_per_doc=float(tot_pairs / max(int(valid.sum()), 1)),
               top_buckets=[{"size": int(s), "band": int(b)} for s, b in top[:10]],
               per_band=per_band,
               wall_s=round(time.perf_counter() - t0, 2) + out["load_shingles_s"]
               + out["signature_s"],
               peak_rss_mb=round(K.peak_rss_mb(), 1),
               evidence_tier="rerun", hw=K.hw_facts(), ts_utc=S.now())
    p = os.path.join(S.SNAP, "band_stats.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n", "valid_docs", "largest_bucket",
                                          "bucket_pairs_sum", "pairs_per_doc",
                                          "signature_s", "banding_s",
                                          "peak_rss_mb")}), flush=True)
    print("top", out["top_buckets"], flush=True)


if __name__ == "__main__":
    main()
