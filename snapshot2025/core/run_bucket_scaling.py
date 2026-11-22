#!/usr/bin/env python3
"""One point of the path-B bucket-expansion scaling experiment, in a fresh process.

Given a pool prefix (source, N), build/load the shingles, obtain the 112-permutation
signature (sliced from the frozen full-workload cache when N fits, otherwise computed),
then enumerate ALL pairs inside every band bucket and record:
  largest bucket, pairs before dedup (multiplicity over 14 bands), unique pairs after
  dedup, wall time, peak RSS, and whether a memory limit was hit.
The orchestrator (scaling_campaign.py) grows N until a run fails or the memory cap is
reached, so the last safe size is recorded rather than OOM-killed.

Usage: python3 run_bucket_scaling.py --source c4 --n 1400000 --rlimit-gib 40
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K

SCALE_DIR = os.path.join(K.CORE, "scaling")


def known_workload(source, n):
    """A frozen workload that is exactly this source prefix, if any."""
    m = {"c4": ("c4", 700000), "fw": ("fw", 1000000), "wet": ("wet", 175272)}
    if source in m and n == m[source][1]:
        return m[source][0]
    return None


def get_shingles(source, n, workers):
    w = known_workload(source, n)
    if w:
        off, flat, _ = K.load_shingles(w)
        return off, flat, {"shingles_from": K.SHINGLE_CACHE[w], "shingles_cold": False}
    p = os.path.join(SCALE_DIR, f"shingles_{source}_{n}.npz")
    if os.path.exists(p):
        z = np.load(p)
        off, flat = np.asarray(z["off"], np.int64), np.asarray(z["flat"], np.uint64)
        z.close()
        return off, flat, {"shingles_from": p, "shingles_cold": False}
    t0 = time.perf_counter()
    off, flat, _, pre = K.shingle_pool_prefix(source, n, workers=workers)
    os.makedirs(SCALE_DIR, exist_ok=True)
    np.savez(p, off=off, flat=flat)
    return off, flat, {"shingles_from": p, "shingles_cold": True,
                       "t_read_shingle_s": round(time.perf_counter() - t0, 2)}


def get_signatures(source, n, off, flat):
    w = known_workload(source, n)
    if w:
        p = K.sig_path(w, 1)
        if os.path.exists(p):
            return np.load(p), {"signatures_from": p, "signatures_cold": False}
    p = os.path.join(SCALE_DIR, f"sig_{source}_{n}_s1.npy")
    if os.path.exists(p):
        return np.load(p), {"signatures_from": p, "signatures_cold": False}
    t0 = time.perf_counter()
    sig = K.L.signatures(off, flat, K.H, 1, "dt")
    os.makedirs(SCALE_DIR, exist_ok=True)
    np.save(p, sig)
    return sig, {"signatures_from": p, "signatures_cold": True,
                 "t_signature_s": round(time.perf_counter() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=K.THREADS)
    ap.add_argument("--rlimit-gib", type=float, default=40.0)
    args = ap.parse_args()
    K.set_threads(args.workers)
    if args.rlimit_gib:
        import resource
        lim = int(args.rlimit_gib * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    status = "ok"
    row = dict(key=f"{args.source}|n{args.n}", source=args.source, n=args.n,
               workers=args.workers, rlimit_gib=args.rlimit_gib,
               evidence_tier="rerun", ts_utc=K.now(), hw=K.hw_facts())
    t0 = time.perf_counter()
    try:
        off, flat, sh = get_shingles(args.source, args.n, args.workers)
        n = len(off) - 1
        valid = np.diff(off) > 0
        sig, sh_sig = get_signatures(args.source, n, off, flat)
        sig = sig[:n]
        t1 = time.perf_counter()
        Kb = K.band_keys(sig)
        del sig
        t2 = time.perf_counter()
        pi, pj, bstats = K.all_bucket_pairs(Kb, valid, n)
        t3 = time.perf_counter()
        row.update(n=n, valid_docs=int(valid.sum()),
                   largest_bucket=bstats["largest_bucket"],
                   bucket_pairs_sum=bstats["bucket_pairs_sum"],
                   n_pairs_mult=bstats["n_pairs_mult"],
                   n_pairs_unique=bstats["n_pairs_unique"],
                   unique_pairs_bytes=int(bstats["n_pairs_unique"]) * 16,
                   t_banding_s=round(t2 - t1, 2), t_pairs_s=round(t3 - t2, 2),
                   **sh, **sh_sig)
        del pi, pj, Kb
    except MemoryError as e:
        status = "memory_error"
        row["error"] = repr(e)
    except Exception as e:  # noqa: BLE001 - any failure must leave a row behind
        status = f"error:{type(e).__name__}"
        row["error"] = repr(e)
    row["status"] = status
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    row["peak_rss_mb"] = round(K.peak_rss_mb(), 1)
    K.append_row(K.SCALING, row)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
