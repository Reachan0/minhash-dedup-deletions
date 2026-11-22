#!/usr/bin/env python3
"""Shingle cache for the pool, one file per 100k-document pool shard.

The scale ladder reads pool prefixes (fw -> c4 -> wet, the order in which the pool
was written) and stratified samples.  Re-shingling the same 4M documents at every
rung would dominate the ladder's cost, so each pool shard is shingled once with the
paper's protocol (`C03/common.py:shingle_doc`, datatrove simplify_text + 5-gram
xxhash sets, no length cap) and cached as

  .work/dedupjoin/ladder/shards/<src>-<idx:05d>.npz
      off     int64 (n_docs+1)   relative offsets into flat
      flat    uint64             sorted unique 5-gram hashes
      nwords  int32 (n_docs)
      nchar   int32 (n_docs)

Note: the pool's own band pass (pool_minhash.py) used a simplified punctuation table
and a 20k-shingle cap, so its band keys are re-computed here from these shingles
under the paper protocol; it is only used to label the strata.

Usage: python3 scale_shards.py [--workers 8] [--force]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
C03 = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, C03)
import common as C  # noqa: E402

POOL = os.path.join(ROOT, ".work/dedupjoin/pool")
OUT = os.path.join(ROOT, ".work/dedupjoin/ladder/shards")


def shards():
    out = []
    for src in ("fw", "c4", "wet"):
        for p in sorted(glob.glob(os.path.join(POOL, src, "*.parquet"))):
            out.append((src, p))
    return out


def one(job):
    src, path = job
    idx = int(os.path.basename(path).split("-")[1].split(".")[0])
    dest = os.path.join(OUT, f"{src}-{idx:05d}.npz")
    if os.path.exists(dest):
        return dest, 0, 0
    t0 = time.time()
    d = pq.read_table(path, columns=["text", "nchar"]).to_pydict()
    texts = d["text"]
    nchar = np.asarray(d["nchar"], np.int32)
    sizes = np.empty(len(texts), np.int64)
    flats, nws = [], np.empty(len(texts), np.int32)
    for i, t in enumerate(texts):
        s, w = C.shingle_doc(t)
        flats.append(s)
        sizes[i] = len(s)
        nws[i] = w
    off = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=off[1:])
    flat = np.concatenate(flats) if flats else np.zeros(0, np.uint64)
    del flats, texts
    tmp = dest + ".tmp.npz"
    np.savez(tmp, off=off, flat=flat, nwords=nws, nchar=nchar)
    os.replace(tmp, dest)
    return dest, len(off) - 1, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = shards()
    if a.force:
        for _, p in jobs:
            idx = int(os.path.basename(p).split("-")[1].split(".")[0])
            src = os.path.basename(os.path.dirname(p))
            d = os.path.join(OUT, f"{src}-{idx:05d}.npz")
            if os.path.exists(d):
                os.remove(d)
    from concurrent.futures import ProcessPoolExecutor
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for dest, n, sec in ex.map(one, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {os.path.basename(dest)} n={n} {sec}s "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("all shards cached", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
