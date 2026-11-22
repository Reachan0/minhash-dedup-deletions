#!/usr/bin/env python3
"""Exact-duplicate shingle-set groups (diagnostic for the reveal).

Each document's shingle set is stored sorted and unique. Hashing the raw
bytes of that slice gives an exact identity of the set; grouping identifies
documents whose 5-gram shingle sets are identical (J = 1). Needed to size the
all-pairs paths: a clique of m identical documents costs m(m-1)/2 edges.
"""
import json
import os
import time

import numpy as np
import xxhash

import env_paths

S = env_paths.prepared()

out_path = os.path.join(S.SNAP, "dup_sets.json")


def main():
    t0 = time.perf_counter()
    off, flat, nw = S.load_snapshot()
    n = len(off) - 1
    h = np.empty(n, np.uint64)
    sizes = np.empty(n, np.int64)
    for i in range(n):
        a, b = int(off[i]), int(off[i + 1])
        sizes[i] = b - a
        h[i] = xxhash.xxh64_intdigest(flat[a:b].tobytes()) if b > a else np.uint64(0)
    t_hash = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, inv, counts = np.unique(h, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    top = []
    for g in order[:20]:
        members = np.flatnonzero(inv == g)
        top.append(dict(set_hash=int(h[members[0]]), n_docs=int(counts[g]),
                        n_shingles=int(sizes[members[0]]),
                        first_ids=[str(x) for x in members[:3]]))
    nonzero = counts[sizes[np.unique(h, return_index=True)[1]] > 0] if False else counts
    res = dict(n=n, n_unique_sets=int(len(counts)),
               n_docs_in_duplicate_groups=int(counts[counts >= 2].sum()),
               n_duplicate_groups=int((counts >= 2).sum()),
               largest_group=int(counts.max()),
               top=top,
               t_hash_s=round(t_hash, 2),
               t_group_s=round(time.perf_counter() - t0, 2),
               peak_rss_mb=round(S.K.peak_rss_mb(), 1), ts_utc=S.now())
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
