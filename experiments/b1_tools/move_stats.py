#!/usr/bin/env python3
"""Data movement of bucket-partitioned verification: which documents need their shingle sets shipped.
A document's set is needed only in buckets (band, key) holding two or more valid documents; with buckets
assigned to workers by key, it travels at most once per such bucket. Writes results/move_stats.jsonl."""
import json, sys
import numpy as np
W = "/path/to/scratch/dedupjoin"
for name in sys.argv[1:]:
    d = f"{W}/runs/{name}"
    with np.load(f"{d}/snapshot.npz") as z:
        off = z["off"].astype(np.int64)
    keys = np.load(f"{d}/keys_s1.npy")
    nsh = np.diff(off)
    valid = nsh > 0
    n = len(nsh)
    cnt = np.zeros(n, np.int64)
    for b in range(keys.shape[0]):
        k = keys[b]
        kv = k[valid]
        u, inv, c = np.unique(kv, return_inverse=True, return_counts=True)
        multi = np.zeros(n, bool)
        multi[np.flatnonzero(valid)] = c[inv] >= 2
        cnt += multi
    moved = int((cnt * nsh).sum())
    total = int(nsh.sum())
    row = dict(corpus=name, docs=int(n), valid=int(valid.sum()), docs_in_multi_bucket=int((cnt > 0).sum()),
               share_docs=float((cnt > 0).sum() / valid.sum()), mean_multi_buckets_per_moved_doc=float(cnt[cnt > 0].mean()),
               shingles_total=total, shingle_copies_moved=moved, moved_over_total=moved / total,
               moved_bytes=moved * 8, total_bytes=total * 8, a_key_bytes=int(valid.sum()) * 14 * 16)
    print(json.dumps(row), flush=True)
    with open(f"{W}/results/move_stats.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
