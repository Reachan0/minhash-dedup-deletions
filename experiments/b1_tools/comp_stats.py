#!/usr/bin/env python3
"""Partition by candidate component: every within-bucket pair lies inside one component of the band
collision graph (A's union-find), so a worker owning whole components needs each document's set once.
Reports the shingle bytes of documents in non-singleton components and of the largest component."""
import json, sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
W = "/path/to/scratch/dedupjoin"
for name in sys.argv[1:]:
    d = f"{W}/runs/{name}"
    with np.load(f"{d}/snapshot.npz") as z:
        off = z["off"].astype(np.int64)
    keys = np.load(f"{d}/keys_s1.npy")
    nsh = np.diff(off)
    valid = nsh > 0
    ids = np.flatnonzero(valid)
    n = len(nsh)
    ei, ej = [], []
    for b in range(keys.shape[0]):
        k = keys[b, ids]
        o = np.argsort(k, kind="stable")
        ks, ds = k[o], ids[o]
        same = ks[1:] == ks[:-1]
        ei.append(ds[:-1][same]); ej.append(ds[1:][same])
    ei = np.concatenate(ei); ej = np.concatenate(ej)
    g = coo_matrix((np.ones(len(ei), np.int8), (ei, ej)), shape=(n, n)).tocsr()
    nc, lab = connected_components(g, directed=False)
    csize = np.bincount(lab, minlength=nc)
    cbytes = np.bincount(lab, weights=nsh * 8, minlength=nc)
    multi = valid & (csize[lab] >= 2)
    row = dict(corpus=name, docs_in_multi_component=int(multi.sum()),
               bytes_multi=int((nsh[multi] * 8).sum()), bytes_total=int((nsh[valid] * 8).sum()),
               share_bytes=float((nsh[multi]).sum() / nsh[valid].sum()),
               largest_component_docs=int(csize.max()), largest_component_bytes=int(cbytes.max()),
               second_component_bytes=int(np.sort(cbytes)[-2]))
    print(json.dumps(row), flush=True)
    with open(f"{W}/results/move_stats.jsonl", "a") as f:
        f.write(json.dumps(dict(kind="component", **row)) + "\n")
