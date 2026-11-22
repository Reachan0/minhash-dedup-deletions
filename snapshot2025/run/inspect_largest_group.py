#!/usr/bin/env python3
"""Describe the largest identical-shingle-set group from the corpus itself.

Writes newsnap/largest_group.json: size, shingle-set size, first document row,
its id/url/character count and the first 120 characters of its text. This is
descriptive evidence for the duplicate-density finding, not a method input.
"""
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = os.path.dirname(HERE)
SNAP = os.path.join(SLUG, "newsnap")
CACHE = os.path.join(SNAP, "cache")
CORPUS = os.path.join(SLUG, "..", "dedupjoin-pvldb-v3", "newsnap", "corpus")

z = np.load(os.path.join(CACHE, "collapsed_sets.npz"))
g, m, rep, nsh = z["group_of_doc"], z["group_size"], z["group_rep"], z["group_nshingle"]
big = int(np.argmax(m))
row_idx = int(rep[big])
docs = np.where(g == big)[0]

row = None
tot = 0
for f in sorted(glob.glob(os.path.join(CORPUS, "shard-*.parquet"))):
    t = pq.read_table(f)
    if row_idx < tot + len(t):
        row = t.slice(row_idx - tot, 1).to_pylist()[0]
        break
    tot += len(t)

txt = row.get("text") or ""
out = dict(largest_group=int(big), size=int(m[big]),
           n_shingles=int(nsh[big]), first_doc_row=row_idx,
           first_docs=[int(x) for x in docs[:10]],
           last_doc_row=int(docs[-1]),
           doc_id=row.get("id"), url=row.get("url"),
           nchar=int(row.get("nchar") or len(txt)),
           first_120_chars=txt[:120])
json.dump(out, open(os.path.join(SNAP, "largest_group.json"), "w"),
          indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False))
