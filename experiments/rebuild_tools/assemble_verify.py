#!/usr/bin/env python3
"""Assemble the six corpus shards from eng/<rank>.parquet and verify document identity.

Order: segments by frozen rank 0..103, records in file order (as extracted); 100,000 rows per shard.
Identity checks against the frozen references (read-only):
  * per-document xxh64 of the UTF-8 text equals inputs/doc_text_xxh64.npy, element by element;
  * id_sequence_sha256 (sha256 over LE32(len(utf8(id))) || utf8(id)) equals INPUT_PATHS.json;
  * total words equals 522,207,916 and the per-segment English counts equal the manifest.
Shard bytes are NOT expected to match the original files (writer metadata may differ).
Writes shards to <package>/newsnap/corpus/ and a report to logs/identity_check.json.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

W = "/path/to/home/dedupjoin-postreveal"
PKG = os.path.join(W, "dedupjoin-newsnap-selection")
CORPUS = os.path.join(PKG, "newsnap", "corpus")
REF_HASH = os.path.join(PKG, "inputs", "doc_text_xxh64.npy")
REF_PATHS = os.path.join(PKG, "inputs", "INPUT_PATHS.json")
MANIFEST = os.path.join(W, "dedupjoin-pvldb-v3", "experiment", "data", "newsnap", "manifest.json")
ROWS = 100_000


def main():
    man = json.load(open(MANIFEST))
    ref = np.load(REF_HASH)
    refp = json.load(open(REF_PATHS))
    segs = sorted(man["segments"], key=lambda s: s["rank"])
    tables = [pq.read_table(os.path.join(W, "eng", f"{s['rank']:03d}.parquet")) for s in segs]
    seg_counts_ok = all(t.num_rows == s["english_docs"] for t, s in zip(tables, segs))
    t = pa.concat_tables(tables).combine_chunks()
    n = t.num_rows
    texts = t["text"].to_pylist()
    ids = t["id"].to_pylist()
    h = np.fromiter((xxhash.xxh64_intdigest(x.encode("utf-8")) for x in texts), dtype=np.uint64, count=n)
    idh = hashlib.sha256()
    for d in ids:
        b = d.encode("utf-8")
        idh.update(len(b).to_bytes(4, "little"))
        idh.update(b)
    same_len = (n == ref.shape[0])
    mism = int((h != ref).sum()) if same_len else None
    first_bad = int(np.argmax(h != ref)) if same_len and mism else None
    total_words = int(np.asarray(t["nwords"].to_numpy(), dtype=np.int64).sum())
    rep = dict(n_docs=n, n_ref=int(ref.shape[0]), segment_counts_match=seg_counts_ok,
               text_hash_mismatches=mism, first_mismatch_row=first_bad,
               id_sequence_sha256=idh.hexdigest(),
               id_sequence_matches=(idh.hexdigest() == refp["id_sequence_sha256"]),
               total_words=total_words, total_words_matches=(total_words == 522_207_916))
    rep["identity_ok"] = bool(same_len and mism == 0 and rep["id_sequence_matches"]
                              and rep["total_words_matches"] and seg_counts_ok)
    if rep["identity_ok"] or (len(sys.argv) > 1 and sys.argv[1] == "--write-anyway"):
        os.makedirs(CORPUS, exist_ok=True)
        for k in range(0, n, ROWS):
            pq.write_table(t.slice(k, ROWS), os.path.join(CORPUS, f"shard-{k // ROWS:05d}.parquet"),
                           compression="zstd")
        rep["shards_written"] = (n + ROWS - 1) // ROWS
    json.dump(rep, open(os.path.join(W, "logs", "identity_check.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
