"""Build the document pool for the DedupJoin experiments.

Sources
  fw  : FineWeb sample-10BT, file sample/10BT/000_00000.parquet (1,048,581 rows)
  c4  : C4 en train shards 00000..00007 (first 8 shards, ~356k docs each)
  wet : CC-MAIN-2019-18 WET segments, text-extracted by wet_extract.py

Output
  <pool>/<source>/shard-XXXXX.parquet   id, source, url, text, nchar, nwords
  <pool>/index-XXXXX.parquet            id, row_id, source, shard, local_idx, url,
                                        domain, nchar, nwords, raw_hash, norm_hash

Streaming only: each input shard is read in batches and written out in shards of
`--shard-docs` documents, so peak memory is O(batch + shard).

Hashes
  raw_hash  = xxh64 of the UTF-8 text exactly as stored (byte-identical duplicates)
  norm_hash = xxh64 of " ".join(text.lower().split()) (case/whitespace-insensitive)
"""
import argparse
import gzip
import json
import os
import sys
import time
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

POOL_SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("url", pa.string()),
    ("text", pa.string()), ("nchar", pa.int32()), ("nwords", pa.int32()),
])
INDEX_SCHEMA = pa.schema([
    ("id", pa.string()), ("row_id", pa.int64()), ("source", pa.string()),
    ("shard", pa.string()), ("local_idx", pa.int64()), ("url", pa.string()),
    ("domain", pa.string()), ("nchar", pa.int32()), ("nwords", pa.int32()),
    ("raw_hash", pa.uint64()), ("norm_hash", pa.uint64()),
])

SECOND_LEVEL = {"co", "com", "net", "org", "gov", "ac", "edu", "or", "ne", "go"}


def domain_of(url):
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def hashes(text):
    t = text.encode("utf-8", "replace")
    raw = xxhash.xxh64_intdigest(t)
    norm = " ".join(text.lower().split()).encode("utf-8", "replace")
    return raw, xxhash.xxh64_intdigest(norm)


class PoolWriter:
    def __init__(self, pool_dir, shard_docs, start_row):
        self.pool_dir = pool_dir
        self.shard_docs = shard_docs
        self.row = start_row
        self.buf = {k: [] for k in POOL_SCHEMA.names}
        self.idx = {k: [] for k in INDEX_SCHEMA.names}
        self.shard_no = 0
        self.index_no = 0
        self.counts = {}
        self.bytes = 0

    def _flush_pool(self):
        if not self.buf["id"]:
            return
        src = self.buf["source"][0]
        d = os.path.join(self.pool_dir, src)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"shard-{self.shard_no:05d}.parquet")
        pq.write_table(pa.table(self.buf, schema=POOL_SCHEMA), p,
                       compression="zstd", row_group_size=20000)
        self.bytes += os.path.getsize(p)
        self.shard_no += 1
        self.buf = {k: [] for k in POOL_SCHEMA.names}

    def _flush_index(self):
        if not self.idx["id"]:
            return
        p = os.path.join(self.pool_dir, "meta", f"index-{self.index_no:05d}.parquet")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pq.write_table(pa.table(self.idx, schema=INDEX_SCHEMA), p,
                       compression="zstd", row_group_size=50000)
        self.index_no += 1
        self.idx = {k: [] for k in INDEX_SCHEMA.names}

    def add(self, source, shard, local_idx, doc_id, url, text):
        # never mix sources inside one shard: flush at the source boundary
        if self.buf["id"] and self.buf["source"][0] != source:
            self._flush_pool()
            self._flush_index()
        nchar = len(text)
        nwords = len(text.split())
        rh, nh = hashes(text)
        self.buf["id"].append(doc_id)
        self.buf["source"].append(source)
        self.buf["url"].append(url)
        self.buf["text"].append(text)
        self.buf["nchar"].append(nchar)
        self.buf["nwords"].append(nwords)
        self.idx["id"].append(doc_id)
        self.idx["row_id"].append(self.row)
        self.idx["source"].append(source)
        self.idx["shard"].append(shard)
        self.idx["local_idx"].append(local_idx)
        self.idx["url"].append(url)
        self.idx["domain"].append(domain_of(url))
        self.idx["nchar"].append(nchar)
        self.idx["nwords"].append(nwords)
        self.idx["raw_hash"].append(rh)
        self.idx["norm_hash"].append(nh)
        self.row += 1
        self.counts[source] = self.counts.get(source, 0) + 1
        if len(self.buf["id"]) >= self.shard_docs:
            self._flush_pool()
        if len(self.idx["id"]) >= self.shard_docs:
            self._flush_index()

    def close(self):
        self._flush_pool()
        self._flush_index()


def add_c4(w, c4_dir, shards):
    for shard in shards:
        name = f"{shard:05d}"
        p = os.path.join(c4_dir, f"c4-train.{name}-of-01024.json.gz")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                text = r.get("text", "")
                if not text:
                    continue
                w.add("c4", f"c4-{name}", i, f"c4-{name}-{i}",
                      r.get("url", ""), text)
        print(f"c4 shard {name} done; total={sum(w.counts.values())}", flush=True)


def add_fw(w, fw_path):
    pf = pq.ParquetFile(fw_path)
    i = 0
    for bi in pf.iter_batches(batch_size=10000, columns=["text", "id", "url"]):
        d = bi.to_pydict()
        for text, did, url in zip(d["text"], d["id"], d["url"]):
            w.add("fw", "fw-000_00000", i, did, url, text)
            i += 1
    print(f"fw done: {i} rows", flush=True)


def add_wet(w, wet_dir):
    for fn in sorted(os.listdir(wet_dir)):
        if not fn.endswith(".parquet"):
            continue
        t = pq.read_table(os.path.join(wet_dir, fn),
                          columns=["id", "segment", "record_index", "url", "text"])
        seg = fn[len("wet-"):-len(".parquet")]
        for r in t.to_pylist():
            w.add("wet", f"wet-{seg}", r["record_index"], r["id"], r["url"], r["text"])
        print(f"wet {seg} done; total={sum(w.counts.values())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--c4-dir", required=True)
    ap.add_argument("--fw-parquet", required=True)
    ap.add_argument("--wet-dir", default="")
    ap.add_argument("--c4-shards", default="0-7")
    ap.add_argument("--sources", default="fw,c4,wet")
    ap.add_argument("--shard-docs", type=int, default=100000)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.c4_shards.split("-"))
    shards = list(range(lo, hi + 1))
    os.makedirs(os.path.join(args.pool, "meta"), exist_ok=True)
    w = PoolWriter(args.pool, args.shard_docs, start_row=0)
    t0 = time.time()
    if "fw" in args.sources:
        add_fw(w, args.fw_parquet)
    if "c4" in args.sources:
        add_c4(w, args.c4_dir, shards)
    if "wet" in args.sources:
        add_wet(w, args.wet_dir)
    w.close()
    summary = {"counts": w.counts, "rows": w.row, "pool_bytes": w.bytes,
               "pool_shards": w.shard_no, "index_shards": w.index_no,
               "seconds": round(time.time() - t0, 1), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(args.pool, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
