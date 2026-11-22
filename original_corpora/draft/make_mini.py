"""Small public mini-corpus for parity tests.

Fixed 5,000 documents, no sampling randomness: the first 2,000 FineWeb rows,
the first 2,000 C4 shard-0 lines, and the first 1,000 English WET documents
(top-1 language en, score >= 0.8) in extraction order.

Output: <pool>/mini/mini.parquet + <pool>/mini/manifest.json (sha256, ids, counts)
"""
import argparse
import glob
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wet-raw", required=True,
                    help="wet_extract.py output (has lang/lang_score columns)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = {"id": [], "source": [], "url": [], "text": []}

    def take(src, path, n, pred=None):
        got = 0
        for f in path:
            for bi in pq.ParquetFile(f).iter_batches(batch_size=2000):
                d = bi.to_pydict()
                for i in range(len(d["id"])):
                    if pred and not pred(d, i):
                        continue
                    rows["id"].append(d["id"][i])
                    rows["source"].append(src)
                    rows["url"].append(d["url"][i])
                    rows["text"].append(d["text"][i])
                    got += 1
                    if got >= n:
                        return got
        return got

    fw = sorted(glob.glob(os.path.join(args.pool, "fw", "*.parquet")))[:1]
    c4 = sorted(glob.glob(os.path.join(args.pool, "c4", "*.parquet")))[:1]
    wet = sorted(glob.glob(os.path.join(args.wet_raw, "wet-*.parquet")))
    counts = {"fw": take("fw", fw, 2000), "c4": take("c4", c4, 2000),
              "wet_en": take("wet", wet, 1000,
                             pred=lambda d, i: d["lang"][i] == "en" and d["lang_score"][i] >= 0.8)}
    schema = pa.schema([("id", pa.string()), ("source", pa.string()),
                        ("url", pa.string()), ("text", pa.string())])
    p = os.path.join(args.out, "mini.parquet")
    pq.write_table(pa.table(rows, schema=schema), p, compression="zstd")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    manifest = {"provenance": "independently rerun (this run)", "path": p, "sha256": h.hexdigest(), "size_bytes": os.path.getsize(p),
                "counts": counts, "total": len(rows["id"]),
                "definition": ("first 2,000 FineWeb rows (file order), first 2,000 C4 "
                               "shard-0 lines, first 1,000 WET documents of the English "
                               "subset (lang=en, score>=0.8) in segment order"),
                "first_ids": {k: rows["id"][next(i for i, s in enumerate(rows["source"]) if s == k)]
                              for k in ("fw", "c4")}}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
