"""Extract text records from Common Crawl WET segments into sharded parquet.

Input : WET .warc.wet.gz files (CC-MAIN-2019-18, the crawl from which C4 was built).
Output: one parquet shard per segment with id, segment, url, fetched_at, text,
        nchar, nwords, lang, lang_score, plus a per-segment stats jsonl line.

WARC parsing is done by hand (fixed header block + Content-Length bytes of UTF-8
text); this is the format documented at
https://commoncrawl.org/the-data/get-started/ and in the WARC 1.0 spec.
Language identification is fastText lid.176 (Joulin et al., 2016), the same
model family the C4 pipeline used; the low-level predictor is called because
fasttext-wheel's wrapper is incompatible with NumPy 2.
"""
import argparse
import gzip
import io
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


def iter_wet(path):
    """Yield (headers dict, text bytes) for each WARC record in a WET gzip file."""
    with gzip.open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith(b"WARC/"):
                # tolerate blank separators between records
                if line.strip() == b"":
                    continue
                raise ValueError(f"unexpected WARC start {line[:40]!r}")
            headers = {}
            while True:
                h = f.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode("utf-8", "replace").partition(":")
                headers[k.strip()] = v.strip()
            n = int(headers.get("Content-Length", "0"))
            body = f.read(n)
            yield headers, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--segments", required=True,
                    help="JSON file: list of {name, path} WET segments")
    ap.add_argument("--lang-model", required=True)
    ap.add_argument("--limit", type=int, default=0, help="records per segment (0=all)")
    args = ap.parse_args()

    import fasttext
    ft = fasttext.load_model(args.lang_model)

    os.makedirs(args.out, exist_ok=True)
    segments = json.load(open(args.segments))
    stats_path = os.path.join(args.out, "extract_stats.jsonl")
    if os.path.exists(stats_path):
        os.remove(stats_path)

    schema = pa.schema([
        ("id", pa.string()), ("segment", pa.string()), ("record_index", pa.int32()),
        ("url", pa.string()), ("fetched_at", pa.string()), ("text", pa.string()),
        ("nchar", pa.int32()), ("nwords", pa.int32()),
        ("lang", pa.string()), ("lang_score", pa.float32()),
    ])

    for seg in segments:
        t0 = time.time()
        name = seg["name"]
        out_path = os.path.join(args.out, f"wet-{name}.parquet")
        writer = pq.ParquetWriter(out_path + ".tmp", schema, compression="zstd")
        batch = {k: [] for k in schema.names}
        n_rec = n_doc = 0
        lang_counts = {}
        t_extract = t_lang = 0.0
        flush = 2000

        def flush_batch():
            nonlocal batch
            if not batch["id"]:
                return
            writer.write_table(pa.table({k: pa.array(v, type=schema.field(k).type)
                                         for k, v in batch.items()}))
            batch = {k: [] for k in schema.names}

        for headers, body in iter_wet(seg["path"]):
            if headers.get("WARC-Type") != "conversion":
                continue
            n_rec += 1
            text = body.decode("utf-8", "replace").strip()
            if not text:
                continue
            t = time.time()
            try:
                pred = ft.f.predict(text.replace("\n", " ")[:20000], 1, 0.0, "strict")
                lang = pred[0][1].replace("__label__", "")
                score = float(pred[0][0])
            except Exception:
                lang, score = "unknown", 0.0
            t_lang += time.time() - t
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            nwords = len(text.split())
            batch["id"].append(f"wet-{name}-{n_rec - 1}")
            batch["segment"].append(name)
            batch["record_index"].append(n_rec - 1)
            batch["url"].append(headers.get("WARC-Target-URI", ""))
            batch["fetched_at"].append(headers.get("WARC-Date", ""))
            batch["text"].append(text)
            batch["nchar"].append(len(text))
            batch["nwords"].append(nwords)
            batch["lang"].append(lang)
            batch["lang_score"].append(score)
            n_doc += 1
            if len(batch["id"]) >= flush:
                flush_batch()
            if args.limit and n_rec >= args.limit:
                break
        flush_batch()
        writer.close()
        os.replace(out_path + ".tmp", out_path)
        top = sorted(lang_counts.items(), key=lambda kv: -kv[1])[:8]
        st = {"segment": name, "records": n_rec, "docs": n_doc,
              "lang_top": top, "seconds": round(time.time() - t0, 1),
              "lang_seconds": round(t_lang, 1),
              "sha256": None}
        import hashlib
        h = hashlib.sha256()
        with open(out_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        st["sha256"] = h.hexdigest()
        st["size_bytes"] = os.path.getsize(out_path)
        with open(stats_path, "a") as f:
            f.write(json.dumps(st) + "\n")
        print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
