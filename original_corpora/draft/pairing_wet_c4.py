"""Pair raw crawl documents with the released C4 derivative by URL.

Both sides come from CC-MAIN-2019-18: the WET segments are the raw text
extraction of the crawl, C4 (en) is the released corpus built from that crawl
after its own filtering and exact line/page deduplication.  A pair is defined
by an identical URL after normalisation (scheme dropped, host lower-cased and
`www.` removed, trailing slash removed); the query string is kept as-is.

Outputs (under --out):
  wet_c4_pairing.json   counts, length and 5-gram Jaccard distributions
  wet_c4_pairs_sample.jsonl  first 300 matched pairs with both texts truncated
"""
import argparse
import glob
import gzip
import json
import os
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pyarrow.parquet as pq

from pool_minhash import shingle_doc


def norm_url(u):
    try:
        p = urlsplit(u.strip())
    except ValueError:
        return ""
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/")
    return urlunsplit(("", host, path, p.query, ""))


def jacc(a, b):
    sa, sb = shingle_doc(a), shingle_doc(b)
    if len(sa) == 0 and len(sb) == 0:
        return 1.0
    inter = np.intersect1d(sa, sb, assume_unique=True).size
    union = len(sa) + len(sb) - inter
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wet-dir", required=True)
    ap.add_argument("--c4-dir", required=True)
    ap.add_argument("--c4-shards", default="0-7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang-score", type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wet = {}
    n_wet = n_wet_en = 0
    for p in sorted(glob.glob(os.path.join(args.wet_dir, "wet-*.parquet"))):
        t = pq.read_table(p, columns=["id", "url", "text", "lang", "lang_score", "nwords"])
        d = t.to_pydict()
        for i in range(len(d["id"])):
            n_wet += 1
            if d["lang"][i] != "en" or d["lang_score"][i] < args.lang_score:
                continue
            k = norm_url(d["url"][i])
            if not k or k in wet:
                continue
            wet[k] = {"id": d["id"][i], "url": d["url"][i], "text": d["text"][i],
                      "nwords": d["nwords"][i]}
            n_wet_en += 1
    print(f"wet docs={n_wet} english(unique url)={n_wet_en}", flush=True)

    lo, hi = (int(x) for x in args.c4_shards.split("-"))
    pairs = []
    n_c4 = 0
    for shard in range(lo, hi + 1):
        path = os.path.join(args.c4_dir, f"c4-train.{shard:05d}-of-01024.json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n_c4 += 1
                k = norm_url(r.get("url", ""))
                if k in wet:
                    pairs.append({"url": r.get("url", ""),
                                  "wet": wet[k], "c4_text": r.get("text", "")})
        print(f"c4 shard {shard:05d}: scanned={n_c4} matched={len(pairs)}", flush=True)

    # stable c4 ids: recompute with the true line index (done below in a second pass
    # only for matched lines is unnecessary; ids are not used as keys in the analysis)
    js, lw, lc = [], [], []
    out_sample = []
    for pr in pairs:
        a, b = pr["wet"]["text"], pr["c4_text"]
        j = jacc(a, b)
        js.append(j)
        lw.append(len(a.split()))
        lc.append(len(b.split()))
        if len(out_sample) < 300:
            out_sample.append({"url": pr["url"], "wet_id": pr["wet"]["id"],
                               "jaccard_5gram": round(j, 4), "nwords_wet": lw[-1],
                               "nwords_c4": lc[-1],
                               "wet_head": a[:400], "c4_head": b[:400]})
    js = np.array(js) if js else np.zeros(1)
    lw = np.array(lw) if lw else np.zeros(1)
    lc = np.array(lc) if lc else np.zeros(1)
    stats = {
        "wet_docs_total": n_wet,
        "wet_english_unique_urls": n_wet_en,
        "c4_docs_scanned": n_c4,
        "c4_shards": f"{lo}-{hi}",
        "matched_pairs": len(pairs),
        "matched_fraction_of_wet_en": len(pairs) / max(1, n_wet_en),
        "note": ("C4 shards scanned are 8 of 1024, so the matched fraction is a lower "
                 "bound on coverage; absence from these shards does not mean a document "
                 "was dropped by C4."),
        "jaccard_5gram": {"mean": float(js.mean()), "p10": float(np.percentile(js, 10)),
                           "p50": float(np.percentile(js, 50)),
                           "p90": float(np.percentile(js, 90)),
                           "frac_ge_0.8": float((js >= 0.8).mean()),
                           "frac_ge_0.5": float((js >= 0.5).mean())},
        "nwords_wet": {"mean": float(lw.mean()), "p50": float(np.median(lw))},
        "nwords_c4": {"mean": float(lc.mean()), "p50": float(np.median(lc))},
        "c4_shorter_frac": float((lc < lw).mean()) if len(pairs) else 0.0,
    }
    with open(os.path.join(args.out, "wet_c4_pairing.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "wet_c4_pairs_sample.jsonl"), "w") as f:
        for r in out_sample:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
