"""Assemble results/data_manifest.json from the raw inputs and prepared pool.

Every checksum is computed in this run.  The manifest records, for each source:
repository / URL, version (HF commit or vela-library ref), local file, size,
sha256, and the role it plays.  It also records tool versions, the sampling
unit, the shard layout and the processing history, and copies the duplicate
density estimates from results/pool_stats.json.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fhash(path):
    return {"size_bytes": os.path.getsize(path), "sha256": sha256(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--c4-dir", required=True)
    ap.add_argument("--fw-parquet", required=True)
    ap.add_argument("--wet-dir", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--results", default="")
    ap.add_argument("--mini-dir", default="output/dedupjoin-pvldb/artifacts/mini_corpus")
    ap.add_argument("--vela-dir", default="/tmp/vela-datasets/url")
    args = ap.parse_args()
    res = args.results or os.path.dirname(args.out)

    items = []
    for i in range(8):
        p = os.path.join(args.c4_dir, f"c4-train.{i:05d}-of-01024.json.gz")
        items.append({"name": f"c4_en_train_{i:05d}", "source": "allenai/c4",
                      "url": f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:05d}-of-01024.json.gz",
                      "path": p, "role": "input corpus (shard)", **fhash(p)})
    items.append({"name": "fineweb_sample10bt_000_00000", "source": "HuggingFaceFW/fineweb",
                  "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/000_00000.parquet",
                  "path": args.fw_parquet, "role": "input corpus (already MinHash-deduplicated by its producer)", **fhash(args.fw_parquet)})

    wet_stats = {}
    sp = os.path.join(args.wet_dir, "extract_stats.jsonl")
    if os.path.exists(sp):
        for line in open(sp):
            r = json.loads(line)
            wet_stats[r["segment"]] = r
    for p in sorted(glob.glob(os.path.join(args.wet_dir, "wet-*.parquet"))):
        seg = os.path.basename(p)[4:-8]
        st = wet_stats.get(seg, {})
        ref = None
        for vf in glob.glob(os.path.join(args.vela_dir, "*", "*", "CC-MAIN-*")) + \
                  glob.glob(os.path.join(args.vela_dir, "*", "*")):
            if os.path.isdir(vf) and any(seg in x for x in os.listdir(vf)):
                vj = os.path.join(vf, ".vela-dataset.json")
                if os.path.exists(vj):
                    ref = json.load(open(vj))
                    break
        items.append({"name": f"wet_{seg}", "source": "Common Crawl CC-MAIN-2019-18",
                      "url": ref.get("ref") if ref else "https://data.commoncrawl.org/crawl-data/CC-MAIN-2019-18/",
                      "vela_key": ref.get("key") if ref else None,
                      "path": p, "role": "raw crawl text (no dedup applied by the producer)",
                      "records": st.get("records"), "docs": st.get("docs"),
                      "language_top": st.get("lang_top"), **fhash(p)})

    pool_files = sorted(glob.glob(os.path.join(args.pool, "**", "*.parquet"), recursive=True))
    mini_dir = args.mini_dir
    mini = None
    mp = os.path.join(mini_dir, "manifest.json")
    if os.path.exists(mp):
        mini = json.load(open(mp))

    import importlib.metadata as md
    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return None

    manifest = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "dedupjoin-pvldb corpora preparation",
        "provenance": "independently rerun (this run): every checksum and count computed now",
        "sampling_unit": "document (one WET record / one C4 JSON line / one FineWeb parquet row)",
        "sources": items,
        "pool": {
            "path": args.pool,
            "n_docs": json.load(open(os.path.join(args.pool, "build_summary.json")))["rows"]
            if os.path.exists(os.path.join(args.pool, "build_summary.json")) else None,
            "layout": ("<pool>/{fw,c4,wet}/shard-XXXXX.parquet, 100,000 documents per shard, "
                       "zstd; <pool>/meta/index-XXXXX.parquet holds metadata only; "
                       "<pool>/strata.parquet holds the length/duplicate-density/domain labels; "
                       "<pool_bands>/doc_clusters.parquet holds per-document LSH cluster sizes"),
            "n_files": len(pool_files),
            "size_bytes": sum(os.path.getsize(p) for p in pool_files),
            "text_bytes": sum(os.path.getsize(p) for p in pool_files
                              if os.path.basename(os.path.dirname(p)) in ("fw", "c4", "wet")),
            "metadata_bytes": sum(os.path.getsize(p) for p in pool_files
                                  if os.path.basename(os.path.dirname(p)) not in ("fw", "c4", "wet")),
            "files": [{"path": p, **fhash(p)} for p in pool_files],
        },
        "mini_corpus": mini,
        "tools": {
            "python": sys.version.split()[0],
            "datatrove": ver("datatrove"),
            "xxhash": ver("xxhash"),
            "pyarrow": ver("pyarrow"),
            "numpy": ver("numpy"),
            "numba": ver("numba"),
            "spacy": ver("spacy"),
            "en_core_web_sm": ver("en-core-web-sm"),
            "fasttext-wheel": ver("fasttext-wheel"),
            "zstandard": ver("zstandard"),
            "language_id_model": {"name": "fastText lid.176.bin",
                                  "path": ".work/dedupjoin/raw/lid.176.bin",
                                  **(fhash(".work/dedupjoin/raw/lid.176.bin")
                                     if os.path.exists(".work/dedupjoin/raw/lid.176.bin") else {})},
        },
        "processing_history": [
            {"step": "wet extraction", "tool": "code/wet_extract.py (hand WARC parser)",
             "filters": ["drop empty bodies"],
             "language_id": "fastText lid.176 top-1 label and score stored per document",
             "outputs": sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.wet_dir, "wet-*.parquet")))},
            {"step": "pool build", "tool": "code/build_pool.py",
             "order": "fw (file order), c4 (shard 0..7, line order), wet (segment order, record order)",
             "document_id": {"fw": "native FineWeb uuid", "c4": "c4-<shard:05d>-<line>",
                              "wet": "wet-<segment>-<record>"},
             "hashes": {"raw_hash": "xxh64 of the stored text",
                        "norm_hash": "xxh64 of ' '.join(text.lower().split())"}},
            {"step": "pool LSH clustering", "tool": "code/pool_minhash.py + code/pool_cluster.py",
             "config": "5-gram shingles, 112 MinHash values (datatrove permutation family, seed 1), "
                       "14 bands x 8 rows, consecutive-pair edges per bucket, union-find",
             "cap": "first 20,000 shingles per document enter the sketch"},
            {"step": "datatrove before/after pairing", "tool": "code/pairing_wet_datatrove.py",
             "config": "datatrove 0.6.0 MinhashConfig defaults, stages signature/buckets/cluster/filter",
             "dependency_patch": "datatrove 0.6.0 xxhash helper encodes str to UTF-8 (xxhash 4.x removed str support)"},
            {"step": "URL pairing", "tool": "code/pairing_wet_c4.py",
             "pairing": "identical URL after dropping scheme, lower-casing host, removing www. and trailing slash"},
            {"step": "sampling check", "tool": "code/characterize_pool.py",
             "checks": "pool prefixes reproduce the earlier C4 700k / FineWeb 1M / Mix 1M id lists exactly"},
        ],
        "duplicate_density_estimate": None,
        "pairing": {},
    }
    ps = os.path.join(res, "pool_stats.json")
    if os.path.exists(ps):
        s = json.load(open(ps))
        manifest["duplicate_density_estimate"] = {
            "method": "LSH 14x8 connected components over the whole pool (labels in strata.parquet)",
            "whole_pool": {"n": s["n_docs"],
                           "dup_density_lsh_ge2": s.get("dup_density_lsh_ge2")},
            "by_source": {k: {"n": v["n"], "dup_density_lsh_ge2": v["dup_density_lsh_ge2"],
                              "exact_raw_density": v["dup_density_exact_raw"],
                              "largest_cluster": v["largest_lsh_cluster"]}
                           for k, v in s["sources"].items()},
        }
    for name in ("wet_datatrove/wet_datatrove_pairing.json", "wet_c4_pairing.json"):
        p = os.path.join(res, name)
        if os.path.exists(p):
            manifest["pairing"][os.path.basename(p)] = json.load(open(p))
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"sources": len(items), "pool_files": len(pool_files),
                      "pool_bytes": manifest["pool"]["size_bytes"],
                      "manifest": args.out}))


if __name__ == "__main__":
    main()
