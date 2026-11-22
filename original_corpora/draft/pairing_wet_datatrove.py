"""Paired before/after dataset from the raw crawl: run the official datatrove
MinHash dedup pipeline (v0.6.0, 5-grams, 14 bands x 8 hashes, seed 1) on the
English subset of the CC-MAIN-2019-18 WET segments.

before = every English document that enters the pipeline (id, text)
after  = the documents the datatrove filter keeps (one per connected component)
pairing = same document id on both sides; `removed` is the exact complement.

Usage
  python3 pairing_wet_datatrove.py --wet-dir ... --work ... --out ...
        [--lang-score 0.99] [--tasks 4] [--limit N]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import pyarrow.parquet as pq


def build_input(wet_dir, in_dir, lang_score, limit=0):
    os.makedirs(in_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(wet_dir, "wet-*.parquet")))
    info = []
    for p in files:
        seg = os.path.basename(p)[4:-8]
        out = os.path.join(in_dir, f"wet-{seg}.jsonl")
        n_in = n_en = 0
        with open(out, "w") as f:
            for bi in pq.ParquetFile(p).iter_batches(batch_size=5000,
                                                     columns=["id", "text", "lang", "lang_score", "url"]):
                d = bi.to_pydict()
                for i in range(len(d["id"])):
                    n_in += 1
                    if d["lang"][i] != "en" or d["lang_score"][i] < lang_score:
                        continue
                    f.write(json.dumps({"id": d["id"][i], "text": d["text"][i],
                                        "url": d["url"][i]}) + "\n")
                    n_en += 1
                    if limit and n_en >= limit:
                        break
                if limit and n_en >= limit:
                    break
        info.append({"segment": seg, "records": n_in, "english": n_en, "input": out})
        print(json.dumps(info[-1]), flush=True)
    return info


def _patch_xxhash():
    """xxhash 4.x rejects str where 3.x accepted it; encode UTF-8 first.
    datatrove 0.6.0 was released against xxhash 3.x."""
    import datatrove.utils.hashes.xxhash as dtx
    for name in ("xxhash32", "xxhash64"):
        orig = getattr(dtx, name)

        def wrapper(data, _orig=orig):
            if isinstance(data, str):
                data = data.encode("utf-8")
            return _orig(data)

        setattr(dtx, name, wrapper)


def run_datatrove(in_dir, work, tasks, workers):
    _patch_xxhash()
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import (MinhashDedupSignature, MinhashDedupBuckets,
                                          MinhashDedupCluster, MinhashDedupFilter)
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter

    cfg = MinhashConfig()   # 5-grams, 14 buckets x 8 hashes, seed 1, xxhash, 64-bit
    sig = os.path.join(work, "sigs")
    buc = os.path.join(work, "buckets")
    clu = os.path.join(work, "clusters")
    kept = os.path.join(work, "kept")
    removed = os.path.join(work, "removed")

    def run(pipeline, n_tasks, tag):
        LocalPipelineExecutor(pipeline=pipeline, tasks=n_tasks, workers=workers,
                              logging_dir=os.path.join(work, "logs", tag)).run()

    # stage 1: signatures
    run([JsonlReader(in_dir, glob_pattern="*.jsonl"),
         MinhashDedupSignature(output_folder=sig, config=cfg)], tasks, "s1")
    # stage 2: bucket pairs
    run([MinhashDedupBuckets(input_folder=sig, output_folder=buc, config=cfg)],
        tasks, "s2")
    # datatrove skips empty .dups files; stage 3 asserts one file per bucket
    os.makedirs(buc, exist_ok=True)
    for b in range(cfg.num_buckets):
        for w in range(max(1, tasks // cfg.num_buckets)):
            p = os.path.join(buc, f"{b:05d}_{w:02d}.dups")
            if not os.path.exists(p):
                open(p, "wb").close()
    # stage 3: clustering (single task by datatrove design)
    run([MinhashDedupCluster(input_folder=buc, output_folder=clu, config=cfg,
                             save_cluster_size=True)], 1, "s3")
    # stage 4: filter, writing the kept corpus and the removed documents
    run([JsonlReader(in_dir, glob_pattern="*.jsonl"),
         MinhashDedupFilter(input_folder=clu, exclusion_writer=JsonlWriter(
             removed, output_filename="${rank}.jsonl", compression=None)),
         JsonlWriter(kept, output_filename="${rank}.jsonl", compression=None)],
        tasks, "s4")
    return {"sig": sig, "buckets": buc, "clusters": clu, "kept": kept, "removed": removed}


def summarize(work, out_dir, in_info):
    os.makedirs(out_dir, exist_ok=True)
    ids_in = set()
    for f in glob.glob(os.path.join(work, "input", "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                ids_in.add(json.loads(line)["id"])
    def ids_of(pattern):
        s = set()
        for f in glob.glob(pattern):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    s.add(r.get("id"))
        return s
    kept = ids_of(os.path.join(work, "kept", "*.jsonl"))
    removed = ids_of(os.path.join(work, "removed", "*.jsonl"))
    stats = {
        "before_docs": len(ids_in),
        "after_docs": len(kept),
        "removed_docs": len(removed),
        "removed_fraction": len(removed) / max(1, len(ids_in)),
        "partition_complete": len(kept | removed) == len(ids_in),
        "kept_and_removed_disjoint": len(kept & removed) == 0,
        "missing_ids": len(ids_in - (kept | removed)),
        "datatrove_version": __import__("importlib.metadata", fromlist=["version"]).version("datatrove"),
        "inputs": in_info,
    }
    with open(os.path.join(out_dir, "wet_datatrove_pairing.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wet-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang-score", type=float, default=0.8)
    ap.add_argument("--tasks", type=int, default=14)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if os.path.isdir(args.work):
        shutil.rmtree(args.work)
    in_dir = os.path.join(args.work, "input")
    t0 = time.time()
    info = build_input(args.wet_dir, in_dir, args.lang_score, args.limit)
    paths = run_datatrove(in_dir, args.work, args.tasks, args.workers)
    stats = summarize(args.work, args.out, info)
    stats["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "wet_datatrove_pairing.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
