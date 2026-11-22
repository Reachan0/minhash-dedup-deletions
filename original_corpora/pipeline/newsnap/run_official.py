#!/usr/bin/env python3
"""Official datatrove 0.6.0 MinHash pipeline on the snapshot (PREREG path OFF).

signature -> buckets -> cluster -> filter, 5-gram / 112 hashes / 14x8 / seed 1,
native tokenizer load_word_tokenizer("en") (the same tokenizer that built the
snapshot shingles).  Stage 1 and stage 2 run as separate executor jobs (verified
requirement: a co-scheduled stage 2 sees only finished ranks).

Writes newsnap/official/result.json and appends rows for the OFF path at seed 1.
"""
import glob
import importlib.metadata  # noqa: F401
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402

WORK = os.path.join(S.SNAP, "official")
INP = os.path.join(WORK, "input")
NFILES = 12


def write_inputs():
    shutil.rmtree(INP, ignore_errors=True)
    os.makedirs(INP, exist_ok=True)
    ids_all = []
    for ids, segs, urls, texts, nw, nch in S.iter_text_batches(20000):
        ids_all.extend(ids)
    n = len(ids_all)
    bounds = np.linspace(0, n, NFILES + 1).astype(np.int64)
    offsets = [int(b) for b in bounds]
    # second pass: stream texts and write per-rank files
    r = 0
    fh = open(os.path.join(INP, f"part-{r:02d}.jsonl"), "w", encoding="utf-8")
    k = 0
    for ids, segs, urls, texts, nw, nch in S.iter_text_batches(4000):
        for i, (id_, t) in enumerate(zip(ids, texts)):
            while k >= offsets[r + 1]:
                fh.close()
                r += 1
                fh = open(os.path.join(INP, f"part-{r:02d}.jsonl"), "w", encoding="utf-8")
            fh.write(json.dumps({"id": id_, "text": t}) + "\n")
            k += 1
    fh.close()
    json.dump({"n": n, "offsets": offsets, "files": NFILES}, open(os.path.join(WORK, "input_manifest.json"), "w"))
    return n, offsets


def read_dups(bucket_dir, offsets):
    pairs = []
    for p in sorted(glob.glob(os.path.join(bucket_dir, "*.dups"))):
        b = open(p, "rb").read()
        for off in range(0, len(b) - 15, 16):
            f1, d1, f2, d2 = struct.unpack("<4I", b[off:off + 16])
            if f1 != (1 << 32) - 1:
                pairs.append((offsets[f1] + d1, offsets[f2] + d2))
    return pairs


def read_remove(cluster_dir, offsets):
    rem = set()
    for p in sorted(glob.glob(os.path.join(cluster_dir, "*.remove"))):
        r = int(os.path.basename(p).split(".")[0])
        b = open(p, "rb").read()
        for off in range(0, len(b) - 3, 4):
            (d,) = struct.unpack("<I", b[off:off + 4])
            rem.add(offsets[r] + d)
    return rem


def main():
    from datatrove.pipeline import dedup as DT
    from datatrove.pipeline.dedup import minhash as MH
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader
    import xxhash as _xx

    cfg = MH.MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8, seed=1)
    hash_fn = lambda s: _xx.xxh64_intdigest(s.encode("utf-8"))
    os.makedirs(WORK, exist_ok=True)
    n, offsets = write_inputs()
    print("official input:", n, "docs,", NFILES, "files", flush=True)
    sig = os.path.join(WORK, "signatures")
    buc = os.path.join(WORK, "buckets")
    clu = os.path.join(WORK, "clusters")
    out = os.path.join(WORK, "kept")
    for d in (sig, buc, clu, out):
        shutil.rmtree(d, ignore_errors=True)

    t0 = time.time()
    s1 = DT.MinhashDedupSignature(sig, config=cfg)
    s1._hash_func = hash_fn
    LocalPipelineExecutor(pipeline=[JsonlReader(INP, glob_pattern="*.jsonl"), s1],
                          tasks=NFILES, workers=1,
                          logging_dir=os.path.join(WORK, "logs1")).run()
    t_s1 = time.time() - t0
    print(f"stage1 signature {t_s1:.0f}s", flush=True)

    t0 = time.time()
    s2 = DT.MinhashDedupBuckets(input_folder=sig, output_folder=buc, config=cfg)
    s2._hash_func = hash_fn
    LocalPipelineExecutor(pipeline=[s2], tasks=14, workers=1,
                          logging_dir=os.path.join(WORK, "logs2")).run()
    t_s2 = time.time() - t0
    print(f"stage2 buckets {t_s2:.0f}s", flush=True)

    t0 = time.time()
    LocalPipelineExecutor(
        pipeline=[DT.MinhashDedupCluster(input_folder=buc, output_folder=clu, config=cfg)],
        tasks=1, workers=1, logging_dir=os.path.join(WORK, "logs3")).run()
    t_s3 = time.time() - t0
    print(f"stage3 cluster {t_s3:.0f}s", flush=True)

    t0 = time.time()
    LocalPipelineExecutor(
        pipeline=[JsonlReader(INP, glob_pattern="*.jsonl"),
                  DT.MinhashDedupFilter(input_folder=clu),
                  DT.JsonlWriter(out) if hasattr(DT, "JsonlWriter") else _writer(out)],
        tasks=NFILES, workers=1, logging_dir=os.path.join(WORK, "logs4")).run()
    t_s4 = time.time() - t0
    print(f"stage4 filter {t_s4:.0f}s", flush=True)

    pairs = read_dups(buc, offsets)
    rem = read_remove(clu, offsets)
    kept_ids = set()
    for p in glob.glob(os.path.join(out, "*.jsonl")):
        for line in open(p, encoding="utf-8"):
            try:
                kept_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    off_all = S.load_snapshot()[0]
    from pyarrow import parquet as pq
    meta = pq.read_table(S.META_PARQUET, columns=["id"]).column("id").to_pylist()
    kept_idx = {i for i, x in enumerate(meta) if x in kept_ids}
    removed_by_filter = set(range(len(meta))) - kept_idx

    # compare with project A (same token stream)
    runs = [json.loads(l) for l in open(S.RUNS)] if os.path.exists(S.RUNS) else []
    a_rem = None
    mk = os.path.join(S.CACHE, "masks", "raw_s1.npz")
    if os.path.exists(mk):
        a_rem = np.flatnonzero(np.load(mk)["rem"])

    res = dict(evidence_tier="rerun", generated_utc=S.now(), datatrove_version="0.6.0",
               config=dict(n_grams=5, num_buckets=14, hashes_per_bucket=8, seed=1,
                           tokenizer="native load_word_tokenizer('en') (spacy.blank('en'))"),
               n=n, stage1_s=round(t_s1, 1), stage2_s=round(t_s2, 1),
               stage3_s=round(t_s3, 1), stage4_s=round(t_s4, 1),
               datatrove_pairs=len(pairs), datatrove_removed=len(rem),
               filter_removed=len(removed_by_filter),
               remove_equals_filter_removal=bool(rem == removed_by_filter),
               kept=len(kept_ids))
    if a_rem is not None:
        A = set(a_rem.tolist())
        res["project_A_removed"] = len(A)
        res["symdiff_vs_project_A"] = len(A ^ rem)
        res["jaccard_vs_project_A"] = len(A & rem) / len(A | rem)
        # union-by-size replay on the official pair order
        ord_pi = np.array([a for a, b in pairs], np.int64)
        ord_pj = np.array([b for a, b in pairs], np.int64)
        rt = S.K.uf_datatrove(n, ord_pi, ord_pj)
        replay = set(np.flatnonzero(rt != np.arange(n)).tolist())
        res["union_by_size_replay_equals_official"] = bool(replay == rem)
        res["union_by_size_replay_symdiff"] = len(replay ^ rem)
        rt2 = S.K.uf_minid(n, ord_pi, ord_pj)
        res["minid_on_official_pairs_removed"] = int((rt2 != np.arange(n)).sum())
        res["minid_vs_official_symdiff"] = int(((rt2 != np.arange(n)) ^
                                               np.isin(np.arange(n), list(rem))).sum())
    json.dump(res, open(os.path.join(WORK, "result.json"), "w"), indent=1)
    S.append_row(os.path.join(S.SNAP, "official.jsonl"),
                 dict(key="off|s1", **{k: v for k, v in res.items() if k != "evidence_tier"},
                      evidence_tier="rerun"))
    print(json.dumps(res, indent=1), flush=True)


def _writer(out):
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    return JsonlWriter(out)


if __name__ == "__main__":
    main()
