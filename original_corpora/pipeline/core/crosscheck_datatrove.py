#!/usr/bin/env python3
"""Cross-validate path A's pairing output against a real datatrove 0.6.0 run.

1,400 WET documents are written as 14 JSONL files (100 docs each; rank r reads file r),
so datatrove's (file_stem, doc_id) maps exactly to the global subset index r*100+d.
The word tokenizer is patched to the project's whitespace tokenizer so that the only
remaining variable is datatrove's stage-2 pairing / stage-3 clustering code.

Compares:
  - datatrove .dups pair set vs path A chain edges
  - datatrove stage-3 .remove set vs our union-by-size replay (and vs min-id)
Output: core/crosscheck_datatrove.json
"""
import glob
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K


class WhitespaceTokenizer:
    """Project tokenizer after datatrove's simplify_text: split on single spaces."""
    def word_tokenize(self, text):
        return text.split(" ")


def write_jsonl_inputs(docs, texts, folder, per_file=100):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    for r in range(len(docs) // per_file):
        with open(os.path.join(folder, f"part-{r:02d}.jsonl"), "w", encoding="utf-8") as f:
            for k in range(per_file):
                gi = r * per_file + k
                f.write(json.dumps({"id": f"d{gi}", "text": texts[gi]}) + "\n")


def read_dups(bucket_dir):
    pairs = []
    for p in sorted(glob.glob(os.path.join(bucket_dir, "*.dups"))):
        with open(p, "rb") as f:
            b = f.read()
        for off in range(0, len(b) - 15, 16):
            f1, d1, f2, d2 = struct.unpack("<4I", b[off:off + 16])
            if f1 != (1 << 32) - 1:
                pairs.append((f1 * 100 + d1, f2 * 100 + d2))
    return pairs


def read_remove(cluster_dir):
    rem = set()
    for p in sorted(glob.glob(os.path.join(cluster_dir, "*.remove"))):
        stem = int(os.path.basename(p).split(".")[0])
        b = open(p, "rb").read()
        for off in range(0, len(b) - 3, 4):
            (d,) = struct.unpack("<I", b[off:off + 4])
            rem.add(stem * 100 + d)
    return rem


def main():
    from datatrove.pipeline import dedup as DT
    from datatrove.pipeline.dedup import minhash as MH
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader

    # patch the tokenizer before any step loads spacy
    MH.load_word_tokenizer = lambda language: WhitespaceTokenizer()

    work = os.path.join(K.CORE, "crosscheck_datatrove")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    inp = os.path.join(work, "input")
    sig = os.path.join(work, "signatures")
    buc = os.path.join(work, "buckets")
    clu = os.path.join(work, "clusters")

    tau = 0.8
    off, flat, _ = K.load_shingles("wet")
    ti, tj, _ = K.compute_truth("wet", tau, off)
    root = K.uf_minid(len(off) - 1, ti, tj)
    size = np.bincount(root, minlength=len(off) - 1)
    order = np.argsort(-size, kind="stable")
    chosen, seen = [], set()
    for d in order:
        if size[d] < 2:
            break
        for c in np.flatnonzero(root == root[d]):
            c = int(c)
            if c not in seen:
                seen.add(c)
                chosen.append(c)
        if len(chosen) >= 1000:
            break
    rng = np.random.RandomState(20261119)
    while len(chosen) < 1400:
        c = int(rng.randint(len(off) - 1))
        if c not in seen:
            seen.add(c)
            chosen.append(c)
    docs = np.array(sorted(chosen), dtype=np.int64)
    import pyarrow.parquet as pq
    alltext = {}
    for p in sorted(glob.glob(os.path.join(K.POOL, "wet", "*.parquet"))):
        d = pq.read_table(p, columns=["id", "text"]).to_pydict()
        alltext.update(zip(d["id"], d["text"]))
    ids = np.asarray(K.load_ids("wet"))
    texts = [alltext[ids[d]] for d in docs]
    write_jsonl_inputs(docs, texts, inp)
    n = len(docs)
    print("datatrove crosscheck subset:", n, "docs")

    # our shingles for the same order
    so, sf_ = [], []
    for d in docs:
        so.append(flat[off[d]:off[d + 1]])
    lens = np.array([len(x) for x in so], np.int64)
    soff = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=soff[1:])
    sflat = np.concatenate(so)
    sig_arr = K.L.signatures(soff, sflat, K.H, 1, "dt")
    Kb = K.band_keys(sig_arr)
    empty = np.flatnonzero(lens == 0)
    valid = lens > 0  # datatrove never writes a signature for a doc with no n-grams
    ei, ej, astats = K.chain_edges(Kb, valid)
    a_set = {(int(a), int(b)) for a, b in zip(ei.tolist(), ej.tolist())}

    cfg = MH.MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8, seed=1)
    # datatrove 0.6.0 calls xxhash.xxh64_intdigest(str); xxhash>=4 requires bytes.
    # Patch to UTF-8 encode, which is exactly what vendor/common.py shingle_doc does.
    import xxhash as _xx
    hash_fn = lambda s: _xx.xxh64_intdigest(s.encode("utf-8"))
    s1 = DT.MinhashDedupSignature(sig, config=cfg)
    s1._hash_func = hash_fn
    s2 = DT.MinhashDedupBuckets(input_folder=sig, output_folder=buc, config=cfg)
    s2._hash_func = hash_fn
    # Stage 1 and stage 2 must be separate jobs: with workers=1, LocalPipelineExecutor
    # runs the whole pipeline per rank sequentially, so a co-scheduled stage 2 would see
    # only the sig files of ranks that already finished (measured: bucket 0 saw 1 of 14 files).
    t0 = time.time()
    ex1 = LocalPipelineExecutor(
        pipeline=[JsonlReader(inp, glob_pattern="*.jsonl"), s1],
        tasks=14, workers=1, logging_dir=os.path.join(work, "logs1"))
    ex1.run()
    t_stage1 = time.time() - t0
    t0 = time.time()
    ex2a = LocalPipelineExecutor(pipeline=[s2], tasks=14, workers=1,
                                 logging_dir=os.path.join(work, "logs1b"))
    ex2a.run()
    t_stage2 = time.time() - t0
    t_stage12 = t_stage1 + t_stage2
    dt_pairs = read_dups(buc)
    dt_set = {(min(a, b), max(a, b)) for a, b in dt_pairs}
    print("datatrove pairs", len(dt_pairs), "unique", len(dt_set), "our chain", len(a_set),
          "stage1+2", round(t_stage12, 1), "s")

    t0 = time.time()
    ex2 = LocalPipelineExecutor(
        pipeline=[DT.MinhashDedupCluster(input_folder=buc, output_folder=clu, config=cfg)],
        tasks=1, workers=1, logging_dir=os.path.join(work, "logs2"))
    ex2.run()
    t_stage34 = time.time() - t0
    dt_remove = read_remove(clu)
    # replay the official rule ourselves on datatrove's pair order (validates our UF)
    ord_pi = np.array([a for a, b in dt_pairs], np.int64)
    ord_pj = np.array([b for a, b in dt_pairs], np.int64)
    _ = K.uf_datatrove(n, ord_pi, ord_pj)
    root_dt = K.uf_datatrove(n, ord_pi, ord_pj)
    our_dt_remove = set(np.flatnonzero(root_dt != np.arange(n)).tolist())
    root_min = K.uf_minid(n, ord_pi, ord_pj)
    our_min_remove = set(np.flatnonzero(root_min != np.arange(n)).tolist())

    out = dict(
        evidence_tier="rerun", generated_utc=K.now(), datatrove_version="0.6.0",
        config=dict(n_grams=5, num_buckets=14, hashes_per_bucket=8, seed=1,
                    tokenizer="patched to project whitespace tokenizer"),
        subset_n=n, subset_shingles=int(soff[-1]),
        docs_without_5grams=int(len(empty)), docs_without_5grams_ids=empty.tolist(),
        stage1_seconds=round(t_stage1, 2), stage2_seconds=round(t_stage2, 2),
        stage12_seconds=round(t_stage12, 2), stage34_seconds=round(t_stage34, 2),
        datatrove_pairs=len(dt_pairs), datatrove_unique_pairs=len(dt_set),
        our_chain_edges=len(a_set),
        pair_sets_identical=bool(dt_set == a_set),
        pair_symmetric_difference=len(dt_set ^ a_set),
        datatrove_removed=len(dt_remove),
        our_union_by_size_equals_datatrove_removed=bool(dt_remove == our_dt_remove),
        union_by_size_symmetric_difference=len(dt_remove ^ our_dt_remove),
        min_id_removed=len(our_min_remove),
        min_id_vs_datatrove_symmetric_difference=len(our_min_remove ^ dt_remove),
        **{k: astats[k] for k in ("largest_bucket", "bucket_pairs_sum", "n_chain_pairs")},
    )
    out["all_checks_pass"] = bool(out["pair_sets_identical"] and
                                  out["our_union_by_size_equals_datatrove_removed"])
    p = os.path.join(K.CORE, "crosscheck_datatrove.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
