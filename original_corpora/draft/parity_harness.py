#!/usr/bin/env python3
"""Layer-by-layer parity check against the official datatrove 0.6.0 MinHash dedup.

The *official* code is executed end to end on a small public corpus
(FineWeb sample-10BT + synthetic near-duplicates), stage 1 -> 4, by calling the
pipeline steps directly on a list of Document objects (no executor needed):

  1. MinhashDedupSignature  -> per-doc tokens, shingles, signatures, bucket files
  2. MinhashDedupBuckets    -> (one call per bucket worker, world_size = 14) .dups pairs
  3. MinhashDedupCluster    -> .remove files (union by size)
  4. MinhashDedupFilter     -> kept document ids

Our C03 implementation is then compared layer by layer on the same documents:
tokenization, shingle sets, 112-hash signatures, candidate edges, connected
components, kept ids.  The shared-signature-set test checks that configurations
with different 14x8-style bandings (112, 117, 120, 130 hashes) can share one
signature computation.

Notes:
  * datatrove 0.6.0's xxhash wrapper calls xxhash.xxh64_intdigest(str), which the
    installed python-xxhash >= 3 rejects (API requires bytes).  We wrap it with
    str.encode("utf-8"); the hash value is unchanged.
  * datatrove's representative rule is union-by-size; ours keeps the minimum doc
    id.  The *deletion set* is identical for the same edges; kept ids differ
    inside a component.  We therefore compare both partitions and, for kept ids,
    apply datatrove's rule to both edge lists.

Writes results/parity.json (+ raw per-layer diffs).
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import struct
import sys
import time

import numpy as np

import importlib.metadata  # noqa: F401  (datatrove 0.6.0 imports importlib.metadata lazily)
import xxhash

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/parity")

from datatrove.data import Document  # noqa: E402
from datatrove.pipeline.dedup.minhash import (  # noqa: E402
    MinhashConfig, MinhashDedupSignature, MinhashDedupBuckets,
    MinhashDedupCluster, MinhashDedupFilter,
)
from datatrove.utils.text import simplify_text as dt_simplify_text  # noqa: E402


def hash_bytes(s):
    return xxhash.xxh64_intdigest(s.encode("utf-8"))


# ---------------------------------------------------------------- union find
def uf_dt(n, edges):
    """datatrove stage-3 union-by-size, run over an explicit edge list.

    Returns root array; roots are the union-by-size representatives.
    """
    par = {}
    sz = {}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in edges:
        for v in (a, b):
            if v not in par:
                par[v] = v
                sz[v] = 1
        ra, rb = find(a), find(b)
        if ra != rb:
            if sz[ra] < sz[rb]:
                ra, rb = rb, ra
            par[rb] = ra
            sz[ra] += sz[rb]
            del sz[rb]
    root = np.arange(n)
    for v in list(par):
        root[v] = find(v)
    return root


def partition(root):
    comp = {}
    for d in np.nonzero(root != np.arange(len(root)))[0]:
        comp.setdefault(int(root[d]), []).append(int(d))
    return comp


# ---------------------------------------------------------------- corpus
def build_corpus(n, n_perturb=200, seed=0):
    texts, meta = C.load_corpus("fw", n)
    rnd = random.Random(seed)
    extra, origin = [], []
    n_perturb = min(n_perturb, len(texts))
    for k, idx in enumerate(rnd.sample(range(len(texts)), n_perturb)):
        w = texts[idx].split()
        p = [0.0, 0.0, 0.02, 0.05, 0.1][k % 5]
        if p == 0.0:
            extra.append(texts[idx])                    # exact duplicate
        else:
            extra.append(" ".join(x for x in w if rnd.random() >= p))
        origin.append(idx)
    return texts + extra, origin


# ---------------------------------------------------------------- layers
def compare_tokens(dt_simplified, dt_tokens, our_simplified, our_tokens, n_examples=5):
    n = len(dt_tokens)
    simp_diff, tok_diff = [], []
    for i in range(n):
        if dt_simplified[i] != our_simplified[i]:
            simp_diff.append(i)
        if dt_tokens[i] != our_tokens[i]:
            tok_diff.append(i)
    ex = []
    for i in tok_diff[:n_examples]:
        a, b = dt_tokens[i], our_tokens[i]
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        ex.append(dict(doc=i, pos=k, dt_len=len(a), our_len=len(b),
                       dt_ctx=a[max(0, k - 2):k + 3], our_ctx=b[max(0, k - 2):k + 3]))
    return dict(n_docs=n,
                simplified_text_differ=len(simp_diff),
                token_list_differ=len(tok_diff),
                token_list_equal=n - len(tok_diff),
                token_count_dt_total=sum(len(t) for t in dt_tokens),
                token_count_ours_total=sum(len(t) for t in our_tokens),
                examples=ex,
                differ_doc_ids=tok_diff[:500])


def compare_shingles(dt_arrs, our_sets, tokens_equal):
    n = len(dt_arrs)
    diff, same_tok_but_diff = [], []
    ngram_counts = 0
    for i in range(n):
        a = np.unique(dt_arrs[i])
        b = our_sets[i]
        ngram_counts += len(dt_arrs[i])
        if len(a) != len(b) or not np.array_equal(a, b):
            diff.append(i)
            if tokens_equal[i]:
                same_tok_but_diff.append(i)
    return dict(n_docs=n, docs_with_different_shingle_set=len(diff),
                docs_same_tokens_different_shingles=len(same_tok_but_diff),
                shingles_identical=n - len(diff),
                dt_raw_ngram_total=int(ngram_counts),
                diff_doc_ids=diff[:500],
                same_token_diff_doc_ids=same_tok_but_diff[:20])


def compare_signatures(dt_sigs, our_sig, valid):
    n = len(dt_sigs)
    equal, diff = 0, []
    for i in range(n):
        if dt_sigs[i] is None:      # empty shingle set: no signature written
            continue
        a = np.asarray(dt_sigs[i], np.uint64).reshape(-1)
        ok = bool(np.array_equal(a, our_sig[i]))
        equal += ok
        if not ok:
            diff.append(i)
    return dict(n_with_signature=equal + len(diff), identical=equal,
                differ=len(diff), differ_doc_ids=diff[:50])


def read_dups(d):
    raw = 0
    pairs = set()
    files = sorted(f for f in os.listdir(d) if f.endswith(".dups"))
    for f in files:
        with open(os.path.join(d, f), "rb") as fh:
            b = fh.read()
        for k in range(0, len(b), 16):
            f1, d1, f2, d2 = struct.unpack("<4I", b[k:k + 16])
            raw += 1
            a, c = d1, d2
            pairs.add((min(a, c), max(a, c)))
    return raw, pairs, files


def compare_edges(dt_pairs, dt_raw, ci, cj):
    ours = set(zip(np.minimum(ci, cj).tolist(), np.maximum(ci, cj).tolist()))
    only_dt = sorted(dt_pairs - ours)
    only_ours = sorted(ours - dt_pairs)
    inter = len(dt_pairs & ours)
    return dict(dt_raw_emissions=dt_raw, dt_unique_pairs=len(dt_pairs),
                ours_chain_edges_raw=int(len(ci)), ours_unique_pairs=len(ours),
                intersection=inter,
                only_in_dt=len(only_dt), only_in_ours=len(only_ours),
                identical=(not only_dt and not only_ours),
                only_in_dt_sample=only_dt[:20], only_in_ours_sample=only_ours[:20])


def read_removes(d, n):
    rem = np.zeros(n, bool)
    p = os.path.join(d, "000000.remove")
    if os.path.exists(p):
        with open(p, "rb") as f:
            b = f.read()
        for k in range(0, len(b), 4):
            rem[struct.unpack("<I", b[k:k + 4])[0]] = True
    return rem


def compare_kept(dt_rem, dt_pairs, ci, cj, n):
    dt_kept = set(np.nonzero(~dt_rem)[0].tolist())
    # 1) our union-by-size mirror on datatrove's own edges must reproduce dt removes
    r_dt = uf_dt(n, dt_pairs)
    mirror_rem = r_dt != np.arange(n)
    mirror_ok = bool(np.array_equal(mirror_rem, dt_rem))
    # 2) our edges -> datatrove rule -> kept set
    r_ours = uf_dt(n, zip(np.minimum(ci, cj).tolist(), np.maximum(ci, cj).tolist()))
    ours_kept = set(np.nonzero(r_ours == np.arange(n))[0].tolist())
    # 3) our implementation's min-id rule
    r_min = L.uf_components(n, ci, cj)
    ours_kept_min = set(np.nonzero(r_min == np.arange(n))[0].tolist())
    return dict(n=n, dt_removed=int(dt_rem.sum()), dt_kept=len(dt_kept),
                mirror_of_dt_edges_reproduces_removes=mirror_ok,
                dt_rule_on_our_edges_kept=len(ours_kept),
                dt_rule_kept_identical=(ours_kept == dt_kept),
                dt_rule_kept_symdiff=len(ours_kept ^ dt_kept),
                minid_rule_kept=len(ours_kept_min),
                minid_vs_dt_kept_symdiff=len(ours_kept_min ^ dt_kept),
                components_size_gt1=len(partition(r_dt)),
                component_sizes_dt=sorted((len(v) + 1 for v in partition(r_dt).values()), reverse=True)[:10])


def shared_signature_test(off, flat, sig112):
    """Configurations with >=112 hashes share one signature computation."""
    res = {}
    for b, r in [(14, 8), (13, 9), (10, 13), (20, 6), (15, 8)]:
        H = b * r
        gen = np.random.RandomState(1)
        a = gen.randint(1, L.P61, dtype=np.uint64, size=(1, H))[0]
        bb = gen.randint(0, L.P61, dtype=np.uint64, size=(1, H))[0]
        gen112 = np.random.RandomState(1)
        a112 = gen112.randint(1, L.P61, dtype=np.uint64, size=(1, 112))[0]
        b112 = gen112.randint(0, L.P61, dtype=np.uint64, size=(1, 112))[0]
        prefix_np = bool(np.array_equal(a[:112], a112) and np.array_equal(bb[:112], b112))
        our_h = L.signatures(off, flat, H, 1, "dt") if H != 112 else sig112
        prefix_sig = bool(np.array_equal(our_h[:, :112], sig112))
        res[f"{b}x{r}"] = dict(H=H, randomstate_prefix_equal=prefix_np,
                               signature_prefix_equal=prefix_sig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="FineWeb docs before synthetic near-dupes")
    ap.add_argument("--perturb", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    shutil.rmtree(WRK, ignore_errors=True)
    os.makedirs(WRK, exist_ok=True)
    C.set_threads(max(1, a.workers))

    texts, origin = build_corpus(a.n, a.perturb)
    n = len(texts)
    docs = [Document(text=t, id=str(i)) for i, t in enumerate(texts)]

    cfg = MinhashConfig()  # 5-grams, 14 buckets x 8 hashes, seed 1, 64-bit xxhash
    s1dir, s2dir, s3dir = (os.path.join(WRK, x) for x in ("s1", "s2", "s3"))

    # ---- official datatrove stage 1 (recording tokens/shingles/signatures) ----
    st1 = MinhashDedupSignature(s1dir, config=cfg)
    st1._hash_func = hash_bytes          # see module docstring (xxhash >= 3 API)
    rec_tokens, rec_simplified, rec_shingles, rec_sigs = [], [], [], []
    wt_orig = st1.word_tokenizer.word_tokenize

    def wt_rec(text):
        toks = wt_orig(text)
        rec_simplified.append(text)
        rec_tokens.append(toks)
        return toks

    st1.word_tokenizer.word_tokenize = wt_rec
    gs_orig, gsg_orig = st1.get_shingles, st1.get_signature

    def gs_rec(text):
        arr = gs_orig(text)
        rec_shingles.append(arr.reshape(-1))
        return arr

    def gsg_rec(sh):
        sig = gsg_orig(sh)
        rec_sigs.append(sig)
        return sig

    st1.get_shingles = gs_rec
    st1.get_signature = gsg_rec
    t = time.time()
    st1.run(iter(docs), rank=0, world_size=1)
    t_stage1 = time.time() - t
    assert len(rec_tokens) == n and len(rec_shingles) == n
    # docs with no shingles get no signature record: align by walking the shingle list
    dt_sigs = []
    k = 0
    for i in range(n):
        if rec_shingles[i].size == 0:
            dt_sigs.append(None)
        else:
            dt_sigs.append(rec_sigs[k])
            k += 1
    assert k == len(rec_sigs)

    st2 = MinhashDedupBuckets(s1dir, s2dir, config=cfg)
    t = time.time()
    for r in range(cfg.num_buckets):
        st2.run(rank=r, world_size=cfg.num_buckets)
    t_stage2 = time.time() - t
    st3 = MinhashDedupCluster(s2dir, s3dir, config=cfg)
    t = time.time()
    st3.run(world_size=1)
    t_stage3 = time.time() - t
    st4 = MinhashDedupFilter(s3dir)
    kept_ids = [int(d.id) for d in st4.run(iter(docs), rank=0, world_size=1)]

    # ---- our implementation on the same documents ----
    t = time.time()
    off, flat, nw = C.build_shingles(texts, a.workers)
    t_shingle = time.time() - t
    our_simplified, our_tokens = [], []
    for txt in texts:
        s = C.simplify_text(txt)
        our_simplified.append(s)
        our_tokens.append(s.split(" ") if s else [])
    our_sets = [flat[off[i]:off[i + 1]] for i in range(n)]
    valid = np.diff(off) > 0

    t = time.time()
    sig112 = L.signatures(off, flat, 112, 1, "dt")
    t_sig = time.time() - t
    K = L.band_keys(sig112, 14, 8)
    codes, ci, cj, _ = L.lsh_pairs(K, valid, 100_000_000)

    # ---- compare ----
    layer1 = compare_tokens(rec_simplified, rec_tokens, our_simplified, our_tokens)
    tok_equal = [rec_tokens[i] == our_tokens[i] for i in range(n)]
    layer2 = compare_shingles(rec_shingles, our_sets, tok_equal)
    layer3 = compare_signatures(dt_sigs, sig112, valid)

    dt_raw, dt_pairs, dup_files = read_dups(s2dir)
    layer4 = compare_edges(dt_pairs, dt_raw, ci, cj)
    dt_rem = read_removes(s3dir, n)
    layer5 = compare_kept(dt_rem, dt_pairs, ci, cj, n)
    layer6 = dict(dt_kept=len(kept_ids), dt_removed_vectors=int(dt_rem.sum()),
                  filter_matches_remove_file=bool(set(kept_ids) == set(np.nonzero(~dt_rem)[0].tolist())),
                  kept_symdiff_vs_our_pipeline=len(set(kept_ids) ^ set(np.nonzero(~L.removed_set(L.uf_components(n, ci, cj)))[0].tolist())))

    # ---- conditional parity: our shingle/signature/banding/union code fed with the
    # exact token stream datatrove produced.  If this is identical, the only difference
    # between the two pipelines is the word tokenizer (layer 1). ----
    dt_sets = []
    for toks in rec_tokens:
        if len(toks) < 5:
            dt_sets.append(np.zeros(0, np.uint64))
        else:
            dt_sets.append(np.unique(np.fromiter(
                (hash_bytes(" ".join(toks[i:i + 5])) for i in range(len(toks) - 4)),
                dtype=np.uint64, count=len(toks) - 4)))
    sizes2 = np.array([len(s) for s in dt_sets], np.int64)
    off2 = np.zeros(n + 1, np.int64)
    np.cumsum(sizes2, out=off2[1:])
    flat2 = np.concatenate(dt_sets) if n else np.zeros(0, np.uint64)
    valid2 = sizes2 > 0
    sig2 = L.signatures(off2, flat2, 112, 1, "dt")
    K2 = L.band_keys(sig2, 14, 8)
    codes2, ci2, cj2, _ = L.lsh_pairs(K2, valid2, 100_000_000)
    layer3c = compare_signatures(dt_sigs, sig2, valid2)
    layer4c = compare_edges(dt_pairs, dt_raw, ci2, cj2)
    layer5c = compare_kept(dt_rem, dt_pairs, ci2, cj2, n)
    layer5c["kept_symdiff_vs_dt"] = layer5c["dt_rule_kept_symdiff"]
    layer5c["identical"] = layer4c["identical"] and layer5c["dt_rule_kept_identical"]
    layer3c["identical"] = layer3c["differ"] == 0

    for lay in (layer1, layer2, layer3, layer4, layer5, layer6):
        lay["identical"] = (lay.get("token_list_differ", 0) == 0
                            and lay.get("docs_with_different_shingle_set", 0) == 0
                            and lay.get("differ", 0) == 0
                            and lay.get("only_in_dt", 0) == 0
                            and lay.get("only_in_ours", 0) == 0
                            and lay.get("dt_rule_kept_identical", True)
                            and lay.get("filter_matches_remove_file", True))

    shared = shared_signature_test(off, flat, sig112)

    def file_sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    import datatrove
    mh = os.path.join(os.path.dirname(datatrove.__file__), "pipeline/dedup/minhash.py")
    dist = importlib.metadata.distribution("datatrove")
    doc = dict(
        version=1,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        provenance="independently rerun (this run)",
        datatrove_version=dist.version,
        datatrove_commit=None,
        datatrove_minhash_py_sha256=file_sha(mh),
        python_xxhash_version=importlib.metadata.version("xxhash"),
        datatrove_xxhash_workaround="xxh64_intdigest(text.encode('utf-8')) because xxhash>=3 rejects str; value identical",
        corpus=dict(name="fineweb sample-10BT 000_00000 rows 0..%d plus %d synthetic near-duplicates/exact copies"
                         % (a.n - 1, a.perturb),
                    n_docs=n, synthetic=dict(n=a.perturb, p_drop_cycle=[0.0, 0.0, 0.02, 0.05, 0.1])),
        config=dict(n_grams=5, num_buckets=14, hashes_per_bucket=8, seed=1, precision=64,
                    tokenizer=str(type(st1.word_tokenizer).__name__),
                    norm="DEF_TEXT_NORM_CONFIG (weekdays/months off)"),
        timings=dict(stage1_s=t_stage1, stage2_s=t_stage2, stage3_s=t_stage3,
                     our_shingle_s=t_shingle, our_sig_s=t_sig, total_s=time.time() - t0),
        layers=dict(layer1_tokenization=layer1, layer2_shingles=layer2,
                    layer3_signatures=layer3, layer4_candidate_edges=layer4,
                    layer5_components=layer5, layer6_kept_ids=layer6),
        conditional_on_dt_tokens=dict(
            note="our layer 2-6 code run on datatrove's own token stream; identical means the only divergence is the word tokenizer",
            layer3_signatures=layer3c, layer4_candidate_edges=layer4c,
            layer5_components_and_kept=layer5c),
        shared_signature=shared,
    )
    with open(os.path.join(OUT, "parity.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps({k: v.get("identical") for k, v in doc["layers"].items()}, indent=1))
    print(json.dumps({k: v.get("identical") for k, v in doc["conditional_on_dt_tokens"].items()
                      if isinstance(v, dict)}, indent=1))
    print(json.dumps(dict(shared={k: v["signature_prefix_equal"] for k, v in shared.items()}), indent=1))
    print("wrote", os.path.join(OUT, "parity.json"))


if __name__ == "__main__":
    main()
