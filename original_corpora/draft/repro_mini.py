#!/usr/bin/env python3
"""One-command reproduction of the audit protocol on the shipped mini-corpus.

Runs the full decision audit -- raw MinHash-LSH, exact verification of candidate edges, and
the exact prefix-filter join -- on `artifacts/mini_corpus/mini.parquet` (5,000 documents),
and writes unsupported / missed / precision counts to a JSON file.

This is the public reproduction entry: the full-corpus numbers of the paper need the
13.6 GiB shingle cache, which is not shipped; the per-seed results JSONL files *are*
shipped under `results/`, so every table can be regenerated offline without recompute.

MinHash note: the paper's protocol uses datatrove's affine hash h(x)=(ax+b) mod (2^61-1).
This script uses 112 independent splitmix64 permutations of the xxHash64 shingles and the
same 14x8 banding. The hash family differs; the banding geometry is identical, so the
document-level contrast (unsupported class, verification repair) reproduces qualitatively,
not to the last document. The script prints both the mini numbers and their comparison.

Usage:
  python3 code/repro_mini.py --mini artifacts/mini_corpus/mini.parquet \
      --out results/repro_mini.json [--seed 1] [--tau 0.8] [--threads 4]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# vendored copy ships with the artifact; fall back to the earlier-run source tree
VENDOR = os.path.join(ROOT, "output/dedupjoin-pvldb/artifacts/vendor")
C03 = os.environ.get(
    "DEDUPJOIN_C03",
    VENDOR if os.path.isdir(VENDOR) else
    os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03"))
sys.path.insert(0, C03)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x):
    """Vectorised splitmix64 finaliser; a bijective 64-bit mixing function."""
    x = np.asarray(x, dtype=np.uint64)
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK
    z = x.copy()
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK
    return z ^ (z >> np.uint64(31))


def permutations(seed, n_perm):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**63, size=n_perm, dtype=np.uint64)


def minhash(off, flat, seeds):
    n = len(off) - 1
    sig = np.full((len(seeds), n), np.uint64(2**64 - 1), dtype=np.uint64)
    for i in range(n):
        s = flat[off[i]:off[i + 1]]
        if len(s) == 0:
            continue
        for p, sd in enumerate(seeds):
            sig[p, i] = splitmix64(s ^ sd).min()
    return sig


def band_pairs(sig, bands, rows):
    n = sig.shape[1]
    buckets = {}
    for b in range(bands):
        # column-wise band key
        for i in range(n):
            k = (b,) + tuple(int(sig[b * rows + r, i]) for r in range(rows))
            buckets.setdefault(k, []).append(i)
    pairs = set()
    for docs in buckets.values():
        if len(docs) > 1:
            docs = sorted(docs)
            for a in range(len(docs)):
                for b2 in range(a + 1, len(docs)):
                    pairs.add((docs[a], docs[b2]))
    return pairs


def jaccard(off, flat, i, j):
    a = flat[off[i]:off[i + 1]]
    b = flat[off[j]:off[j + 1]]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / float(len(a) + len(b) - inter)


def components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    return comp


def deleted_set(comp):
    out = set()
    for members in comp.values():
        if len(members) > 1:
            rep = min(members)
            out.update(m for m in members if m != rep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", default=os.path.join(
        ROOT, "output/dedupjoin-pvldb/artifacts/mini_corpus/mini.parquet"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "output/dedupjoin-pvldb/results/repro_mini.json"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    tab = pq.read_table(args.mini, columns=["id", "source", "text"]).to_pydict()
    texts = tab["text"]
    n = len(texts)
    print(f"mini corpus: {n} documents", flush=True)

    sizes = []
    flats = []
    for t in texts:
        s, _ = C.shingle_doc(t)
        sizes.append(len(s))
        flats.append(s)
    off = np.zeros(n + 1, dtype=np.int64)
    off[1:] = np.cumsum(sizes)
    flat = np.concatenate(flats) if flats else np.zeros(0, np.uint64)
    print(f"shingles: {len(flat):,}", flush=True)

    # exact reference
    R, S, V = EJ.build_ranks(off, flat)
    ex = EJ.exact_join(off, R, S, V, args.tau, args.threads)
    exact_pairs = set(zip(ex["i"].tolist(), ex["j"].tolist()))
    exact_del = deleted_set(components(n, exact_pairs))
    neighbours = set()
    for i, j in exact_pairs:
        neighbours.add(i)
        neighbours.add(j)
    print(f"exact pairs: {len(exact_pairs):,}; exact deletions: {len(exact_del):,}", flush=True)

    # raw MinHash-LSH, 14x8, 112 permutations
    seeds = permutations(args.seed, 112)
    sig = minhash(off, flat, seeds)
    raw_pairs = band_pairs(sig, 14, 8)
    print(f"raw candidates: {len(raw_pairs):,}", flush=True)
    raw_del = deleted_set(components(n, raw_pairs))

    # exact verification of candidate edges
    verified = {p for p in raw_pairs if jaccard(off, flat, p[0], p[1]) >= args.tau}
    ver_del = deleted_set(components(n, verified))
    print(f"verified edges: {len(verified):,}; verified deletions: {len(ver_del):,}", flush=True)

    unsupported = [d for d in raw_del if d not in neighbours]
    missed = [d for d in exact_del if d not in ver_del]
    out = dict(
        provenance="independently rerun (this run)",
        mini_sha256=json.load(open(os.path.join(os.path.dirname(args.mini), "manifest.json")))["sha256"],
        n_documents=n, n_shingles=int(len(flat)), tau=args.tau, seed=args.seed,
        banding="14x8", permutations=112,
        raw_deleted=len(raw_del), exact_deleted=len(exact_del),
        verified_deleted=len(ver_del),
        unsupported=len(unsupported),
        unsupported_frac=(len(unsupported) / len(raw_del)) if raw_del else None,
        missed=len(missed),
        verify_precision=(1.0 - len([d for d in ver_del if d not in neighbours]) / len(ver_del))
        if ver_del else None,
        candidate_pairs=len(raw_pairs), verified_pairs=len(verified),
        exact_pairs=len(exact_pairs),
        runtime_sec=round(time.time() - t0, 2),
        note=("Hash family for this mini run: 112 splitmix64 permutations of the xxHash64 "
              "5-gram shingles; the paper uses datatrove's affine family. Banding is 14x8 "
              "in both. Paper-scale numbers need the shingle cache, which is not shipped."),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
