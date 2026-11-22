"""MinHash signatures and datatrove-style band keys for every document in the pool.

Pipeline per document (identical to the datatrove 0.6.0 defaults used in the
parity harness): simplify text -> word 5-gram shingles (xxhash64 of the joined
n-gram) -> 112 MinHash values under the datatrove permutation family
((x*a+b) mod (2^61-1), a,b from RandomState(seed)) -> 14 bands of 8 hashes.

Only band keys are persisted (N x 14 uint64), plus a signature sample for spot
checks; a full 4M x 112 signature matrix is 3.6 GB and is not needed to label
duplicate-density strata.  Shingles themselves are streamed per chunk.

Outputs under <out>/:
  bands-XXXXX.npy        (n_chunk, 14) uint64 band keys
  sigsample.npz          signatures of a fixed random sample (for verification)
  progress.jsonl         one line per chunk (rows, seconds, rows/s)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

H = 112
B = 14
R = 8
P61 = np.uint64((1 << 61) - 1)
MAX_SHINGLES = 20000       # documented cap: only the first 20k shingles of a doc enter its sketch
SAMPLE_DOCS = 2000


# ---- text canonicalisation: datatrove simplify_text (see datatrove/utils/text.py) ----
import re
import string
import unicodedata

PUNCTUATION = "".join(map(chr, (x for a, b in ((0, 9), (11, 13), (13, 32), (127, 160))
                                for x in range(a, b))))
PUNCTUATION_TRANS = str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))
NUMBERS_PATTERN = re.compile(r"\d+([.,،٫⎖⎗⎘]{1}\d+)?")
WHITESPACE_PATTERN = re.compile(r"\s+")


def simplify_text(text):
    text = text.lower()
    text = NUMBERS_PATTERN.sub("0", text)
    text = text.translate(PUNCTUATION_TRANS)
    text = WHITESPACE_PATTERN.sub(" ", text.strip())
    if not text.isascii():
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
    return text.strip()


def shingle_doc(text, n=5):
    import xxhash
    w = simplify_text(text).split(" ")
    if len(w) < n or w == [""]:
        return np.zeros(0, dtype=np.uint64)
    h = xxhash.xxh64_intdigest
    m = len(w) - n + 1
    if m > MAX_SHINGLES:
        m = MAX_SHINGLES
    arr = np.fromiter((h(" ".join(w[i:i + n]).encode("utf-8")) for i in range(m)),
                      dtype=np.uint64, count=m)
    return np.unique(arr)


def _shingle_chunk(texts):
    out, lens = [], []
    for t in texts:
        s = shingle_doc(t)
        out.append(s)
        lens.append(len(s))
    sizes = np.array(lens, dtype=np.int64)
    flat = np.concatenate(out) if out else np.zeros(0, np.uint64)
    return sizes, flat


# ---- numba kernels (same as the parity harness lsh.py) ----
from numba import njit, prange

@njit(parallel=True, cache=True)
def sig_dt(off, flat, a, b, P):
    n = off.shape[0] - 1
    Hn = a.shape[0]
    out = np.full((n, Hn), np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64)
    for d in prange(n):
        for i in range(off[d], off[d + 1]):
            x = flat[i]
            for h in range(Hn):
                v = (x * a[h] + b[h]) % P
                if v < out[d, h]:
                    out[d, h] = v
    return out


@njit(cache=True)
def splitmix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def band_keys(sig, b, r):
    n = sig.shape[0]
    K = np.empty((b, n), np.uint64)
    for d in range(n):
        for i in range(b):
            h = np.uint64(0x12345678ABCDEF)
            for j in range(r):
                h = splitmix(h ^ sig[d, i * r + j])
            K[i, d] = h
    return K


def permutation_params(seed):
    gen = np.random.RandomState(seed)
    a = gen.randint(1, P61, dtype=np.uint64, size=(1, H))[0]
    b = gen.randint(0, P61, dtype=np.uint64, size=(1, H))[0]
    return a, b


def read_texts(pool_dir):
    """Yield (row_id array, texts) per pool shard, in global row order."""
    srcs = ["fw", "c4", "wet"]
    for src in srcs:
        d = os.path.join(pool_dir, src)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".parquet"):
                continue
            pf = pq.ParquetFile(os.path.join(d, fn))
            for bi in pf.iter_batches(batch_size=5000, columns=["text"]):
                yield bi.column("text").to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    a, bb = permutation_params(args.seed)
    rng = np.random.RandomState(12345)
    sample_rows = set(int(x) for x in rng.choice(4_500_000, size=SAMPLE_DOCS, replace=False))
    sig_sample, sample_idx = [], []
    prog = open(os.path.join(args.out, "progress.jsonl"), "a")
    pending, texts_buf = [], []
    row = 0
    chunk_no = 0
    t0 = time.time()
    ex = ProcessPoolExecutor(args.workers)

    def process(buf):
        nonlocal chunk_no, sig_sample, sample_idx
        row_start = row - len(buf)
        tc = time.time()
        parts = [buf[i:i + 2000] for i in range(0, len(buf), 2000)]
        res = list(ex.map(_shingle_chunk, parts))
        sizes = np.concatenate([r[0] for r in res])
        flat = np.concatenate([r[1] for r in res]) if res else np.zeros(0, np.uint64)
        off = np.zeros(len(sizes) + 1, np.int64)
        np.cumsum(sizes, out=off[1:])
        del res, parts
        sig = sig_dt(off, flat, a, bb, P61)
        del off, flat
        keys = band_keys(sig, B, R).T.copy()   # (n, 14)
        np.save(os.path.join(args.out, f"bands-{chunk_no:05d}.npy"), keys)
        n = sig.shape[0]
        local = [r - row_start for r in sample_rows if row_start <= r < row_start + n]
        if local:
            sig_sample.append(sig[local])
            sample_idx.extend([row_start + i for i in local])
        del sig, keys
        prog.write(json.dumps({"chunk": chunk_no, "row_start": row_start, "n": n,
                               "seconds": round(time.time() - tc, 1)}) + "\n")
        prog.flush()
        print(f"chunk {chunk_no}: rows {row_start}..{row_start+n} in {time.time()-tc:.1f}s", flush=True)
        chunk_no += 1

    for texts in read_texts(args.pool):
        texts_buf.extend(texts)
        row += len(texts)
        if len(texts_buf) >= args.chunk:
            process(texts_buf)
            texts_buf = []
        if args.max_docs and row >= args.max_docs:
            break
    if texts_buf:
        process(texts_buf)
    ex.shutdown()
    np.savez(os.path.join(args.out, "sigsample.npz"),
             sig=np.concatenate(sig_sample) if sig_sample else np.zeros((0, H), np.uint64),
             row_id=np.array(sample_idx, np.int64),
             a=a, b=bb, seed=np.array([args.seed]))
    print(json.dumps({"rows": row, "chunks": chunk_no, "seconds": round(time.time() - t0, 1)}))
    prog.close()


if __name__ == "__main__":
    main()
