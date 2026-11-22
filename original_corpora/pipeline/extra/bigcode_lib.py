#!/usr/bin/env python3
"""BigCode (bigcode-dataset/near_deduplication) MinHash estimator, vectorised.

The upstream file `extra_bigcode_minhash.py` (sha256 recorded in MANIFEST) defines:
  * tokenisation   : NON_ALPHA.split(content), 5-gram joined by " ",
                     documents with fewer than 5 tokens produce no shingles
  * hashing        : SHA1 first 4 bytes, little endian (32-bit)
  * permutations   : RandomState(42), per permutation draw a in [1, 2^61-2],
                     b in [0, 2^61-2]; a is drawn before b
  * signature      : min over shingles of ((h*a + b) mod (2^61-1)) & (2^32-1)
  * banding        : optimal_param(tau, 256) minimising 0.5*FP+0.5*FN
  * clustering     : every band bucket unions each member with the bucket minimum
  * no exact Jaccard verification

Here the same permutation/banding/union semantics are applied to the *same shingle
hashes* as the core experiment (so the comparison isolates estimator/config effects),
and additionally the upstream tokenisation is available for a tokenisation-sensitivity
slice.  The vectorised kernels replace only the Python loops; the arithmetic is the
upstream arithmetic (uint64 wrap-around included).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_CODE = os.path.abspath(os.path.join(HERE, "..", "core"))
sys.path.insert(0, CORE_CODE)

import core_lib as K  # noqa: E402

MERSENNE = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)
NUM_PERM = 256


def permutations(seed=42, num_perm=NUM_PERM):
    """Exactly upstream: one RandomState; per permutation draw a then b."""
    gen = np.random.RandomState(seed)
    ab = [(gen.randint(1, MERSENNE, dtype=np.uint64),
           gen.randint(0, MERSENNE, dtype=np.uint64)) for _ in range(num_perm)]
    arr = np.array(ab, dtype=np.uint64).T
    return np.ascontiguousarray(arr[0]), np.ascontiguousarray(arr[1])


def optimal_param(threshold, num_perm=NUM_PERM, fp_weight=0.5, fn_weight=0.5):
    """Verbatim float version of upstream optimal_param (scipy quad)."""
    from scipy.integrate import quad

    def fp(t, b, r):
        return quad(lambda s: 1 - (1 - s ** float(r)) ** float(b), 0.0, t)[0]

    def fn(t, b, r):
        return quad(lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b)), t, 1.0)[0]

    best = (float("inf"), (0, 0))
    for b in range(1, num_perm + 1):
        for r in range(1, int(num_perm / b) + 1):
            e = fp_weight * fp(threshold, b, r) + fn_weight * fn(threshold, b, r)
            if e < best[0]:
                best = (e, (b, r))
    return best[1]


def _sig_kernel(off, flat, a, b, P, MASK):
    import numba as nb

    @nb.njit(parallel=True, cache=True)
    def _k(off, flat, a, b, P, MASK):
        n = off.shape[0] - 1
        H = a.shape[0]
        out = np.empty((n, H), np.uint32)
        for d in nb.prange(n):
            lo, hi = off[d], off[d + 1]
            for h in range(H):
                m = MASK
                aa = a[h]
                bb = b[h]
                for i in range(lo, hi):
                    v = ((flat[i] * aa + bb) % P) & MASK
                    if v < m:
                        m = v
                out[d, h] = m
        return out

    return _k(off, flat, a, b, P, MASK)


def signatures(off, flat, seed=42, num_perm=NUM_PERM):
    """BigCode signatures over the given shingle hashes; uint32 (n, num_perm).

    Band keys below must use the array's own itemsize: this function returns
    uint32, but the core datatrove signatures (uint64) are fed to the same
    band_groups in the synthetic stress test."""
    a, b = permutations(seed, num_perm)
    return _sig_kernel(np.asarray(off, np.int64), np.asarray(flat, np.uint64),
                       a, b, MERSENNE, MAX_HASH)


def band_groups(sig, B, R):
    """Yield (band, docs_sorted) for buckets of size >= 2, using the R-column band key."""
    n = sig.shape[0]
    for bi in range(B):
        cols = np.ascontiguousarray(sig[:, bi * R:(bi + 1) * R])
        keys = cols.view(np.dtype((np.void, R * cols.itemsize))).ravel()
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[brk[1:], n]
        sizes = ends - brk
        for gi in np.flatnonzero(sizes >= 2):
            s = int(brk[gi])
            yield bi, order[s:s + int(sizes[gi])]


def banding_stats(sig, B, R):
    """largest bucket, pairs before dedup, number of non-singleton buckets."""
    n = sig.shape[0]
    largest, mult, nbuck = 0, 0, 0
    for bi in range(B):
        cols = np.ascontiguousarray(sig[:, bi * R:(bi + 1) * R])
        keys = cols.view(np.dtype((np.void, R * cols.itemsize))).ravel()
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[brk[1:], n]
        sizes = ends - brk
        big = sizes[sizes >= 2]
        if len(big):
            largest = max(largest, int(big.max()))
            nbuck += int(len(big))
            mult += int((big.astype(np.int64) * (big - 1) // 2).sum())
    return dict(largest_bucket=largest, n_pairs_mult=mult, n_buckets_ge2=nbuck)


def bucket_pairs(sig, B, R, n):
    """All unique doc pairs sharing at least one band bucket (BigCode candidate set)."""
    codes = []
    mult, largest = 0, 0
    for bi, docs in band_groups(sig, B, R):
        s = len(docs)
        largest = max(largest, s)
        mult += s * (s - 1) // 2
        ii, jj = np.triu_indices(s, 1)
        codes.append(docs[ii].astype(np.int64) * np.int64(n) + docs[jj])
    c = np.unique(np.concatenate(codes)) if codes else np.zeros(0, np.int64)
    return c // n, c % n, dict(n_pairs_mult=int(mult), largest_bucket=int(largest),
                               n_pairs_unique=int(len(c)))


def bigcode_clusters(sig, B, R, n):
    """Union (bucket_min, member) over every band bucket; return min-id roots."""
    ei, ej = [], []
    for bi, docs in band_groups(sig, B, R):
        docs = np.sort(docs)
        m = int(docs[0])
        rest = docs[1:]
        if len(rest):
            ei.append(np.full(len(rest), m, np.int64))
            ej.append(rest.astype(np.int64))
    if ei:
        ei = np.concatenate(ei)
        ej = np.concatenate(ej)
    else:
        ei = np.zeros(0, np.int64)
        ej = np.zeros(0, np.int64)
    return K.L.uf_components(n, ei, ej), (ei, ej)


def removal_mask(root):
    return K.L.removed_set(root)
