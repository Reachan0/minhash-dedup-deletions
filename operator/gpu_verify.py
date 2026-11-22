#!/usr/bin/env python3
"""GPU verification of within-bucket pairs (numba.cuda), three modes:

materialised  every within-bucket pair is tested with exact Jaccard (path B as usually implemented,
              without materialising the pair list); accepted pairs are unioned.
signature     every within-bucket pair is tested by the share of equal MinHash signature values
              (SEDD-style); accepted pairs are unioned.
stream        streaming B on the GPU: a star pass over every bucket, then the remaining pairs of the
              buckets the star pass left split; a pair is tested only if its endpoints are not yet
              connected. Unions are lock-free (the larger root is hooked to the smaller with CAS), so
              the components, and B's min-id deletion set, do not depend on the order in which
              threads run: a skipped pair already lies inside one component of true edges.
Each thread walks a contiguous range of pair indices; buckets are found by one binary search per range.
Usage (library): gpu_b(off, flat, keys, tau, mode, valid=None, sig=None) -> dict
"""
import math
import time

import numpy as np
from numba import cuda, int64, int32

THREADS = 256


@cuda.jit(device=True)
def _find(parent, x):
    while True:
        p = parent[x]
        if p == x:
            return x
        gp = parent[p]
        if gp != p:
            parent[x] = gp  # path halving; parents only ever move to smaller roots
        x = p


@cuda.jit(device=True)
def _union(parent, a, b):
    while True:
        ra = _find(parent, a)
        rb = _find(parent, b)
        if ra == rb:
            return
        lo = ra if ra < rb else rb
        hi = rb if ra < rb else ra
        if cuda.atomic.cas(parent, hi, hi, lo) == hi:
            return


@cuda.jit(device=True)
def _jaccard_ok(off, flat, a, b, tau):
    i = off[a]
    ea = off[a + 1]
    j = off[b]
    eb = off[b + 1]
    na = ea - i
    nb = eb - j
    common = 0
    while i < ea and j < eb:
        x = flat[i]
        y = flat[j]
        if x == y:
            common += 1
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    den = na + nb - common
    if den == 0:
        return False
    return common / den + 1e-15 >= tau


@cuda.jit(device=True)
def _sig_ok(sig, a, b, tau):
    h = sig.shape[1]
    eq = 0
    for k in range(h):
        if sig[a, k] == sig[b, k]:
            eq += 1
    return eq / h + 1e-12 >= tau


@cuda.jit(device=True)
def _bucket_of(cum, p):
    lo = 0
    hi = cum.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= p:
            lo = mid
        else:
            hi = mid
    return lo


@cuda.jit
def _pairs_kernel(bstart, bsize, skip_first, cum, total, docs, off, flat, sig, parent, tau, mode, chunk, tested):
    """Pairs (i, j), skip_first <= i < j < m, of every listed bucket; global pair index in [0, total).
    mode 0: exact Jaccard, always tested; 1: signature, always tested; 2: exact Jaccard with skip."""
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    p = int64(t) * chunk
    if p >= total:
        return
    end = p + chunk
    if end > total:
        end = total
    b = _bucket_of(cum, p)
    # position (i, j) of pair p - cum[b] inside bucket b, members skip_first..m-1
    q = p - cum[b]
    m = int64(bsize[b]) - skip_first
    i = int64(0)
    row = m - 1
    while q >= row:
        q -= row
        i += 1
        row -= 1
    j = i + 1 + q
    n_tested = int64(0)
    while p < end:
        base = bstart[b] + skip_first
        a = docs[base + i]
        c = docs[base + j]
        do_test = True
        if mode == 2:
            if _find(parent, a) == _find(parent, c):
                do_test = False
        if do_test:
            n_tested += 1
            ok = _sig_ok(sig, a, c, tau) if mode == 1 else _jaccard_ok(off, flat, a, c, tau)
            if ok:
                _union(parent, a, c)
        p += 1
        j += 1
        if j == m:
            i += 1
            j = i + 1
            if j >= m:
                b += 1
                while b < bsize.shape[0] and int64(bsize[b]) - skip_first < 2:
                    b += 1
                if b >= bsize.shape[0]:
                    break
                m = int64(bsize[b]) - skip_first
                i = int64(0)
                j = int64(1)
    cuda.atomic.add(tested, 0, n_tested)


@cuda.jit
def _star_kernel(bstart, bsize, cum, total, docs, off, flat, parent, tau, chunk, tested):
    """Star pass: pairs (first member, member j), j >= 1, with the connectivity skip."""
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    p = int64(t) * chunk
    if p >= total:
        return
    end = p + chunk
    if end > total:
        end = total
    b = _bucket_of(cum, p)
    j = p - cum[b] + 1
    n_tested = int64(0)
    while p < end:
        a = docs[bstart[b]]
        c = docs[bstart[b] + j]
        if _find(parent, a) != _find(parent, c):
            n_tested += 1
            if _jaccard_ok(off, flat, a, c, tau):
                _union(parent, a, c)
        p += 1
        j += 1
        if j == bsize[b]:
            b += 1
            j = int64(1)
            if b >= bsize.shape[0]:
                break
    cuda.atomic.add(tested, 0, n_tested)


@cuda.jit
def _split_kernel(bstart, bsize, docs, parent, split):
    """split[b] = 1 if some member of bucket b is not in its first member's component."""
    b = cuda.blockIdx.x
    if b >= bsize.shape[0]:
        return
    r0 = _find(parent, docs[bstart[b]])
    k = cuda.threadIdx.x + 1
    while k < bsize[b]:
        if _find(parent, docs[bstart[b] + k]) != r0:
            split[b] = 1
        k += cuda.blockDim.x


@cuda.jit
def _flatten_kernel(parent):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if x < parent.shape[0]:
        parent[x] = _find(parent, x)


def buckets(keys, valid):
    """All buckets of size >= 2 over every band: members in identifier order."""
    ids = np.flatnonzero(valid).astype(np.int32)
    docs, sizes = [], []
    for band in range(keys.shape[0]):
        o = np.argsort(keys[band, ids], kind="stable")
        d = ids[o]
        v = keys[band, d]
        st = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
        sz = np.diff(np.r_[st, len(d)])
        keep = sz >= 2
        mask = np.repeat(keep, sz)
        docs.append(d[mask])
        sizes.append(sz[keep])
    docs = np.concatenate(docs).astype(np.int32)
    sizes = np.concatenate(sizes).astype(np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    return docs, starts, sizes


def _launch_pairs(d, starts, sizes, skip_first, docs_d, off_d, flat_d, sig_d, parent_d, tau, mode, threads_total):
    m = sizes - skip_first
    npairs = np.where(m >= 2, m * (m - 1) // 2, 0).astype(np.int64)
    sel = npairs > 0
    if not sel.any():
        return 0, 0
    st, sz, npairs = starts[sel], sizes[sel], npairs[sel]
    cum = np.r_[0, np.cumsum(npairs)].astype(np.int64)
    total = int(cum[-1])
    chunk = max(1, math.ceil(total / threads_total))
    nthreads = math.ceil(total / chunk)
    tested = cuda.to_device(np.zeros(1, np.int64))
    _pairs_kernel[math.ceil(nthreads / THREADS), THREADS](
        cuda.to_device(st), cuda.to_device(sz), int64(skip_first), cuda.to_device(cum), int64(total),
        docs_d, off_d, flat_d, sig_d, parent_d, float(tau), int32(mode), int64(chunk), tested)
    cuda.synchronize()
    return total, int(tested.copy_to_host()[0])


def gpu_b(off, flat, keys, tau, mode, valid=None, sig=None, threads_total=1 << 21):
    off = np.asarray(off, np.int64)
    n = len(off) - 1
    if valid is None:
        valid = off[1:] > off[:-1]
    t0 = time.perf_counter()
    docs, starts, sizes = buckets(keys, valid)
    t_buckets = time.perf_counter() - t0
    t0 = time.perf_counter()
    off_d = cuda.to_device(off)
    flat_d = cuda.to_device(np.asarray(flat, np.uint64))
    docs_d = cuda.to_device(docs)
    sig_d = cuda.to_device(np.asarray(sig, np.uint64)) if sig is not None else cuda.to_device(np.zeros((1, 1), np.uint64))
    parent_d = cuda.to_device(np.arange(n, dtype=np.int32))
    cuda.synchronize()
    t_copy = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = dict(mode=mode, n=n, buckets=int(len(sizes)), largest_bucket=int(sizes.max()) if len(sizes) else 0,
               within_bucket_pairs=int((sizes * (sizes - 1) // 2).sum()))
    if mode in ("materialised", "signature"):
        enum, tested = _launch_pairs(None, starts, sizes, 0, docs_d, off_d, flat_d, sig_d, parent_d, tau,
                                     0 if mode == "materialised" else 1, threads_total)
        out.update(enumerated=enum, tested=tested)
    else:
        nstar = (sizes - 1).astype(np.int64)
        cum = np.r_[0, np.cumsum(nstar)].astype(np.int64)
        total = int(cum[-1])
        chunk = max(1, math.ceil(total / threads_total))
        tested = cuda.to_device(np.zeros(1, np.int64))
        _star_kernel[math.ceil(math.ceil(total / chunk) / THREADS), THREADS](
            cuda.to_device(starts), cuda.to_device(sizes), cuda.to_device(cum), int64(total), docs_d, off_d,
            flat_d, parent_d, float(tau), int64(chunk), tested)
        split_d = cuda.to_device(np.zeros(len(sizes), np.int32))
        _split_kernel[len(sizes), THREADS](cuda.to_device(starts), cuda.to_device(sizes), docs_d, parent_d, split_d)
        cuda.synchronize()
        star_tested = int(tested.copy_to_host()[0])
        split = split_d.copy_to_host().astype(bool)
        enum, rest_tested = _launch_pairs(None, starts[split], sizes[split], 1, docs_d, off_d, flat_d, sig_d,
                                          parent_d, tau, 2, threads_total)
        out.update(star_pairs=total, split_buckets=int(split.sum()), enumerated=total + enum,
                   tested=star_tested + rest_tested)
    _flatten_kernel[math.ceil(n / THREADS), THREADS](parent_d)
    cuda.synchronize()
    t_gpu = time.perf_counter() - t0
    root = parent_d.copy_to_host().astype(np.int64)
    out.update(deleted=(valid & (root != np.arange(n))), t_buckets_s=round(t_buckets, 2), t_copy_s=round(t_copy, 2),
               t_gpu_s=round(t_gpu, 2))
    return out
