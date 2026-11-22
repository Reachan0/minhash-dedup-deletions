#!/usr/bin/env python3
"""Python wrapper of gpu_verify_kernels.mu (MUSA or CUDA build, libgpuverify.so next to this file).
Same interface and output as gpu_verify.gpu_b; the buckets are built on the CPU by gpu_verify.buckets."""
import ctypes
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {"materialised": 0, "signature": 1, "stream": 2}


def buckets(keys, valid):
    ids = np.flatnonzero(valid).astype(np.int32)
    docs, sizes = [], []
    for band in range(keys.shape[0]):
        o = np.argsort(keys[band, ids], kind="stable")
        d = ids[o]
        v = keys[band, d]
        st = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
        sz = np.diff(np.r_[st, len(d)])
        keep = sz >= 2
        docs.append(d[np.repeat(keep, sz)])
        sizes.append(sz[keep])
    docs = np.concatenate(docs).astype(np.int32)
    sizes = np.concatenate(sizes).astype(np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    return docs, starts, sizes


def _lib(path=None):
    lib = ctypes.CDLL(path or os.path.join(HERE, "libgpuverify.so"))
    p = np.ctypeslib.ndpointer
    lib.gpu_b_run.restype = ctypes.c_int
    lib.gpu_b_run.argtypes = [ctypes.c_longlong, p(np.int64), ctypes.c_longlong, p(np.uint64), ctypes.c_longlong,
                              p(np.int64), p(np.int64), ctypes.c_longlong, p(np.int32), p(np.int32), p(np.uint64),
                              ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_longlong, p(np.int32),
                              p(np.int64), p(np.float64)]
    return lib


def gpu_b(off, flat, keys, tau, mode, valid=None, sig=None, threads_total=1 << 21, lib=None):
    lib = lib or _lib()
    off = np.ascontiguousarray(off, np.int64)
    flat = np.ascontiguousarray(flat, np.uint64)
    n = len(off) - 1
    if valid is None:
        valid = off[1:] > off[:-1]
    t0 = time.perf_counter()
    docs, starts, sizes = buckets(keys, valid)
    slot_bucket = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
    t_buckets = time.perf_counter() - t0
    H = 0 if sig is None else sig.shape[1]
    sig = np.ascontiguousarray(sig if sig is not None else np.zeros((1, 1)), np.uint64)
    parent = np.empty(n, np.int32)
    stats = np.zeros(4, np.int64)
    times = np.zeros(2, np.float64)
    rc = lib.gpu_b_run(n, off, int(off[-1]), flat, len(sizes), starts, sizes, len(docs), docs, slot_bucket, sig, H,
                       float(tau), MODES[mode], int(threads_total), parent, stats, times)
    if rc != 0:
        raise RuntimeError("gpu_b_run failed")
    root = parent.astype(np.int64)
    out = dict(mode=mode, n=n, buckets=int(len(sizes)), largest_bucket=int(sizes.max()) if len(sizes) else 0,
               within_bucket_pairs=int((sizes * (sizes - 1) // 2).sum()), enumerated=int(stats[0]), tested=int(stats[1]),
               deleted=valid & (root != np.arange(n)), t_buckets_s=round(t_buckets, 2), t_copy_s=round(times[0], 2),
               t_gpu_s=round(times[1], 2))
    if mode == "stream":
        out.update(star_pairs=int(stats[2]), split_buckets=int(stats[3]))
    return out
