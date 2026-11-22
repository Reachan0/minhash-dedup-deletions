"""Correctness of the MUSA/CUDA build (gpu_native.py) on random small inputs against literal B,
and of the signature mode against B with the signature test. Usage: python test_gpu_native.py <lib.so>"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import gpu_native as G  # noqa: E402
from test_gpu_sim import literal  # noqa: E402


def main():
    lib = G._lib(sys.argv[1])
    if len(sys.argv) > 2:
        import ctypes
        lib.gpu_set_span.argtypes = [ctypes.c_longlong]
        lib.gpu_set_span(int(sys.argv[2]))  # small launches exercise the batching
    rng = np.random.default_rng(2)
    for trial in range(200):
        n = int(rng.integers(10, 300))
        bases = [rng.choice(400, 25, replace=False) for _ in range(4)]
        sets = []
        for d in range(n):
            if rng.random() < 0.2:
                s = rng.choice(np.arange(5000, 9000), 25, replace=False)
            else:
                s = bases[int(rng.integers(0, 4))].copy()
                s[rng.choice(25, int(rng.integers(0, 6)), replace=False)] = rng.integers(1000, 4000)
            sets.append(np.unique(s))
        flat = np.concatenate(sets).astype(np.uint64)
        off = np.r_[0, np.cumsum([len(s) for s in sets])].astype(np.int64)
        keys = rng.integers(0, int(rng.integers(1, 6)), size=(3, n)).astype(np.uint64)
        sig = rng.integers(0, 3, size=(n, 8)).astype(np.uint64)
        ref = literal(off, flat, keys, 0.8)
        for mode in ("materialised", "stream"):
            for tt in (1, 7, 1 << 12):
                r = G.gpu_b(off, flat, keys, 0.8, mode, threads_total=tt, lib=lib)
                assert (r["deleted"] == ref).all(), (trial, mode, tt)
        rs = literal(off, flat, keys, 0.5, sig=sig)
        r = G.gpu_b(off, flat, keys, 0.5, "signature", sig=sig, lib=lib)
        assert (r["deleted"] == rs).all(), (trial, "signature")
    print("GPU_NATIVE_OK")


if __name__ == "__main__":
    main()
