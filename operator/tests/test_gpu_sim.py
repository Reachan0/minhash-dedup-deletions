"""Correctness of gpu_verify.py under the numba CUDA simulator (NUMBA_ENABLE_CUDASIM=1):
materialised and streaming GPU B must equal literal B, and the signature mode must equal
B with the signature test, on random small inputs."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gpu_verify import gpu_b  # noqa: E402


def literal(off, flat, keys, tau, sig=None):
    n = len(off) - 1
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for band in range(keys.shape[0]):
        for k in np.unique(keys[band]):
            d = np.flatnonzero(keys[band] == k)
            for i in range(len(d)):
                for j in range(i + 1, len(d)):
                    a, b = d[i], d[j]
                    if sig is None:
                        sa, sb = set(flat[off[a]:off[a + 1]].tolist()), set(flat[off[b]:off[b + 1]].tolist())
                        ok = bool(sa | sb) and len(sa & sb) / len(sa | sb) + 1e-15 >= tau
                    else:
                        ok = (sig[a] == sig[b]).mean() + 1e-12 >= tau
                    if ok:
                        ra, rb = find(a), find(b)
                        parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(x) for x in range(n)])
    return (root != np.arange(n)) & (off[1:] > off[:-1])


def main():
    rng = np.random.default_rng(1)
    for trial in range(6):
        n = int(rng.integers(10, 40))
        bases = [rng.choice(200, 15, replace=False) for _ in range(3)]
        sets = []
        for d in range(n):
            if rng.random() < 0.2:
                s = rng.choice(np.arange(5000, 9000), 15, replace=False)
            else:
                s = bases[int(rng.integers(0, 3))].copy()
                s[rng.choice(15, int(rng.integers(0, 4)), replace=False)] = rng.integers(1000, 4000)
            sets.append(np.unique(s))
        flat = np.concatenate(sets).astype(np.uint64)
        off = np.r_[0, np.cumsum([len(s) for s in sets])].astype(np.int64)
        keys = rng.integers(0, 3, size=(2, n)).astype(np.uint64)
        sig = rng.integers(0, 3, size=(n, 8)).astype(np.uint64)
        ref = literal(off, flat, keys, 0.8)
        for mode in ("materialised", "stream"):
            for tt in (1, 3, 16):
                r = gpu_b(off, flat, keys, 0.8, mode, threads_total=tt)
                assert (r["deleted"] == ref).all(), (trial, mode, tt)
        rs = literal(off, flat, keys, 0.5, sig=sig)
        r = gpu_b(off, flat, keys, 0.5, "signature", sig=sig, threads_total=5)
        assert (r["deleted"] == rs).all(), (trial, "signature")
        print("trial", trial, "ok", flush=True)
    print("GPU_SIM_OK")


if __name__ == "__main__":
    main()
