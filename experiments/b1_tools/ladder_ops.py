#!/usr/bin/env python3
"""Cost of A, streaming B and exact W on the scale-ladder rungs, seed 1, tau 0.8, at document level
(no collapse), each in a fresh process so that peak RSS is the operator's own.

keys    loads the rung's shingles, builds the seed-1 signatures and band keys (timed), saves the
        keys, and runs A (candidate chain + min-id union-find, timed after signatures)
stream  code/operator/stream_b.py; the deletion mask is compared with B's (runs/<rung>/remB_s1.npy,
        written by sig_verify.py)
W       code/operator/exact_w.py
Writes results/ladder_ops.jsonl.
Usage: env/bin/python ladder_ops.py <rung N> [kinds]
"""
import json
import os
import resource
import subprocess
import sys
import time

W = os.environ.get("B2_W", "/path/to/scratch/dedupjoin")
PKG = f"{W}/dedupjoin-newsnap-selection"


def peak():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)


def worker(kind, name):
    import numpy as np
    sys.path[:0] = [f"{PKG}/code/operator", f"{PKG}/code/run"]
    d = f"{W}/runs/{name}"
    t0 = time.time()
    with np.load(f"{d}/snapshot.npz") as z:
        off, flat = z["off"].astype(np.int64), z["flat"]
    valid = np.diff(off) > 0
    n = len(off) - 1
    out = dict(corpus=name, kind=kind, n=n, t_load_s=round(time.time() - t0, 1))
    if kind == "keys":
        from stream_b import candidate_keys
        import core_lib as K
        K.set_threads(K.THREADS)
        t0 = time.time()
        keys = candidate_keys(off, flat, 1)
        out["t_signature_s"] = round(time.time() - t0, 1)
        np.save(f"{d}/keys_s1.npy", keys)
        t0 = time.time()
        ei, ej, _ = K.chain_edges(keys, valid)
        root = K.uf_minid(n, ei, ej)
        out["t_after_signature_s"] = round(time.time() - t0, 1)
        out["deleted"] = int(((root != np.arange(n)) & valid).sum())
    else:
        keys = np.load(f"{d}/keys_s1.npy")
        t0 = time.time()
        if kind == "stream":
            from stream_b import stream_b
            r = stream_b(off, flat, keys, 0.8, valid=valid)
            out.update(enumerated_pairs=r["enumerated_pairs"], verified_pairs=r["verified_pairs"])
        else:
            from exact_w import exact_w
            r = exact_w(off, flat, keys, 0.8, valid=valid)
            out.update(enumerated_pairs=r["enumerated_pairs"], verified_pairs=r["verified_pairs"])
        out["t_after_signature_s"] = round(time.time() - t0, 1)
        out["deleted"] = int(r["deleted"].sum())
        if kind == "stream" and os.path.exists(f"{d}/remB_s1.npy"):
            ref = np.load(f"{d}/remB_s1.npy")
            out["equals_B"] = bool((ref == r["deleted"]).all())
    out["peak_rss_gib"] = peak()
    print("RESULT " + json.dumps(out), flush=True)


def main():
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    name = f"rung_{int(sys.argv[1])}"
    kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["keys", "stream", "W"]
    for kind in kinds:
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "--worker", kind, name], capture_output=True, text=True)
        res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        row = json.loads(res[-1][7:]) if res else dict(corpus=name, kind=kind, status="failed",
                                                       error=p.stderr.strip()[-400:])
        row.setdefault("status", "ok")
        row["wall_s"] = round(time.time() - t0, 1)
        with open(f"{W}/results/ladder_ops.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    print(f"{name} LADDER_OPS_DONE", flush=True)


if __name__ == "__main__":
    main()
