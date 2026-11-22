#!/usr/bin/env python3
"""GPU verification on the nested corpora, seed 1, tau 0.8 (code/operator/gpu_verify_kernels.mu through
gpu_native.py; MUSA or CUDA build, the architecture is picked by a warm-up run).

For each rung and mode, a fresh process loads the shingles and the seed-1 band keys saved by
ladder_ops.py (runs/<rung>/keys_s1.npy), warms up the kernels on a tiny input, and runs one mode:
  stream        GPU streaming B (star pass, split buckets, connectivity skip); compared with B's mask
  materialised  exact Jaccard on every within-bucket pair; compared with B's mask
  signature     SEDD-style signature test on every within-bucket pair (112 seed-1 signature values)
Writes results/gpu_runs.jsonl with times (bucket building on the CPU, copy to the device, GPU work),
pairs enumerated and tested, deleted count, equality with B, device name and memory.
Usage: env/bin/python gpu_runs.py <rung N> <mode,mode,...> [limit_s]
"""
import json
import os
import resource
import subprocess
import sys
import time

W = os.environ.get("B2_W", "/path/to/scratch/dedupjoin")
PKG = f"{W}/dedupjoin-newsnap-selection"


def worker(name, mode):
    import numpy as np
    sys.path[:0] = [f"{PKG}/code/operator", f"{PKG}/code/core"]
    import gpu_native as G
    d = f"{W}/runs/{name}"
    with np.load(f"{d}/snapshot.npz") as z:
        off, flat = z["off"].astype(np.int64), z["flat"]
    keys = np.load(f"{d}/keys_s1.npy")
    valid = np.diff(off) > 0
    # warm-up on a tiny input; it also picks the library built for this device
    toff = np.array([0, 3, 6, 9], np.int64)
    tflat = np.array([1, 2, 3, 1, 2, 3, 7, 8, 9], np.uint64)
    tkeys = np.zeros((2, 3), np.uint64)
    lib, libname = None, None
    for cand in [os.environ.get("GPU_LIB")] + [f"{PKG}/code/operator/libgpuverify_{a}.so" for a in ("mp_22", "mp_31", "mp_21")] + [f"{PKG}/code/operator/libgpuverify.so"]:
        if not cand or not os.path.exists(cand):
            continue
        try:
            L = G._lib(cand)
            ok = all(G.gpu_b(toff, tflat, tkeys, 0.8, m, sig=np.zeros((3, 4), np.uint64), threads_total=4, lib=L)["deleted"].tolist()
                     == ([False, True, False] if m != "signature" else [False, True, True]) for m in ("stream", "materialised", "signature"))
        except Exception:
            ok = False
        if ok:
            lib, libname = L, cand
            break
    if lib is None:
        raise SystemExit("no working GPU library")
    sig = None
    t_sig = None
    if mode == "signature":
        sys.path.insert(0, f"{PKG}/code/core/vendor")
        import lsh
        t0 = time.time()
        sig = lsh.signatures(off, np.asarray(flat, np.uint64), 112, 1, "dt")
        t_sig = round(time.time() - t0, 1)
    if os.environ.get("GPU_SPAN"):
        import ctypes
        lib.gpu_set_span.argtypes = [ctypes.c_longlong]
        lib.gpu_set_span(int(os.environ["GPU_SPAN"]))  # pairs per kernel launch
    r = G.gpu_b(off, flat, keys, 0.8, mode, valid=valid, sig=sig, lib=lib)
    out = {k: v for k, v in r.items() if k != "deleted"}
    out.update(corpus=name, deleted=int(r["deleted"].sum()), library=os.path.basename(libname),
               device=os.environ.get("GPU_NAME", ""), span=int(os.environ.get("GPU_SPAN") or 1 << 32), t_cpu_signature_s=t_sig,
               peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2))
    if mode != "signature" and os.path.exists(f"{d}/remB_s1.npy"):
        out["equals_B"] = bool((np.load(f"{d}/remB_s1.npy") == r["deleted"]).all())
    print("RESULT " + json.dumps(out), flush=True)


def main():
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    name = f"rung_{int(sys.argv[1])}"
    modes = sys.argv[2].split(",")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 7200
    for mode in modes:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, __file__, "--worker", name, mode], capture_output=True, text=True,
                               timeout=limit)
            res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            gerr = [l for l in p.stderr.splitlines() if "gpu error" in l]
            row = json.loads(res[-1][7:]) if res else dict(corpus=name, mode=mode, status="failed",
                                                           error=(gerr[0] if gerr else "") + " | " + p.stderr.strip()[-300:])
        except subprocess.TimeoutExpired:
            row = dict(corpus=name, mode=mode, status="timeout", limit_s=limit)
        row.setdefault("status", "ok")
        row["wall_s"] = round(time.time() - t0, 1)
        with open(f"{W}/results/gpu_runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    print(f"{name} GPU_DONE", flush=True)


if __name__ == "__main__":
    main()
