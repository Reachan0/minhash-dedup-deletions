#!/usr/bin/env python3
"""B2 cost runs: literal, streaming and capped B, and exact W, on the injected 1M-document variants.

For each corpus in b2/ (base_1m, exact_<m>, near_<m>), seed 1, tau 0.8:
  literal B   all within-bucket pairs materialised (core_lib.all_bucket_pairs + exact Jaccard),
              in a subprocess limited to 24 GiB of address space; failure is recorded;
  stream B    code/operator/stream_b.py (union-find skip, no materialised pairs, no collapse);
  cap q       stream_b_cap with q in {100, 1000}; extra retained documents vs stream B and the
              bound U (unchecked candidate pairs). Rows in b2_runs.jsonl count the star prefix
              against the cap; `b2_runs.py cap ...` writes b2_cap.jsonl with the prefix exempt;
  exact W     code/operator/exact_w.py.
Each run is a fresh subprocess with a 1-hour limit; peak RSS and wall time are recorded.
Usage: env/bin/python b2_runs.py [variant ...] | cap [variant ...]
"""
import json
import os
import resource
import subprocess
import sys
import time

W = os.environ.get("B2_W", "/path/to/scratch/dedupjoin")
PKG = f"{W}/dedupjoin-newsnap-selection"
LIMIT_S = 3600


def worker(kind, path, q=None, cap_prefix=True):
    import numpy as np
    sys.path[:0] = [f"{PKG}/code/operator", f"{PKG}/code/core"]
    from stream_b import candidate_keys, stream_b, stream_b_cap
    with np.load(path) as z:
        off, flat = z["off"].astype(np.int64), z["flat"]
    valid = np.diff(off) > 0
    n = len(off) - 1
    t0 = time.time()
    keys = candidate_keys(off, flat, 1)
    t_keys = time.time() - t0
    t1 = time.time()
    out = dict(n=n, t_keys_s=round(t_keys, 1))
    if kind == "literal":
        import core_lib as K
        pi, pj, st = K.all_bucket_pairs(keys, valid, n)
        jac = K.L.jaccard_pairs(off, flat, pi, pj)
        ver = jac >= 0.8 - 1e-12
        root = K.uf_minid(n, pi[ver], pj[ver])
        out.update(deleted=int(((root != np.arange(n)) & valid).sum()), pairs=int(len(pi)),
                   verified_edges=int(ver.sum()))
    elif kind == "stream":
        r = stream_b(off, flat, keys, 0.8, valid=valid)
        out.update(deleted=int(r["deleted"].sum()), **{k: r[k] for k in
                   ("enumerated_pairs", "skipped_pairs", "verified_pairs", "accepted_edges")})
        np.save(f"{W}/b2/{os.path.basename(path)[:-4]}_streamB.npy", r["deleted"])
    elif kind == "cap":
        r = stream_b_cap(off, flat, keys, 0.8, q, valid=valid, cap_prefix=cap_prefix)
        ref = np.load(f"{W}/b2/{os.path.basename(path)[:-4]}_streamB.npy")
        out.update(q=q, cap_prefix=cap_prefix, deleted=int(r["deleted"].sum()), unchecked_U=r["unchecked_pairs"],
                   extra_bound=r["extra_retained_bound"],
                   verified_pairs=r["verified_pairs"], subset_of_B=bool((r["deleted"] <= ref).all()),
                   extra_retained=int(ref.sum() - r["deleted"].sum()))
    elif kind == "W":
        from exact_w import exact_w
        r = exact_w(off, flat, keys, 0.8, valid=valid)
        out.update(deleted=int(r["deleted"].sum()), enumerated_pairs=r["enumerated_pairs"],
                   verified_pairs=r["verified_pairs"])
    out["t_method_s"] = round(time.time() - t1, 1)
    out["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)
    print("RESULT " + json.dumps(out), flush=True)


def run(variant, kind, q=None, cap_prefix=True, out="b2_runs.jsonl"):
    path = f"{W}/b2/{variant}.npz"
    cmd = [sys.executable, __file__, "--worker", kind, path] + ([str(q), str(int(cap_prefix))] if q else [])
    pre = None
    if kind == "literal":
        def pre():
            resource.setrlimit(resource.RLIMIT_AS, (24 * 1024 ** 3, 24 * 1024 ** 3))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=LIMIT_S, preexec_fn=pre)
        res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        row = json.loads(res[-1][7:]) if res else dict(status="failed", exit_code=p.returncode,
                                                       error=p.stderr.strip().splitlines()[-1][-300:] if p.stderr.strip() else "")
        row.setdefault("status", "ok")
    except subprocess.TimeoutExpired:
        row = dict(status="timeout", limit_s=LIMIT_S)
    row.update(variant=variant, kind=kind, wall_s=round(time.time() - t0, 1))
    with open(f"{W}/results/{out}", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None,
               bool(int(sys.argv[5])) if len(sys.argv) > 5 else True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "cap":
        for v in sys.argv[2:]:
            for q in (100, 1000):
                run(v, "cap", q, cap_prefix=False, out="b2_cap.jsonl")
        print("B2_CAP_DONE", flush=True)
        return
    variants = sys.argv[1:] or ["base_1m", "exact_1000", "exact_10000", "exact_100000",
                                "near_1000", "near_10000", "near_100000"]
    os.makedirs(f"{W}/results", exist_ok=True)
    for v in variants:
        run(v, "stream")
        for q in (100, 1000):
            run(v, "cap", q)
        run(v, "W")
        run(v, "literal")
    print("B2_RUNS_DONE", flush=True)


if __name__ == "__main__":
    main()
