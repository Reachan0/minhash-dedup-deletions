#!/usr/bin/env python3
"""Worst cases for exact verification: one bucket whose members form two groups of near-duplicates
with no duplicate pair across the groups ("two"), m documents with no duplicate pair at all
("distinct"), or one group of m - 1 near-duplicates whose bucket's first member is unrelated to it ("noisy").

Each document is a sorted set of 64 shingle ids. Group g has a base set; every document replaces
2 base ids by fresh ids, so pairs within a group have J of about 0.9 and pairs across groups about 0.07.
Group membership alternates with the document id; in "distinct" every document is fresh. All m documents share one key in band 0 and
have unique keys in the other 13 bands, so the candidate set is exactly the one bucket.
For each m: streaming B (uncapped), capped B (q = 1000, two star passes exempt from the cap; kind
"cap1" uses one star pass) and exact W, each in
a fresh process with a 1-hour limit. Verified pairs, time and extra retained documents relative
to uncapped B are recorded.
Usage: env/bin/python adv_bucket.py <operator dir> <out.jsonl> <two|distinct> [m ...]
"""
import json
import os
import resource
import subprocess
import sys
import time

L, R, TAU = 64, 2, 0.8


def build(m, groups):
    import numpy as np
    rng = np.random.default_rng(7)
    base = [np.arange(0, L, dtype=np.uint64), np.arange(L - 8, 2 * L - 8, dtype=np.uint64)]
    fresh = 10 ** 6
    flat = np.empty(m * L, dtype=np.uint64)
    for d in range(m):
        if groups == "distinct" or (groups == "noisy" and d == 0):
            s = np.arange(fresh, fresh + L, dtype=np.uint64)
            fresh += L
        else:
            s = base[0 if groups == "noisy" else d % 2].copy()
            s[rng.choice(L, R, replace=False)] = np.arange(fresh, fresh + R, dtype=np.uint64)
            fresh += R
        flat[d * L:(d + 1) * L] = np.sort(s)
    off = np.arange(0, (m + 1) * L, L, dtype=np.int64)
    keys = np.empty((14, m), dtype=np.uint64)
    keys[0] = 1
    keys[1:] = (np.arange(13, dtype=np.uint64)[:, None] * np.uint64(m) + np.arange(m, dtype=np.uint64) + np.uint64(2))
    return off, flat, keys


def worker(opdir, kind, m, groups):
    import numpy as np
    sys.path.insert(0, opdir)
    from stream_b import stream_b, stream_b_cap
    from exact_w import exact_w
    off, flat, keys = build(m, groups)
    t0 = time.time()
    out = dict(m=m, kind=kind, groups=groups)
    if kind == "stream":
        r = stream_b(off, flat, keys, TAU)
        np.save(f"/tmp/adv_streamB_{groups}_{m}.npy", r["deleted"])
        out.update(deleted=int(r["deleted"].sum()), verified_pairs=r["verified_pairs"],
                   enumerated_pairs=r["enumerated_pairs"])
    elif kind in ("cap", "cap1"):
        r = stream_b_cap(off, flat, keys, TAU, 1000, stars=2 if kind == "cap" else 1)
        out.update(q=1000, deleted=int(r["deleted"].sum()), verified_pairs=r["verified_pairs"],
                   unchecked_U=r["unchecked_pairs"], extra_bound=r["extra_retained_bound"])
        try:
            ref = np.load(f"/tmp/adv_streamB_{groups}_{m}.npy")
            out["extra_retained"] = int(ref.sum() - r["deleted"].sum())
        except FileNotFoundError:
            out["extra_retained"] = int((0 if groups == "distinct" else m - 2) - r["deleted"].sum())
    elif kind == "W":
        r = exact_w(off, flat, keys, TAU)
        out.update(deleted=int(r["deleted"].sum()), verified_pairs=r["verified_pairs"])
    out["t_s"] = round(time.time() - t0, 2)
    out["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)
    print("RESULT " + json.dumps(out), flush=True)


def main():
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
        return
    opdir, outp, groups = sys.argv[1], sys.argv[2], sys.argv[3]
    kinds = os.environ.get("ADV_KINDS", "stream,cap,W").split(",")
    ms = [int(x) for x in sys.argv[4:]] or [1000, 10000, 30000, 100000]
    for m in ms:
        for kind in kinds:
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, __file__, "--worker", opdir, kind, str(m), groups],
                                   capture_output=True, text=True, timeout=3600)
                res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                row = json.loads(res[-1][7:]) if res else dict(m=m, kind=kind, groups=groups, status="failed",
                                                               error=p.stderr.strip()[-300:])
            except subprocess.TimeoutExpired:
                row = dict(m=m, kind=kind, groups=groups, status="timeout", limit_s=3600)
            row.setdefault("status", "ok")
            row["wall_s"] = round(time.time() - t0, 1)
            open(outp, "a").write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    print("ADV_DONE", flush=True)


if __name__ == "__main__":
    main()
